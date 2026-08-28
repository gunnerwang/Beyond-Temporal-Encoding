#!/usr/bin/env python3
"""Build the frozen LimiX prediction and embedding cache for one dataset.

The online experiments never re-run the prior: its predictions and embeddings are computed
once per dataset and read from `cache/limix/` afterwards. This script is what writes them.

Each prior variant is a separate run of the same model over the same data, differing only
in how the input is encoded in time and in the seed used to draw the in-context set:

    base        temporal policy `indices`, seed 0    -> cache/limix/<dataset>/
    ts          temporal policy `time_series`        -> cache/limix/<dataset>-ts/
    tn          temporal policy `time_num`           -> cache/limix/<dataset>-tn/
    seed1..3    temporal policy `indices`, seed 1..3 -> cache/limix/<dataset>-seed<n>/
    ts-seed1/2  temporal policy `time_series`, seed 1/2

`configs/<dataset>/adaptive_best.json` lists under `ensemble_weights` which variants that
dataset's pool actually uses; building more than those is wasted GPU time.

Predictions are written in the normalised target space the model works in.
`scripts/prior_loader.py` converts back to raw target units when it blends.

Usage
    python -u scripts/build_limix_cache.py --dataset weather
    python -u scripts/build_limix_cache.py --dataset cooking-time --variants base,ts,ts-seed1
    python -u scripts/build_limix_cache.py --dataset weather --dry-run

Run it from the TALENT checkout that `setup.sh` produced, not from this repository: it
imports `model.lib.data` and `model.lib.slow_priors`, and writes `cache/` relative to the
working directory. Needs a GPU and the LimiX checkout described in `docs/data.md` §3.
"""
import argparse
import os
import sys

import numpy as np

# temporal policy, seed, cache-directory suffix
VARIANTS = {
    "base":     ("indices", 0, ""),
    "ts":       ("time_series", 0, "-ts"),
    "tn":       ("time_num", 0, "-tn"),
    "seed1":    ("indices", 1, "-seed1"),
    "seed2":    ("indices", 2, "-seed2"),
    "seed3":    ("indices", 3, "-seed3"),
    "ts-seed1": ("time_series", 1, "-ts-seed1"),
    "ts-seed2": ("time_series", 2, "-ts-seed2"),
}

# in-context sample count, as reported
CONTEXT_SIZE = {"weather": 10000}
DEFAULT_CONTEXT_SIZE = 3000


def dataset_args(dataset, temporal_policy, seed):
    """The loader arguments the reported runs use."""
    class Args:
        pass
    Args.dataset = dataset
    Args.dataset_path = "data"
    Args.dataset_path_tabred = "tabred/data"
    Args.cat_policy = "ohe"
    Args.num_nan_policy = "mean"
    Args.cat_nan_policy = "new"
    Args.normalization = "standard"
    Args.num_policy = "none"
    Args.cat_min_frequency = 0.0
    Args.n_bins = 2
    Args.batch_size = 1024
    Args.validate_option = "holdout_foremost_sample"
    Args.enable_timestamp = True
    Args.model_type = "three_tier_pfc_bio"
    Args.temporal_policy = temporal_policy
    Args.seed = seed
    return Args()


def build_variant(dataset, variant, ctx, splits, out_root, overwrite, dry_run):
    policy, seed, suffix = VARIANTS[variant]
    cache_dir = os.path.join(out_root, "limix", dataset + suffix)

    wanted = [s for s in splits
              if overwrite or not os.path.exists(
                  os.path.join(cache_dir, f"limix_predictions_{s}_ctx{ctx}.npy"))]
    if not wanted:
        print(f"  {variant:9s} already cached, skipping (use --overwrite to rebuild)")
        return
    if dry_run:
        print(f"  {variant:9s} policy={policy:12s} seed={seed}  ->  {cache_dir}  splits={wanted}")
        return

    import torch
    from model.lib.data import get_dataset_tabred
    from model.lib.slow_priors import LimiXSlowPrior

    print(f"\n=== {dataset} / {variant}  (policy={policy}, seed={seed}, ctx={ctx}) ===", flush=True)
    train_val_data, test_data, info = get_dataset_tabred(dataset_args(dataset, policy, seed))
    X = {"train": train_val_data[0]["train"],
         "val": train_val_data[0]["val"],
         "test": test_data[0]["test"]}
    y_train = train_val_data[3]["train"]
    print(f"  features {X['train'].shape[1]}   "
          + "  ".join(f"{s} {len(X[s])}" for s in ("train", "val", "test")), flush=True)

    prior = LimiXSlowPrior(d_in=X["train"].shape[1], d_out=1,
                           context_size=ctx, use_retrieval=False)
    np.random.seed(seed)
    prior.set_context(X["train"], y_train)

    os.makedirs(cache_dir, exist_ok=True)
    for split in wanted:
        Xs = X[split]
        prior.precompute_predictions(Xs, np.arange(len(Xs)), split=split, cache_dir=cache_dir)
        preds = np.array([prior.prediction_cache[split][i] for i in range(len(Xs))])
        embs = np.array([prior.embedding_cache[split][i] for i in range(len(Xs))])
        np.save(os.path.join(cache_dir, f"limix_predictions_{split}_ctx{ctx}.npy"), preds)
        np.save(os.path.join(cache_dir, f"limix_embeddings_{split}_ctx{ctx}.npy"), embs)
        print(f"  {split:5s} predictions {preds.shape}  embeddings {embs.shape}", flush=True)

    np.save(os.path.join(cache_dir, "y_mean.npy"), np.array(float(np.mean(y_train))))
    np.save(os.path.join(cache_dir, "y_std.npy"), np.array(float(np.std(y_train)) + 1e-8))

    del prior
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variants", default="base,ts,tn,seed1,seed2,seed3",
                    help="comma-separated; one of " + ", ".join(VARIANTS))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--context-size", type=int, default=None,
                    help="in-context sample count; defaults to 10000 for weather, 3000 elsewhere")
    ap.add_argument("--out", default="cache", help="cache root (default: cache)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild variants that are already cached")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be built without loading the model")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {', '.join(VARIANTS)}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if any(s not in ("train", "val", "test") for s in splits):
        raise SystemExit("--splits takes train, val and test")

    ctx = args.context_size or CONTEXT_SIZE.get(args.dataset, DEFAULT_CONTEXT_SIZE)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    print(f"dataset {args.dataset}   ctx {ctx}   variants {variants}   splits {splits}")
    for v in variants:
        build_variant(args.dataset, v, ctx, splits, args.out, args.overwrite, args.dry_run)
    print("\ndone.")


if __name__ == "__main__":
    main()
