#!/usr/bin/env python3
"""Build the tree leaf-index retrieval space for one classification dataset.

The classification instantiation does not retrieve in the foundation model's embedding
space. It trains XGBoost and CatBoost on train+val, reads off which leaf each sample lands
in, concatenates the two leaf-index vectors, standardises them against the training split
and projects to 64 dimensions with a truncated SVD fitted on the training split alone.
That projection is the space the kNN correction searches.

Writes `cache/tree_leaf/<dataset>/`:

    leaf_embeddings_{train,val,test}.npy    the 64-dimensional retrieval space
    pca_components.npy, train_mean.npy, train_std.npy   the fitted projection

XGBoost hyperparameters come from `analysis/<dataset>/xgb-tuning.json` where that record
is present, and from library defaults otherwise. Either way the script names the source it
used, so the provenance of a rebuilt space is in the log.

Usage
    python -u scripts/build_leaf_embeddings.py --dataset homesite-insurance
    python -u scripts/build_leaf_embeddings.py --dataset ecom-offers --dry-run

Run it from the TALENT checkout, so that `tabred/data/` and `cache/` resolve. Needs a GPU;
pass --cpu to train on the processor instead.
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import TruncatedSVD

TUNED_PARAMS = ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "scale_pos_weight")


def load_features(dataset, root="tabred/data"):
    """Numeric and binary blocks as-is; categorical columns ordinal-encoded per column."""
    d = os.path.join(root, dataset)
    parts = [np.load(os.path.join(d, f)).astype(np.float32)
             for f in ("X_num.npy", "X_bin.npy") if os.path.exists(os.path.join(d, f))]
    X = np.concatenate(parts, axis=1) if parts else None
    cat_path = os.path.join(d, "X_cat.npy")
    if os.path.exists(cat_path):
        raw = np.load(cat_path)
        enc = np.zeros(raw.shape, dtype=np.float32)
        for col in range(raw.shape[1]):
            uniq = {v: i for i, v in enumerate(np.unique(raw[:, col]))}
            enc[:, col] = [uniq.get(v, -1) for v in raw[:, col]]
        X = enc if X is None else np.concatenate([X, enc], axis=1)
    if X is None:
        raise SystemExit(f"no feature files under {d}")
    return X


def tuned_xgb_params(dataset, analysis_root="analysis"):
    p = os.path.join(analysis_root, dataset, "xgb-tuning.json")
    if not os.path.exists(p):
        print(f"  no tuning record at {p}; using library defaults for XGBoost")
        return {}
    best = json.load(open(p)).get("best_params", {})
    kept = {k: v for k, v in best.items() if k in TUNED_PARAMS}
    print(f"  XGBoost hyperparameters from {p}: {sorted(kept)}")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default="tabred/data")
    ap.add_argument("--analysis-root", default="analysis")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the data and report shapes without training")
    args = ap.parse_args()

    out_dir = os.path.join(args.out, "tree_leaf", args.dataset)
    if os.path.exists(os.path.join(out_dir, "leaf_embeddings_test.npy")) and not args.overwrite:
        print(f"{out_dir} already built, skipping (use --overwrite to rebuild)")
        return

    d = os.path.join(args.data_root, args.dataset)
    Y = np.load(os.path.join(d, "Y.npy")).astype(np.float32)
    idx = {s: np.load(os.path.join(d, "split-default", f"{s}_idx.npy"))
           for s in ("train", "val", "test")}
    X = load_features(args.dataset, args.data_root)
    Xs = {s: X[i] for s, i in idx.items()}

    print(f"dataset {args.dataset}   features {X.shape[1]}   "
          + "  ".join(f"{s} {len(idx[s])}" for s in ("train", "val", "test")))
    print(f"  positives in train: {Y[idx['train']].mean():.4f}")
    if args.dry_run:
        print(f"  would write {args.dim}-dimensional embeddings to {out_dir}/")
        tuned_xgb_params(args.dataset, args.analysis_root)
        return

    device = "cpu" if args.cpu else "cuda"
    X_fit = np.concatenate([Xs["train"], Xs["val"]])
    Y_fit = np.concatenate([Y[idx["train"]], Y[idx["val"]]])

    import xgboost as xgb
    from catboost import CatBoostClassifier

    print("\n  training XGBoost ...", flush=True)
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                                  tree_method="hist", device=device, n_estimators=3000,
                                  random_state=42, verbosity=0,
                                  **tuned_xgb_params(args.dataset, args.analysis_root))
    xgb_model.fit(X_fit, Y_fit)
    xgb_leaves = {s: xgb_model.apply(Xs[s]) for s in ("train", "val", "test")}
    print(f"  XGBoost trees: {xgb_leaves['train'].shape[1]}", flush=True)
    del xgb_model

    print("  training CatBoost ...", flush=True)
    cb_model = CatBoostClassifier(iterations=2000, depth=8, learning_rate=0.05,
                                  loss_function="Logloss", eval_metric="AUC",
                                  task_type="GPU" if device == "cuda" else "CPU",
                                  verbose=0, random_seed=42)
    cb_model.fit(X_fit, Y_fit)
    cb_leaves = {s: cb_model.calc_leaf_indexes(Xs[s]) for s in ("train", "val", "test")}
    print(f"  CatBoost trees: {cb_leaves['train'].shape[1]}", flush=True)
    del cb_model

    leaves = {s: np.concatenate([xgb_leaves[s], cb_leaves[s]], axis=1).astype(np.float32)
              for s in ("train", "val", "test")}
    train_mean = leaves["train"].mean(0)
    train_std = leaves["train"].std(0) + 1e-8
    leaves = {s: (v - train_mean) / train_std for s, v in leaves.items()}
    print(f"  combined leaf features: {leaves['train'].shape[1]}", flush=True)

    n_components = min(args.dim, leaves["train"].shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42).fit(leaves["train"])
    print(f"  SVD to {n_components} dims, explained variance "
          f"{svd.explained_variance_ratio_.sum():.4f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        emb = svd.transform(leaves[s]).astype(np.float32)
        np.save(os.path.join(out_dir, f"leaf_embeddings_{s}.npy"), emb)
        print(f"  {s:5s} {emb.shape}", flush=True)
    np.save(os.path.join(out_dir, "pca_components.npy"), svd.components_.astype(np.float32))
    np.save(os.path.join(out_dir, "train_mean.npy"), train_mean.astype(np.float32))
    np.save(os.path.join(out_dir, "train_std.npy"), train_std.astype(np.float32))
    print(f"\nwritten to {out_dir}/")


if __name__ == "__main__":
    main()
