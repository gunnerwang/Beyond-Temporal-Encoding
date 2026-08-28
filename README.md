# Hierarchical Model-Level Online Adaptation for Tabular Distribution Shift

Reference implementation for *Beyond Temporal Encoding: Efficient and Hierarchical
Model-Level Online Adaptation for Tabular Distribution Shift*.

A pretrained tabular prior is kept frozen and its predictions are corrected online, using
a dual-buffer nearest-neighbour memory over the prior's embedding space. Nothing is
retrained at test time; adaptation happens entirely in the correction pathway.

## What is here

Only the files written for this work. The method is a plugin for the
[TALENT](https://github.com/LAMDA-Tabular/TALENT) tabular benchmark, which is not
redistributed here — `setup.sh` clones it and installs the plugin into that checkout.
`plugin/` mirrors the paths the files have to occupy inside it; they add to TALENT and
overwrite none of it.

```
plugin/model/models/three_tier_pfc_bio.py     architecture: three tiers, dual-buffer kNN correction
plugin/model/methods/three_tier_pfc_bio.py    training and prequential evaluation loop
plugin/model/lib/slow_priors.py               frozen prior interface (LimiX, TabPFN, tree ensembles)
configs/<dataset>/                            operating points; see docs/configurations.md
configs/default/, configs/opt_space/          defaults and search space for the method
scripts/                                      analyses behind the paper's tables
analysis/<dataset>/                           recorded search results the analyses read back
docs/                                         data, cache layout, and configuration reference
```

Baselines reported in the paper (TabR, TabM, FT-Transformer, TARS, Koodos, AdapTable,
Drift-Resilient TabPFN and others) are not included; run them from their own repositories.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./setup.sh talent          # clones TALENT and installs this work's files into ./talent
```

TALENT has no plugin registry, so `setup.sh` also patches one upstream file,
`model/utils.py`: it registers the method with `get_method()`, adds `three_tier_pfc_bio`
to the `--model_type` choices, and adds `--config_name` so a run can select one of the
per-dataset configurations below. All three edits are idempotent, and the script refuses
to write a file that would not import.

Tested with Python 3.11 and PyTorch 2.x.

## Data

Datasets and cached prior predictions are not redistributed. See
[`docs/data.md`](docs/data.md) for how to obtain the TabReD benchmark and how to generate
the prior cache the method consumes.

## Run

```bash
cd talent
python -u train_model_deep.py \
    --dataset weather \
    --model_type three_tier_pfc_bio \
    --config_name adaptive_best \
    --dataset_path_tabred tabred/data \
    --enable_timestamp --temporal_policy indices \
    --validate_option holdout_foremost_sample \
    --cat_policy ohe --gpu 0 --seed_num 1
```

Per-dataset settings live in `configs/<dataset>/`; `adaptive_best.json` is the
configuration behind the main results. `--config_name` is what selects one — without it
the run uses `configs/default/three_tier_pfc_bio.json` instead, which is not the reported
configuration. A named file that does not exist is an error rather than a silent
fallback. [`docs/configurations.md`](docs/configurations.md) lists every configuration and
what it sets.

## Analyses

| Script | What it does |
|---|---|
| `scripts/delong_auc_ci.py` | DeLong standard errors and 95% intervals for the classification AUCs |
| `scripts/paired_mechanism_ablation.py` | paired on/off measurement of the two adaptive mechanisms at a frozen configuration |
| `scripts/adapter_effect_study.py` | effect of the low-rank embedding adaptation against a matched-perturbation control |
| `scripts/prior_loader.py` | loads the prior ensemble with a consistent prediction scale |

The cache the analyses read is built by three further scripts, described in
[`docs/data.md`](docs/data.md):

| Script | What it builds |
|---|---|
| `scripts/build_limix_cache.py` | frozen LimiX predictions and embeddings, one directory per prior variant |
| `scripts/build_tree_cache.py` | XGBoost, CatBoost and LightGBM predictions, with out-of-fold training predictions |
| `scripts/build_leaf_embeddings.py` | the tree leaf-index retrieval space the classification instantiation searches |

Run them from the repository root. Each takes a dataset name, reads the recorded search
result it needs from [`analysis/`](analysis/README.md), and writes its output under
`results/`. All of them require the prior cache described in `docs/data.md`;
`delong_auc_ci.py` additionally needs the tree-model caches listed there.

`delong_auc_ci.py` is the quickest way to check that the caches are in place. It prints
one block per dataset:

```
=== homecredit-default ===
  rebuilt AUC 0.869686   (saved in results json: 0.869686; paper: 0.8697)
  n=56001  n_pos=1229  n_neg=54772
  DeLong SE = 0.00475   95% CI = [0.8604, 0.8790]
  gap vs XGBoost (0.867) = +0.0027  ->  0.57 x SE
```

The rebuilt AUC matches the value stored in `analysis/<dataset>/classification.json` to
six decimals when the caches in place are the ones the search was run on, which makes this
a quick way to confirm an install.

## Implementation notes

**The low-rank embedding adaptation uses an asymmetric initialisation.** `U` is drawn at
random and `V` is zero, so `U V^T = 0` at the start while the gradient is not; initialising
both factors to zero makes the origin an exact fixed point and the component can never
act. The update is the frozen-neighbour gradient of the batch loss, taken before the batch
enters the buffers. It is enabled on `weather` and `cooking-time` and off in the other
six configurations, for the three separate reasons given in `docs/data.md` §6 — the
widest being that the derivation is in squared-error currency and so belongs to the
regression instantiation.

**Some switches are exploratory.** The model exposes 20 feature switches, of which six
are on in at least one shipped configuration: `use_online_adaptation`, `use_dual_buffer`,
`use_whitening`, `use_ensemble_gating`, `use_cascaded_residual`, `use_drift_mixw`, plus
`use_adaptive_embedding` on the two datasets named above. The remaining thirteen —
`use_regime_bias`, `use_regime_clip`, `use_tabpfn_aux` and the like — are false
everywhere, and no reported result uses them. They are single-line guards kept because
they cost nothing to leave in; the larger unreachable branches have been removed.

**Cached prior predictions come in two conventions.** A cache holds predictions either in
raw target units or normalised by the training target statistics.
`scripts/prior_loader.py` detects which and converts before blending; the training path
works in the normalised space. See `docs/data.md` §4.

## Licence

The files written for this work — `plugin/`, `configs/`, `scripts/`, `analysis/` and
`docs/` — are MIT; see [`LICENSE`](LICENSE).

Nothing else that the method needs is redistributed here, and none of it is covered by
that licence:

- **TALENT** is cloned by `setup.sh` from its own repository and stays under its own
  terms. The plugin adds files to that checkout and overwrites none of it; the single
  upstream file `setup.sh` edits is `model/utils.py`, to register the method and its
  command-line options.
- **The datasets** and **the pretrained priors** keep the terms of their own sources.
  [`docs/data.md`](docs/data.md) says where to obtain each.

## Citation

The paper is under review. A BibTeX entry will be added here once it appears; until then,
please cite it by title:

> *Beyond Temporal Encoding: Efficient and Hierarchical Model-Level Online Adaptation for
> Tabular Distribution Shift*, 2026.
