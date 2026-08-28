# Configurations

Every file under `configs/<dataset>/` is a complete operating point: a prior pool with its
blending weights, the online correction hyperparameters, and the switches that decide
which mechanisms are active. Select one with `--config_name`, without the `.json`:

```bash
python -u train_model_deep.py --dataset weather --model_type three_tier_pfc_bio \
    --config_name adaptive_best ...
```

The named file is deep-merged into `configs/default/three_tier_pfc_bio.json`, and wins on
conflicts. `configs/opt_space/three_tier_pfc_bio.json` holds the search space, and is read
only when a run is launched with `--tune`.

## What each family is

| file | what it configures |
|---|
| `adaptive_best.json` | the full pipeline — prior ensemble, dual-buffer kNN correction, drift-aware blending, cascaded L2, and the low-rank embedding adaptation where it applies. This is the configuration behind the main results. |
| `7way_optimized.json` | the prior ensemble alone, with Optuna weights and no online correction. The ensemble-only step of the tier progression. |
| `adaptive_fs.json` | the same pipeline over a forward-selected subset of the priors rather than the full pool, for the sparsity analysis. |
| `cascaded.json` | cascaded L2 correction with a single L2 buffer. |
| `cascaded_dual.json` | cascaded L2 correction with a dual L2 buffer. |
| `cascaded_best.json` | a second dual-buffer L2 operating point, kept where it differs from `cascaded_dual.json`. |
| `cls_tree_ensemble.json` | the classification instantiation: a tree ensemble with leaf embeddings as the retrieval space. |
| `neural_blend.json` | a pool that mixes tree and neural priors. |

## Which datasets have which

| dataset | `adaptive_best` | `7way_optimized` | `adaptive_fs` | `cascaded` | `cascaded_best` | `cascaded_dual` | `cls_tree_ensemble` | `neural_blend` |
|---|---|---|---|---|---|---|---|---|
| `cooking-time` | y | y | y | y |  | y |  |  |
| `delivery-eta` | y | y | y | y |  | y |  |  |
| `ecom-offers` | y | y |  | y | y | y | y | y |
| `homecredit-default` | y |  |  |  |  | y | y |  |
| `homesite-insurance` | y | y |  | y | y | y | y | y |
| `maps-routing` | y |  | y |  |  | y |  | y |
| `sberbank-housing` | y | y | y | y |  | y |  |  |
| `weather` | y | y | y | y |  | y |  |  |

`adaptive_best.json` exists for every dataset. The tree-ensemble and neural-blend
families are specific to the classification instantiation, and to `maps-routing`, whose
pool also mixes tree and neural priors.

## Prior pools

The pool is not the same size everywhere. `ensemble_weights` in each file lists it
explicitly:

| dataset | priors in `adaptive_best.json` |
|---|
| `cooking-time` | 8 — idx, seed1, seed2, seed3, tn, ts, ts-seed1, xgb |
| `delivery-eta` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `ecom-offers` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `homecredit-default` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `homesite-insurance` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `maps-routing` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `sberbank-housing` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |
| `weather` | 7 — idx, seed1, seed2, seed3, tn, ts, xgb |

See `docs/data.md` §2 for what each LimiX variant is and which cache directory it comes
from, and §6 for where the low-rank embedding adaptation is active and why.

## Gates that act at run time

Two mechanisms shape how strongly the correction is applied once a configuration is
loaded, and both announce themselves in the log. Each has a switch that a configuration
can set: unlisted keys reach the model, so an entry in `configs/<dataset>/…json` is enough
to pin either to the behaviour you want.

**Embedding quality.** `use_embedding_quality_scale` measures a leave-one-out kNN error
correlation on the training embeddings and multiplies `online_scale` by a factor between
`embedding_quality_min_scale` (0.1) and 1.0. A correlation of 0.3 or above keeps the full
correction, 0.05 or below gives the floor, and the range in between interpolates, so the
correction is applied in proportion to how well the retrieval space predicts error on the
data at hand. The factor is printed with the correlation it came from:

```
[EmbQuality] kNN-error correlation: <corr> → online_scale multiplier: <mult>
```

Read it alongside `online_scale`: the two multiply into the strength the correction runs
at, which is what a result reflects.

**Cascaded L2.** The second-level correction is admitted by a running estimate of what it
is contributing:

```
[L2 Gate] samples=4032, EMA=0.00041221 (denorm, std=0.3864), hurt_frac=0.11, batch_benefit=0.00315959
```

`EMA` is the smoothed benefit in target units and `hurt_frac` the share of recent samples
where the second level did not improve on the first. The gate is what lets
`use_cascaded_residual` stay on across streams of different character.

Two further lines can appear. `[AdaptiveEmbedding]` reports that the low-rank adaptation
was switched on for a configuration outside the derivation's scope and stays at its initial
value; see `docs/data.md` §6. `[AutoPrior]` belongs to `slow_prior_type='auto'`; the
shipped configurations name their prior directly.
