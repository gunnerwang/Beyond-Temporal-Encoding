# Data and cached priors

Neither the datasets nor the pretrained prior models are redistributed with this
repository.

## 1. TabReD

The eight main datasets come from the TabReD benchmark
(https://github.com/yandex-research/tabred). Follow its instructions to download and
preprocess, then point `--dataset_path_tabred` at the resulting directory. The layout the
code expects is

```
tabred/data/<dataset>/
    X_num.npy  X_bin.npy  X_cat.npy  Y.npy  info.json
    split-default/
        train_idx.npy  val_idx.npy  test_idx.npy
```

The splits are temporal: `train` precedes `val`, which precedes `test`. The method is
evaluated prequentially over the test stream.

Datasets used: `weather`, `cooking-time`, `delivery-eta`, `sberbank-housing`,
`maps-routing`, `homesite-insurance`, `ecom-offers`, `homecredit-default`.

## 2. Frozen prior predictions

The slow tier is a frozen pretrained model. Its predictions and embeddings are computed
once per dataset and cached, so the online experiments do not re-run the prior. The cache
layout is

```
cache/limix/<dataset>/
    limix_predictions_{train,val,test}_ctx<N>.npy
    limix_embeddings_{train,val,test}_ctx<N>.npy
cache/limix/<dataset>-{ts,tn,seed1,seed2,seed3,ts-seed1,ts-seed2}/
    limix_predictions_{train,val,test}_ctx<N>.npy
cache/xgb/<dataset>/
    xgb_predictions_{train,val,test}.npy
```

`ctx<N>` is the in-context sample count: 10000 for `weather`, 3000 elsewhere. The suffixed
directories hold the prior variants that make up the ensemble: `ts` and `tn` use different
temporal encodings of the input, `seed1`–`seed3` differ only in the sampling seed used to
draw the in-context set, and `ts-seed1` / `ts-seed2` combine the two. Six variants plus
XGBoost is the usual pool; `cooking-time` adds `ts-seed1` for seven plus XGBoost.

A configuration's weights refer to the variants it names.
`scripts/prior_loader.py` lists the variants it loaded, so the pool behind a blend is
recorded in the log.

The classification instantiation blends tree models instead, and uses their leaf
embeddings as the retrieval space:

```
cache/{xgb,catboost,lightgbm}/<dataset>/
    <model>_predictions_test_raw.npy
cache/tree_leaf/<dataset>/
    leaf_embeddings_{train,val,test}.npy
```

The `_raw` suffix marks predictions written in target units with no post-hoc calibration;
it is a separate convention from the `xgb_predictions_<split>.npy` files above, which the
regression loader reads.

### Building it

Three scripts write the cache; each takes one dataset, skips what is already there, and
accepts `--dry-run` to show what it would do:

```bash
python -u scripts/build_limix_cache.py     --dataset weather
python -u scripts/build_tree_cache.py      --dataset homesite-insurance
python -u scripts/build_leaf_embeddings.py --dataset homesite-insurance
```

Run them from the TALENT checkout, not from this repository: they import
`model.lib.data` and `model.lib.slow_priors`, and resolve `tabred/data/` and `cache/`
against the working directory. `build_limix_cache.py` needs the prior model described in
§3; the other two need only a GPU, and take `--cpu` if there is none.

`configs/<dataset>/adaptive_best.json` lists under `ensemble_weights` which LimiX variants
that dataset's pool actually uses. Building more than those costs GPU time and changes
nothing.

The tree predictions were fitted with hyperparameters chosen by a search over validation
AUC. Two of those searches ship as records — `analysis/homesite-insurance/xgb-tuning.json`
and `analysis/homecredit-default/xgb-tuning.json` — and a rebuild picks them up
automatically. For a dataset without a record the builder uses library defaults and names
the source it used, so what a given cache was built from is always visible in the log; a
search of your own, saved in the same place, feeds straight back in.

## 3. The prior model

`LimiXSlowPrior` does not vendor LimiX. It expects a checkout of
<https://github.com/limix-ldm/LimiX> beside the framework, at the root of the TALENT
checkout:

```
talent/
    model/          from this repository's plugin, installed by setup.sh
    LimiX/          git clone https://github.com/limix-ldm/LimiX.git
    cache/          written by the scripts below
```

The path is computed from the location of `model/lib/slow_priors.py`, so the directory has
to be named `LimiX` and sit exactly there. Without it the first prior call raises
`ModuleNotFoundError: inference.predictor`.

Weights are fetched on first use from the Hugging Face repository
`stableai-org/LimiX-16M` (file `LimiX-16M.ckpt`) into `cache/`, which needs
`huggingface_hub` installed and network access; pass `model_path` to use a local
checkpoint instead. The inference configuration is read from the LimiX checkout itself,
`config/{reg,cls}_default_{retrieval,noretrieval}.json`, chosen by task and by whether
retrieval is enabled.

LimiX has its own dependencies; see `environment.yml` in its repository. Pin the commit
you build against — the reported caches were produced against `bba5448` (2025-12-18).

## 4. Prediction scale

A prior cache holds predictions either in raw target units or in the space standardised by
the training target statistics. Both conventions are supported and neither has to be
declared: `scripts/prior_loader.py` reads the convention off each array by comparing its
mean against the training target mean and standard deviation, converts to raw target units,
and blends there. The training path does the same in the other direction — the pipeline
standardises regression labels, so the method hands the label statistics to the prior and a
raw-unit cache is converted on load. Each run names the convention it found.

The test reads `|mean(p) - y_mean| / y_std < 1` as raw units, which separates the two
whenever the training target mean is at least one standard deviation from zero. That holds
on the regression targets, where the ratio runs from 2.4 to 30. On the binary targets the
ratio is 0.18 to 0.55 and every array reads as raw units, which is what those caches hold:
they are written as probabilities on the target scale. `load_blend` says so when it runs on
such a target.

Blending weights follow the priors in the same spirit. The weights are renormalised over
the priors that have a cache for the split being read, so the blend keeps its scale
whatever the pool contains, and the run reports which priors it used and what share of the
configured weight they carry.

## 5. Disk

A full cache for the eight datasets is on the order of ten gigabytes, dominated by the
training-split embeddings. Only the test-split predictions are needed to reproduce the
online results; the training split is required to fit the whitening transform and the
prior calibration statistics.

## 6. The low-rank embedding adaptation

The adaptation transforms the retrieval space by `e (I + U V^T)`, with `U` drawn from
`N(0, sigma_0^2)`, `sigma_0 = 1/sqrt(3d)`, and `V = 0`. The product is zero at the start,
so the initial retrieval geometry is unchanged, but the gradient is not — initialising
both factors to zero would make `(0, 0)` an exact fixed point, because the gradient with
respect to each factor carries the other as a factor.

After each batch of revealed labels, `U` and `V` are updated by gradient descent on the
batch loss with the neighbour set held fixed for that batch: the top-k indices are chosen
under the current projection and then treated as constants, and the loss is
differentiated through the distance-weighted aggregation and the embedding transform.
The step is taken before the batch is written to the buffers, so the neighbour sets are
the ones that produced the predictions being scored.

### Scope

The derivation covers the distance-weighted readout in squared-error currency. When the
configuration selects the regime-aware neighbour penalty, logit-space correction, or a
classification objective, the adaptation is left at its initial value rather than updated
by an expression derived for a different path; see `_adaptive_supported` in
`three_tier_pfc_bio.py`. Since `U V^T = 0` at initialisation, left at its initial value
means the retrieval geometry stays the identity for the whole run, which is worth knowing
when reading a result. The model prints one line whenever `use_adaptive_embedding` is set
on a configuration outside that scope.

### Where it is on

`use_adaptive_embedding` is true in two of the shipped configurations:

```
configs/weather/adaptive_best.json         adaptive_rank 16   adaptive_lr 1e-1
configs/cooking-time/adaptive_best.json    adaptive_rank  4   adaptive_lr 1e0
```

It is explicitly false in the other six, for three separate reasons:

* `ecom-offers`, `homesite-insurance` and `homecredit-default` are the classification
  instantiation; the derivation above is for the regression readout.
* `sberbank-housing` has a test stream shorter than the horizon the adaptation
  accumulates over.
* `maps-routing` selects its operating point from the prior variants its cache holds, and
  on `delivery-eta` the coherent-key cells of the sweep sit within the no-adapter baseline
  at the significance threshold it used.

### Choosing a learning rate

Below roughly `1e-2` the transform stays too small to change the retrieval order, and
with weight decay in the loop the product decays back towards zero; the two live
configurations sit at `1e-1` and `1e0` for that reason.
`scripts/adapter_effect_study.py` is the measurement behind those values. It sweeps rank
and learning rate against two controls — a matched-norm frozen-random transform, which
holds the size of the perturbation fixed at what the learned run reached, and coherent
versus stale buffer keys.

`analysis/<dataset>/embedding-adaptation.json` records a separate search over the pipeline
as a whole; the rates above come from the sweep described here.
