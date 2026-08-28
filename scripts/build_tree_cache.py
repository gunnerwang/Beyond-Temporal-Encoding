#!/usr/bin/env python3
"""Build the tree-model prediction cache for one dataset.

The classification instantiation blends gradient-boosted trees instead of the foundation
model, and the regression pools use XGBoost as the one structurally different prior. Both
read their predictions from `cache/<model>/<dataset>/`, which this script writes:

    <model>_predictions_test.npy        normalised by the training target statistics
    <model>_predictions_test_raw.npy    target units
    <model>_predictions_train.npy       5-fold out-of-fold, normalised

The out-of-fold training predictions matter: the blend calibrates every prior against the
training-split statistics, and in-sample predictions would make that calibration optimistic.

All three models train on the same float-encoded feature matrix, with categorical columns
NaN-filled to -1, and use early stopping on the validation split to pick the number of
trees. The task comes from `info.json`: regression datasets are fitted with a squared-error
objective, and the three binary ones with a logistic objective whose predicted probability
is what gets cached. Classification runs average several seeds, which is how the reported
caches were built; XGBoost additionally picks up the tuned hyperparameters recorded in
`analysis/<dataset>/xgb-tuning.json` where one exists.

Usage
    python -u scripts/build_tree_cache.py --dataset homesite-insurance
    python -u scripts/build_tree_cache.py --dataset weather --models xgb
    python -u scripts/build_tree_cache.py --dataset ecom-offers --dry-run

Run it from the TALENT checkout, so that `tabred/data/` and `cache/` resolve. Needs a GPU;
pass --cpu to train on the processor instead, which is much slower but has no other cost.
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

MODELS = ("xgb", "catboost", "lightgbm")
TUNED_PARAMS = ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                "min_child_weight", "reg_alpha", "reg_lambda", "gamma", "scale_pos_weight")


def load_dataset(dataset, root="tabred/data"):
    """Concatenate the numeric, binary and categorical blocks, then apply the split."""
    d = os.path.join(root, dataset)
    parts = []
    for fname in ("X_num.npy", "X_bin.npy", "X_cat.npy"):
        p = os.path.join(d, fname)
        if not os.path.exists(p):
            continue
        x = np.load(p).astype(np.float32)
        if "cat" in fname:
            x = np.nan_to_num(x, nan=-1)
        parts.append(x)
    if not parts:
        raise SystemExit(f"no feature files under {d}")
    X = np.concatenate(parts, axis=1)
    Y = np.load(os.path.join(d, "Y.npy")).astype(np.float32)
    idx = {s: np.load(os.path.join(d, "split-default", f"{s}_idx.npy"))
           for s in ("train", "val", "test")}
    task = json.load(open(os.path.join(d, "info.json"))).get("task_type", "regression")
    return {s: (X[i], Y[i]) for s, i in idx.items()}, task


def tuned_xgb_params(dataset, analysis_root):
    """Hyperparameters recorded by the classification search, when there is a record."""
    p = os.path.join(analysis_root, dataset, "xgb-tuning.json")
    if not os.path.exists(p):
        print(f"  no tuning record at {p}; using library defaults for XGBoost")
        return {}
    kept = {k: v for k, v in json.load(open(p)).get("best_params", {}).items()
            if k in TUNED_PARAMS}
    print(f"  XGBoost hyperparameters from {p}")
    return kept


def make_model(name, task, device, n_estimators=2000, early_stopping=True,
               seed=42, extra=None, lr=0.05):
    cls = task != "regression"
    extra = dict(extra or {})
    extra.setdefault("learning_rate", lr)
    if name == "xgb":
        import xgboost as xgb
        params = dict(tree_method="hist", device=device, n_estimators=n_estimators,
                      verbosity=0, random_state=seed)
        if cls:
            params.update(objective="binary:logistic", eval_metric="auc")
            params.update({"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8,
                           "colsample_bytree": 0.8, "min_child_weight": 10,
                           "reg_alpha": 0.1, "reg_lambda": 1.0})
            params.update(extra)
            if early_stopping:
                params["early_stopping_rounds"] = 50
            return xgb.XGBClassifier(**params)
        params.update(objective="reg:squarederror", max_depth=8, learning_rate=lr,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                      reg_alpha=0.1, reg_lambda=1.0)
        if early_stopping:
            params["early_stopping_rounds"] = 50
        return xgb.XGBRegressor(**params)

    if name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        params = dict(iterations=n_estimators, learning_rate=lr, depth=8,
                      random_seed=seed, task_type="GPU" if device == "cuda" else "CPU",
                      early_stopping_rounds=50 if early_stopping else None, verbose=False)
        if cls:
            return CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", **params)
        return CatBoostRegressor(l2_leaf_reg=3.0, bootstrap_type="Bernoulli",
                                 subsample=0.8, **params)

    if name == "lightgbm":
        import lightgbm as lgb
        params = dict(n_estimators=n_estimators, learning_rate=lr, max_depth=8,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                      reg_alpha=0.1, reg_lambda=1.0,
                      device="gpu" if device == "cuda" else "cpu",
                      verbose=-1, random_state=seed)
        return (lgb.LGBMClassifier(**params) if cls else lgb.LGBMRegressor(**params))

    raise SystemExit(f"unknown model {name}")


def fit_with_early_stopping(name, model, Xtr, ytr, Xva, yva):
    if name == "xgb":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return int(model.best_iteration)
    if name == "catboost":
        from catboost import Pool
        model.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), verbose=False)
        return int(model.get_best_iteration())
    import lightgbm as lgb
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return int(model.best_iteration_)


def predict(model, X, task):
    if task == "regression":
        return model.predict(X).astype(np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def score(y, pred, task):
    if task == "regression":
        return "RMSE", float(np.sqrt(np.mean((pred - y) ** 2)))
    return "AUC", float(roc_auc_score(y, pred))


def build(dataset, name, data, task, out_root, device, overwrite, seeds, extra, lr):
    out_dir = os.path.join(out_root, name, dataset)
    done = os.path.join(out_dir, f"{name}_predictions_test_raw.npy")
    if os.path.exists(done) and not overwrite:
        print(f"  {name:9s} already cached, skipping (use --overwrite to rebuild)")
        return

    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    ym, ys = float(ytr.mean()), float(ytr.std()) + 1e-8

    print(f"\n--- {name} on {dataset} ({task}) ---", flush=True)
    test_preds, train_preds = [], []
    for seed in range(seeds):
        model = make_model(name, task, device, seed=seed,
                           extra=extra if name == "xgb" else None, lr=lr)
        best_iter = fit_with_early_stopping(name, model, Xtr, ytr, Xva, yva)
        pred_test = predict(model, Xte, task)
        metric, value = score(yte, pred_test, task)
        print(f"  seed {seed}: best iteration {best_iter}   test {metric} {value:.5f}", flush=True)
        if best_iter < 10:
            print(f"    early stopping after {best_iter} rounds: validation {metric} peaks "
                  f"at the start on this split. The reported cache used hyperparameters "
                  f"from a search that optimised validation {metric} directly; supply them "
                  f"through analysis/{dataset}/xgb-tuning.json to match it.", flush=True)
        test_preds.append(pred_test)

        pred_train = np.zeros(len(Xtr), dtype=np.float32)
        for tr, va in KFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr):
            m = make_model(name, task, device, n_estimators=max(best_iter, 1),
                           early_stopping=False, seed=seed,
                           extra=extra if name == "xgb" else None, lr=lr)
            if name == "catboost":
                from catboost import Pool
                m.fit(Pool(Xtr[tr], ytr[tr]), verbose=False)
            else:
                m.fit(Xtr[tr], ytr[tr])
            pred_train[va] = predict(m, Xtr[va], task)
        train_preds.append(pred_train)

    pred_test = np.mean(test_preds, axis=0).astype(np.float32)
    pred_train = np.mean(train_preds, axis=0).astype(np.float32)
    if seeds > 1:
        metric, value = score(yte, pred_test, task)
        print(f"  {seeds}-seed average: test {metric} {value:.5f}", flush=True)
    metric, value = score(ytr, pred_train, task)
    print(f"  out-of-fold training {metric} {value:.5f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{name}_predictions_test.npy"), (pred_test - ym) / ys)
    np.save(os.path.join(out_dir, f"{name}_predictions_test_raw.npy"), pred_test)
    np.save(os.path.join(out_dir, f"{name}_predictions_train.npy"), (pred_train - ym) / ys)
    print(f"  written to {out_dir}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated; one of " + ", ".join(MODELS))
    ap.add_argument("--data-root", default="tabred/data")
    ap.add_argument("--analysis-root", default="analysis")
    ap.add_argument("--out", default="cache", help="cache root (default: cache)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="seeds to average; default 5 for classification, 1 for regression")
    ap.add_argument("--learning-rate", type=float, default=0.05,
                    help="shrinkage for all three models (default 0.05); lower it when early "
                         "stopping fires after a handful of rounds")
    ap.add_argument("--cpu", action="store_true", help="train on the processor")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="load the data and report shapes without training")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {', '.join(MODELS)}")

    data, task = load_dataset(args.dataset, args.data_root)
    seeds = args.seeds if args.seeds is not None else (5 if task != "regression" else 1)
    n_feat = data["train"][0].shape[1]
    print(f"dataset {args.dataset}   task {task}   features {n_feat}   "
          + "  ".join(f"{s} {len(data[s][1])}" for s in ("train", "val", "test")))
    if task != "regression":
        print(f"  positives in train: {data['train'][1].mean():.4f}   seeds averaged: {seeds}")

    extra = tuned_xgb_params(args.dataset, args.analysis_root) if (
        task != "regression" and "xgb" in models) else {}

    if args.dry_run:
        for m in models:
            out_dir = os.path.join(args.out, m, args.dataset)
            state = "cached" if os.path.exists(
                os.path.join(out_dir, f"{m}_predictions_test_raw.npy")) else "to build"
            print(f"  {m:9s} {state:9s} -> {out_dir}")
        return

    device = "cpu" if args.cpu else "cuda"
    for m in models:
        build(args.dataset, m, data, task, args.out, device, args.overwrite, seeds, extra,
              args.learning_rate)
    print("\ndone.")


if __name__ == "__main__":
    main()
