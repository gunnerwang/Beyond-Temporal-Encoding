#!/usr/bin/env python3
"""Load the prior ensemble with a consistent prediction scale.

The cached LimiX predictions are not stored in a uniform space: some splits and some
variants hold raw target values, others hold values normalised by the training target
statistics, and the mixture differs per dataset. The original pipeline calibrated every
prior except the base idx prior onto the statistics of the training-split idx predictions
and then applied `pdn = blend * y_std + y_mean`, which is correct only when the idx
predictions are normalised on both splits.

Here each array's space is detected against the training target statistics, every prior is
converted to raw target units, the same calibration is applied in that common space, and
the blend is used directly.

    space(p) = RAW  if |mean(p) - y_mean| / y_std < 1  else NORMALISED

The test separates the two conventions only when |y_mean| / y_std >= 1. That holds on the
regression targets (2.4 to 30) and not on the binary ones (0.18 to 0.55), where every
array is reported as RAW -- which is the correct answer, since those caches hold
probabilities in target units, but is not a distinction the rule drew. `load_blend` says
so when it runs on such a target.

Returns predictions in raw target units, so no further affine map is applied downstream.
"""
import numpy as np, os, json


def _load1(path):
    a = np.load(path).astype(np.float64)
    return a[:, 1] if a.ndim == 2 else a


#: cache directory suffix -> weight name, for every LimiX variant the ensemble can use.
VARIANTS = [("ts", "ts"), ("tn", "tn"), ("seed1", "s1"), ("seed2", "s2"), ("seed3", "s3"),
            ("ts-seed1", "ts1"), ("ts-seed2", "ts2")]


def detect_space(a, ym, ys):
    return "RAW" if abs(float(np.mean(a)) - ym) / ys < 1.0 else "NORM"


def detector_is_decisive(ym, ys):
    """Whether `detect_space` can actually tell the two conventions apart.

    A normalised array has mean about zero, so it is only distinguishable from a raw one
    when the training target mean is at least one standard deviation away from zero. That
    holds comfortably on the regression targets and not at all on the binary ones, where
    every array is reported as RAW. It is the right answer there -- the classification
    caches hold probabilities in target units -- but it is not a decision the rule made.
    """
    return abs(ym) / ys >= 1.0


def load_blend(ds, ctx, split, params, y_train, verbose=True):
    """Blend the prior ensemble for `split` in raw target units.

    params: the recorded configuration dict holding the w_<name> weights.
    Returns (blend, prior_names, spaces).

    A prior that carries a weight but has no cache on disk is named in the output, so the
    pool a call actually blended is visible; its weight goes to the base idx prior.
    """
    ym, ys = float(np.mean(y_train)), float(np.std(y_train)) + 1e-12
    b = f"cache/limix/{ds}"
    to_raw = lambda a: a if detect_space(a, ym, ys) == "RAW" else a * ys + ym

    p_tr_raw = to_raw(_load1(f"{b}/limix_predictions_train_ctx{ctx}.npy"))
    btm, bts = p_tr_raw.mean(), p_tr_raw.std() + 1e-8
    calib = lambda p: (p - p.mean()) / (p.std() + 1e-8) * bts + btm

    spaces = {}
    raw_idx = _load1(f"{b}/limix_predictions_{split}_ctx{ctx}.npy")
    spaces["idx"] = detect_space(raw_idx, ym, ys)
    priors = {"idx": to_raw(raw_idx)}
    for suf, nm in VARIANTS:
        p = f"cache/limix/{ds}-{suf}/limix_predictions_{split}_ctx{ctx}.npy"
        if os.path.exists(p):
            a = _load1(p); spaces[nm] = detect_space(a, ym, ys); priors[nm] = calib(to_raw(a))
    xp = f"cache/xgb/{ds}/xgb_predictions_{split}.npy"
    if os.path.exists(xp):
        a = _load1(xp); spaces["xgb"] = detect_space(a, ym, ys); priors["xgb"] = calib(to_raw(a))

    names = list(priors.keys())
    weighted = [n for n in [nm for _, nm in VARIANTS] + ["xgb"] if f"w_{n}" in params]
    missing = [n for n in weighted if n not in priors]
    if missing:
        lost = sum(float(params[f"w_{n}"]) for n in missing)
        print(f"  note: {missing} carry weight in the configuration and have no cache for "
              f"split '{split}'. Their combined weight {lost:.4f} goes to the base idx "
              f"prior, so the blend covers a subset of the recorded pool.")

    w = {n: float(params[f"w_{n}"]) for n in names if n != "idx" and f"w_{n}" in params}
    w["idx"] = 1.0 - sum(w.values())
    if w["idx"] < 0.0:
        raise ValueError(f"weights of the non-idx priors sum to {sum(w.values()) - w['idx']:.4f} "
                         f"> 1, which leaves the idx prior a negative weight; check `params`")
    blend = sum(w[n] * priors[n] for n in w)
    if verbose:
        print(f"  priors {names}")
        print(f"  cached spaces {spaces}  -> all converted to raw target units")
        if not detector_is_decisive(ym, ys):
            print(f"  note: |mean|/sd of the target is {abs(ym)/ys:.2f}, below the 1.0 the "
                  f"raw/normalised test needs to be decisive, so it reports RAW throughout")
        print(f"  blend mean {blend.mean():.4f}  (y_{split} mean {ym:.4f} from train)")
    return blend.astype(np.float32), names, spaces
