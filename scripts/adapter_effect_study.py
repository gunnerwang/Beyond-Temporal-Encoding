#!/usr/bin/env python3
"""Effect of online low-rank embedding adaptation, measured against two controls.

U and V are updated by gradient descent on the batch prediction loss with the neighbour
set held fixed for the batch: indices are selected under the current projection and then
treated as constants, and the loss is differentiated through the distance-weighted
aggregation and the embedding transform.

    q_i     = e~_i (I + U V^T)
    d_ij    = ||q_i - k_j||          (j in the frozen top-k)
    a_ij    = softmax_j(-d_ij / T)
    corr_i  = sum_j a_ij v_j / sum_j a_ij      (MAD-clipped values)
    y^_i    = pdn_i - s * corr_i
    L       = mean_i (y^_i - y_i)^2      ->  dL/dU, dL/dV by autograd

Every hyperparameter other than the adapter is frozen at the recorded configuration, so
adapter-on and adapter-off runs differ in nothing else. Two controls accompany each cell:
a matched-norm frozen-random transform, and both key-handling conditions.

Usage: python -u scripts/adapter_effect_study.py <dataset>   (run from the repository root)
"""
import numpy as np, torch, json, os, sys, itertools, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prior_loader import load_blend

torch.set_grad_enabled(False)
dataset = sys.argv[1] if len(sys.argv) > 1 else "weather"
dev = torch.device("cpu")

CTX = {"weather": 10000, "delivery-eta": 3000, "sberbank-housing": 3000, "cooking-time": 3000}
ctx = CTX.get(dataset, 3000)

os.makedirs("results", exist_ok=True)
OUT = f"results/adapter_effect_{dataset}.json"

RANKS = [4, 16]
LRS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
KEY_MODES = ["coherent", "stale"]
SEEDS = [0, 1, 2, 3, 4]

Y = np.load(f"tabred/data/{dataset}/Y.npy").astype(np.float32)
tr = np.load(f"tabred/data/{dataset}/split-default/train_idx.npy")
te = np.load(f"tabred/data/{dataset}/split-default/test_idx.npy")
Y_train, Y_test = Y[tr], Y[te]
ym, ys = Y_train.mean(), Y_train.std() + 1e-8

base = f"cache/limix/{dataset}"
emb_train = np.load(f"{base}/limix_embeddings_train_ctx{ctx}.npy").astype(np.float32)
emb_test = np.load(f"{base}/limix_embeddings_test_ctx{ctx}.npy").astype(np.float32)

P = json.load(open(f"analysis/{dataset}/embedding-adaptation.json"))["params"] \
    if os.path.exists(f"analysis/{dataset}/embedding-adaptation.json") \
    else json.load(open(f"analysis/{dataset}/full-pipeline.json"))["params"]
_blend, pn, _spaces = load_blend(dataset, ctx, "test", P, Y_train)
pdn = torch.from_numpy(_blend).to(dev)
Yt = torch.from_numpy(Y_test).to(dev)

def fit_whitening(emb, reg=1e-5):
    mean = emb.mean(0); c = emb - mean
    cov = (c.T @ c) / max(len(c) - 1, 1) + reg * np.eye(c.shape[1])
    ev, evec = np.linalg.eigh(cov); o = np.argsort(ev)[::-1]
    return mean, evec[:, o] * (1.0 / np.sqrt(ev[o] + 1e-10))

w_mean, W = fit_whitening(emb_train)
Ew = torch.from_numpy(((emb_test - w_mean) @ W).astype(np.float32)).to(dev)
N, D = len(Y_test), Ew.shape[1]
chunk = 64
long_buf = 10000 if N > 20000 else 40960
SIG0 = 1.0 / np.sqrt(3.0 * D)

OS_, OK = float(P["online_scale"]), int(P["online_k"])
MIX, ST, LT = float(P["dual_buffer_mix_w"]), float(P["short_temp"]), float(P["long_temp"])
CLIP, BUF, WD = float(P["knn_clip"]), int(P["buf_size"]), float(P["weight_decay"])

print(f"dataset={dataset} N={N} D={D} priors={pn}", flush=True)
print(f"frozen: scale={OS_:.4f} k={OK} mix={MIX:.4f} sT={ST:.3f} lT={LT:.3f} clip={CLIP:.3f} "
      f"buf={BUF} wd={WD:.3e}   sigma0={SIG0:.5f}\n", flush=True)


def proj(X, U, V):
    return X + (X @ U) @ V.t()


def knn_corr(q, keys, vals, k, temp, want_grad):
    """Frozen-neighbour distance-weighted correction. Indices chosen without grad,
    distances to the chosen neighbours recomputed with grad."""
    n = keys.shape[0]; ka = min(k, n)
    if ka < 1:
        return torch.zeros(q.shape[0], device=dev)
    with torch.no_grad():
        d0 = torch.cdist(q, keys)
        idx = torch.topk(d0, ka, dim=1, largest=False).indices          # frozen
    kk = keys[idx]                                                      # (B, ka, D)
    with torch.set_grad_enabled(want_grad):
        td = torch.linalg.vector_norm(q.unsqueeze(1) - kk, dim=2)       # (B, ka)
        tv = vals[idx]
        lg = -td / max(temp, 1e-6)
        lg = lg - lg.max(1, keepdim=True).values.detach()
        w = torch.exp(lg); w = w / (w.sum(1, keepdim=True) + 1e-10)
        if ka >= 3:
            wn = w / (w.sum(1, keepdim=True) + 1e-8)
            c0 = (wn * tv).sum(1, keepdim=True)
            mad = torch.clamp((wn * (tv - c0).abs()).sum(1, keepdim=True), min=1e-6)
            tv = torch.min(torch.max(tv, c0 - CLIP * mad), c0 + CLIP * mad)
        return (w * tv).sum(1) / torch.clamp(w.sum(1), min=1e-8)


def run(mode, key_mode="coherent", rank=None, lr=None, seed=0, target_norm=None):
    U = V = None; learn = False
    if mode == "lora":
        g = torch.Generator().manual_seed(seed)
        U = (torch.randn(D, rank, generator=g) * SIG0).to(dev)
        V = torch.zeros(D, rank, device=dev); learn = True
    elif mode == "random":
        g = torch.Generator().manual_seed(10_000 + seed)
        U = torch.randn(D, rank, generator=g).to(dev)
        V = torch.randn(D, rank, generator=g).to(dev)
        nrm = torch.linalg.matrix_norm(U @ V.t())
        if float(nrm) > 0 and target_norm is not None:
            s = (target_norm / float(nrm)) ** 0.5
            U = U * s; V = V * s
    U0 = None if U is None else U.clone()

    lk = torch.zeros(long_buf, D, device=dev); lv = torch.zeros(long_buf, device=dev); lp = ls = 0
    sk = torch.zeros(BUF, D, device=dev); sv = torch.zeros(BUF, device=dev); sp = ss = 0
    final = pdn.clone(); diverged = None

    for ci, start in enumerate(range(0, N, chunk)):
        end = min(start + chunk, N); B = end - start
        raw = Ew[start:end]
        if learn:
            U.requires_grad_(True); V.requires_grad_(True)
            if U.grad is not None: U.grad = None
            if V.grad is not None: V.grad = None
        with torch.set_grad_enabled(learn):
            q = raw if U is None else proj(raw, U, V)
            corr = None
            if ls >= 100:
                if U is None or key_mode == "stale":
                    lkeys, skeys = lk[:ls], sk[:ss]
                else:
                    lkeys = proj(lk[:ls], U, V)
                    skeys = proj(sk[:ss], U, V) if ss else sk[:ss]
                lc = knn_corr(q, lkeys, lv[:ls], OK, LT, learn)
                if ss >= min(OK, 5):
                    sc = knn_corr(q, skeys, sv[:ss], min(OK, ss), ST, learn)
                    corr = MIX * sc + (1 - MIX) * lc
                else:
                    corr = lc
            pred = pdn[start:end] if corr is None else pdn[start:end] - OS_ * corr

        # gradient FIRST: the buffer writes below modify lk/sk in place, which would
        # invalidate the graph that proj(lk[:ls], U, V) depends on in coherent mode.
        # Mathematically identical -- the writes do not enter this chunk's loss.
        Un = Vn = None
        if learn and corr is not None and B > 1:
            with torch.enable_grad():
                loss = ((pred - Yt[start:end]) ** 2).mean()
            gU, gV = torch.autograd.grad(loss, [U, V], allow_unused=True)
            gU = torch.zeros_like(U) if gU is None else gU
            gV = torch.zeros_like(V) if gV is None else gV
            with torch.no_grad():
                Un = (U - lr * (gU + WD * U)).detach()
                Vn = (V - lr * (gV + WD * V)).detach()

        final[start:end] = pred.detach()
        store = (q.detach() if (U is not None and key_mode == "stale") else raw)
        for j in range(B):
            i = start + j; v = pdn[i] - Yt[i]
            lk[lp] = store[j]; lv[lp] = v; lp = (lp + 1) % long_buf; ls = min(ls + 1, long_buf)
            sk[sp] = store[j]; sv[sp] = v; sp = (sp + 1) % BUF; ss = min(ss + 1, BUF)

        if Un is not None:
            U, V = Un, Vn
            if diverged is None and not (torch.isfinite(U).all() and torch.isfinite(V).all()):
                diverged = ci; break

    if diverged is not None:
        return dict(rmse=float("nan"), diverged_at=diverged)
    rmse = float(torch.sqrt(((final - Yt) ** 2).mean()))
    if U is None:
        return dict(rmse=rmse, diverged_at=None, uvt=0.0, du=0.0)
    return dict(rmse=rmse, diverged_at=None,
                uvt=float(torch.linalg.matrix_norm(U @ V.t())),
                du=float(torch.linalg.matrix_norm(U - U0)))


t0 = time.time()
b = run("none"); BASE = b["rmse"]
print(f"[baseline] adapter deleted (proj = I)   RMSE {BASE:.8f}   ({time.time()-t0:.0f}s)\n", flush=True)

hdr = (f"{'keys':>9} {'rank':>5} {'lr':>7} | {'||UV^T||':>10} {'||U-U0||':>9} | "
       f"{'TRUE-GRAD rmse':>15} {'sd':>9} {'delta':>12} | {'RANDOM rmse':>13} | {'learned-random':>14}")
print(hdr, flush=True); print("-" * len(hdr), flush=True)

rows = []
for km, rank, lr in itertools.product(KEY_MODES, RANKS, LRS):
    res = [run("lora", key_mode=km, rank=rank, lr=lr, seed=s) for s in SEEDS]
    div = [r["diverged_at"] for r in res if r["diverged_at"] is not None]
    ok = [r for r in res if r["diverged_at"] is None]
    if not ok:
        print(f"{km:>9} {rank:>5} {lr:>7.0e} |  diverged (first at chunk {min(div)}, "
              f"{len(div)}/{len(SEEDS)} seeds)", flush=True)
        rows.append(dict(keys=km, rank=rank, lr=lr, diverged=len(div))); continue
    rm = np.array([r["rmse"] for r in ok]); tgt = float(np.mean([r["uvt"] for r in ok]))
    rnd = np.array([run("random", key_mode=km, rank=rank, seed=s, target_norm=tgt)["rmse"]
                    for s in SEEDS])
    print(f"{km:>9} {rank:>5} {lr:>7.0e} | {tgt:>10.3e} "
          f"{np.mean([r['du'] for r in ok]):>9.3e} | {rm.mean():>15.8f} "
          f"{rm.std(ddof=1):>9.2e} {rm.mean()-BASE:>+12.7f} | {rnd.mean():>13.8f} | "
          f"{rm.mean()-rnd.mean():>+14.7f}", flush=True)
    rows.append(dict(keys=km, rank=rank, lr=lr, uvt=tgt,
                     du=float(np.mean([r["du"] for r in ok])),
                     learned_rmse=float(rm.mean()), learned_sd=float(rm.std(ddof=1)),
                     learned_delta=float(rm.mean() - BASE), random_rmse=float(rnd.mean()),
                     random_sd=float(rnd.std(ddof=1)),
                     learned_minus_random=float(rm.mean() - rnd.mean()), diverged=len(div)))

live = [r for r in rows if "learned_rmse" in r]
print("-" * len(hdr))
print(f"\nbaseline                            : {BASE:.8f}")
print(f"cells beating baseline              : {sum(1 for r in live if r['learned_delta']<0)} / {len(live)}")
print(f"cells beating matched-norm random   : {sum(1 for r in live if r['learned_minus_random']<0)} / {len(live)}")
if live:
    bb = min(live, key=lambda r: r["learned_minus_random"])
    print(f"best learned-over-random            : {-bb['learned_minus_random']:+.8f} "
          f"(keys={bb['keys']} rank={bb['rank']} lr={bb['lr']:.0e} sd={bb['learned_sd']:.2e})")
print(f"\nelapsed {time.time()-t0:.0f}s")
json.dump({"dataset": dataset, "baseline_no_adapter": BASE, "sigma0": float(SIG0),
           "seeds": SEEDS, "cells": rows},
          open(OUT, "w"), indent=2)
print("saved", OUT, flush=True)
