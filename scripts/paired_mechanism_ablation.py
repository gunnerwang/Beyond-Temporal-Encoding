#!/usr/bin/env python3
"""Paired 2x2 ablation of the two adaptive mechanisms.

Every hyperparameter is frozen at the configuration recorded in
analysis/<dataset>/full-pipeline.json; the only thing that varies between arms is
which mechanism is switched on. There is no hyperparameter search, so the arm-to-arm
differences carry no search variance.

  base      fixed omega, no embedding adaptation                       (deterministic)
  adapbuf   drift-aware omega_t, no embedding adaptation               (deterministic)
  adapemb   fixed omega, embedding adaptation on                       (5 seeds)
  combined  drift-aware omega_t + embedding adaptation                 (5 seeds)

Fixed omega is set to base_mix_w * 0.5, which is exactly what the drift-aware rule
produces at neutral drift (sigmoid(0)) and during warm-up. Using base_mix_w itself would
confound "does omega move" with "is omega larger on average".

The embedding adaptation uses U ~ N(0, sigma0^2), sigma0 = 1/sqrt(3d), V = 0, updated by
gradient descent on the batch loss with the neighbour set frozen for the batch. It is
evaluated at two learning rates: the one recorded in the configuration, and a common eta
in the range where the transform is large enough to change the retrieval order.

Usage: python -u scripts/paired_mechanism_ablation.py <dataset>   (run from the repository root)
"""
import numpy as np, torch, json, os, sys, time
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prior_loader import load_blend

torch.set_grad_enabled(False)
ds = sys.argv[1] if len(sys.argv) > 1 else "weather"
dev = torch.device("cpu")
SEEDS = [0, 1, 2, 3, 4]
ETA_HI = 1e-1

CLS = {"homesite-insurance", "ecom-offers", "homecredit-default"}
is_cls = ds in CLS
ctx = 10000 if ds == "weather" else 3000

os.makedirs("results", exist_ok=True)
OUT = f"results/paired_mechanism_{ds}.json"

Y = np.load(f"tabred/data/{ds}/Y.npy").astype(np.float32)
tr = np.load(f"tabred/data/{ds}/split-default/train_idx.npy")
te = np.load(f"tabred/data/{ds}/split-default/test_idx.npy")
Y_train, Y_test = Y[tr], Y[te]
ym, ys = Y_train.mean(), Y_train.std() + 1e-8

b = f"cache/limix/{ds}"
emb_train = np.load(f"{b}/limix_embeddings_train_ctx{ctx}.npy").astype(np.float32)
emb_test = np.load(f"{b}/limix_embeddings_test_ctx{ctx}.npy").astype(np.float32)

P = json.load(open(f"analysis/{ds}/full-pipeline.json"))["params"]
blend, pn, spaces = load_blend(ds, ctx, "test", P, Y_train)
pdn = torch.from_numpy(blend)

def fit_whitening(e, reg=1e-5):
    m = e.mean(0); c = e - m
    cov = (c.T @ c) / max(len(c) - 1, 1) + reg * np.eye(c.shape[1])
    ev, evec = np.linalg.eigh(cov); o = np.argsort(ev)[::-1]
    return m, evec[:, o] * (1.0 / np.sqrt(ev[o] + 1e-10))

wm, W = fit_whitening(emb_train)
Ew = torch.from_numpy(((emb_test - wm) @ W).astype(np.float32))
N, D = len(Y_test), Ew.shape[1]
chunk = 64
long_buf = 10000 if N > 20000 else 40960
SIG0 = 1.0 / np.sqrt(3.0 * D)
Yt = torch.from_numpy(Y_test.astype(np.float32))

OS_, OK = float(P["online_scale"]), int(P["online_k"])
ST, LT, CLIP, BUF = float(P["short_temp"]), float(P["long_temp"]), float(P["knn_clip"]), int(P["buf_size"])
W0, ALPHA, DW = float(P["base_mix_w"]), float(P["alpha"]), int(P["drift_window"])
RANK, ETA, WD = int(P["rank"]), float(P["lr"]), float(P["weight_decay"])
OMEGA_FIXED = W0 * 0.5

def metric(pred):
    p = pred.numpy()
    return roc_auc_score(Y_test, p) if is_cls else float(np.sqrt(np.mean((p - Y_test.astype(np.float64)) ** 2)))

print(f"dataset={ds}  task={'classification (AUC)' if is_cls else 'regression (RMSE)'}  N={N} D={D}")
print(f"priors={pn}")
print(f"frozen: scale={OS_:.4f} k={OK} sT={ST:.3f} lT={LT:.3f} clip={CLIP:.3f} buf={BUF}")
print(f"drift : base_mix_w={W0:.4f} -> fixed omega={OMEGA_FIXED:.4f}   alpha={ALPHA:.3f} W={DW}")
print(f"adapt : rank={RANK} eta_recorded={ETA:.3e} eta_high={ETA_HI:.0e} wd={WD:.3e} sigma0={SIG0:.5f}\n",
      flush=True)

def proj(X, U, V): return X + (X @ U) @ V.t()

def knn(q, keys, vals, k, temp, grad):
    n = keys.shape[0]; ka = min(k, n)
    if ka < 1: return torch.zeros(q.shape[0])
    with torch.no_grad():
        idx = torch.topk(torch.cdist(q, keys), ka, dim=1, largest=False).indices
    kk = keys[idx]
    with torch.set_grad_enabled(grad):
        td = torch.linalg.vector_norm(q.unsqueeze(1) - kk, dim=2)
        tv = vals[idx]
        lg = -td / max(temp, 1e-6); lg = lg - lg.max(1, keepdim=True).values.detach()
        w = torch.exp(lg); w = w / (w.sum(1, keepdim=True) + 1e-10)
        if ka >= 3:
            wn = w / (w.sum(1, keepdim=True) + 1e-8)
            c0 = (wn * tv).sum(1, keepdim=True)
            mad = torch.clamp((wn * (tv - c0).abs()).sum(1, keepdim=True), min=1e-6)
            tv = torch.min(torch.max(tv, c0 - CLIP * mad), c0 + CLIP * mad)
        return (w * tv).sum(1) / torch.clamp(w.sum(1), min=1e-8)

def run(drift, adapt, seed=0, eta=None):
    U = V = None
    if adapt:
        g = torch.Generator().manual_seed(seed)
        U = torch.randn(D, RANK, generator=g) * SIG0
        V = torch.zeros(D, RANK)
    lk = torch.zeros(long_buf, D); lv = torch.zeros(long_buf); lp = ls = 0
    sk = torch.zeros(BUF, D); sv = torch.zeros(BUF); sp = ss = 0
    eb = torch.zeros(DW); ep = ec = 0
    final = pdn.clone(); diverged = None; om_trace = []

    for ci, st in enumerate(range(0, N, chunk)):
        en = min(st + chunk, N); B = en - st
        if drift and ec >= DW:
            h = DW // 2; ae = torch.roll(eb, -ep)
            sig = float(ae[h:].mean()) / max(float(ae[:h].mean()), 1e-8)
            om = W0 * float(torch.sigmoid(torch.tensor(ALPHA * (sig - 1.0))))
        else:
            om = OMEGA_FIXED
        om_trace.append(om)

        raw = Ew[st:en]
        if adapt: U.requires_grad_(True); V.requires_grad_(True)
        with torch.set_grad_enabled(bool(adapt)):
            q = raw if U is None else proj(raw, U, V)
            corr = None
            if ls >= 100:
                lkeys = lk[:ls]; skeys = sk[:ss]
                lc = knn(q, lkeys, lv[:ls], OK, LT, bool(adapt))
                if ss >= min(OK, 5):
                    sc = knn(q, skeys, sv[:ss], min(OK, ss), ST, bool(adapt))
                    corr = om * sc + (1 - om) * lc
                else: corr = lc
            pred = pdn[st:en] if corr is None else pdn[st:en] - OS_ * corr

        Un = Vn = None
        if adapt and corr is not None and B > 1:
            with torch.enable_grad():
                loss = ((pred - Yt[st:en]) ** 2).mean()
            gU, gV = torch.autograd.grad(loss, [U, V], allow_unused=True)
            gU = torch.zeros_like(U) if gU is None else gU
            gV = torch.zeros_like(V) if gV is None else gV
            with torch.no_grad():
                Un = (U - eta * (gU + WD * U)).detach(); Vn = (V - eta * (gV + WD * V)).detach()

        final[st:en] = pred.detach()
        store = q.detach() if U is not None else raw
        for j in range(B):
            i = st + j; v = pdn[i] - Yt[i]
            lk[lp] = store[j]; lv[lp] = v; lp = (lp + 1) % long_buf; ls = min(ls + 1, long_buf)
            sk[sp] = store[j]; sv[sp] = v; sp = (sp + 1) % BUF; ss = min(ss + 1, BUF)
            eb[ep] = abs(float(final[i] - Yt[i])); ep = (ep + 1) % DW; ec = min(ec + 1, DW)
        if Un is not None:
            U, V = Un, Vn
            if diverged is None and not (torch.isfinite(U).all() and torch.isfinite(V).all()):
                diverged = ci; break
    if diverged is not None: return None, diverged, om_trace
    return metric(final), None, om_trace

t0 = time.time()
res = {}
base, _, om_fixed = run(drift=False, adapt=False)
ab, _, om_drift = run(drift=True, adapt=False)
res["base"] = base; res["adapbuf"] = ab
print(f"  base     (fixed omega, no adaptation) : {base:.8f}")
print(f"  AdapBuf  (drift-aware omega_t)        : {ab:.8f}   delta {ab-base:+.7f}", flush=True)
omt = np.array(om_drift)
print(f"           omega_t: mean {omt.mean():.4f}  sd {omt.std():.4f}  "
      f"range [{omt.min():.4f}, {omt.max():.4f}]  (fixed arm held at {OMEGA_FIXED:.4f})", flush=True)

for tag, eta in [("recorded", ETA), ("eta=1e-1", ETA_HI)]:
    for name, drift in [("AdapEmb", False), ("Combined", True)]:
        vals, div = [], 0
        for s in SEEDS:
            m, d, _ = run(drift=drift, adapt=True, seed=s, eta=eta)
            if m is None: div += 1
            else: vals.append(m)
        if not vals:
            print(f"  {name:<8} ({tag:<8})                    : step too large ({div}/{len(SEEDS)})", flush=True)
            res[f"{name}_{tag}"] = None; continue
        v = np.array(vals); ref = base if name == "AdapEmb" else ab
        print(f"  {name:<8} ({tag:<8})                    : {v.mean():.8f} +- {v.std(ddof=1):.2e}   "
              f"delta vs base {v.mean()-base:+.7f}   delta vs {'base' if name=='AdapEmb' else 'AdapBuf'} "
              f"{v.mean()-ref:+.7f}", flush=True)
        res[f"{name}_{tag}"] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                    delta_base=float(v.mean()-base), delta_ref=float(v.mean()-ref),
                                    n=len(vals), diverged=div)

res.update(dataset=ds, task="classification" if is_cls else "regression",
           omega_fixed=OMEGA_FIXED, base_mix_w=W0, alpha=ALPHA, drift_window=DW,
           rank=RANK, eta_recorded=ETA, eta_high=ETA_HI, wd=WD, seeds=SEEDS,
           omega_trace_stats=dict(mean=float(omt.mean()), sd=float(omt.std()),
                                  lo=float(omt.min()), hi=float(omt.max())))
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nelapsed {time.time()-t0:.0f}s   saved {OUT}", flush=True)
