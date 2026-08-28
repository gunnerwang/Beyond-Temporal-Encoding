#!/usr/bin/env python3
"""Rebuild the per-sample classification predictions and put a DeLong CI on the AUC.

No tuning is repeated. The ensemble weights and kNN parameters are read back from
analysis/<dataset>/classification.json and the pipeline is run once, so the score
vector behind the published AUC is materialised and can be re-used.

Then: DeLong (1988) structural-components standard error and 95% CI for each AUC, and the
published gap to the strongest competing method expressed in units of that SE.
"""
import numpy as np, json, os, sys
from sklearn.metrics import roc_auc_score

MAX_BUF = 8192
OUT = "results/cls_predictions"
os.makedirs(OUT, exist_ok=True)

def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    out=np.empty(N,float); out[J]=T; return out

def delong(y, s):
    y=np.asarray(y).astype(int); s=np.asarray(s,float)
    pos=s[y==1]; neg=s[y==0]; m,n=len(pos),len(neg)
    tx=midrank(pos); ty=midrank(neg); tz=midrank(np.concatenate([pos,neg]))
    auc=(tz[:m].sum()/m-(m+1)/2)/n
    v01=(tz[:m]-tx)/n; v10=1-(tz[m:]-ty)/m
    return auc, v01.var(ddof=1)/m + v10.var(ddof=1)/n

def knn_batch(q_all, keys, vals, k, temp, clip):
    nk=keys.shape[0]; ka=min(k,nk); res=np.zeros(q_all.shape[0],np.float32)
    for b in range(0,q_all.shape[0],256):
        e=min(b+256,q_all.shape[0]); q=q_all[b:e]
        sq=np.maximum(np.sum(q**2,1,keepdims=True)+np.sum(keys**2,1)-2*q@keys.T,0)
        d=np.sqrt(sq+1e-10)
        ti=np.argsort(d,1)[:,:ka] if ka>=nk else np.argpartition(d,ka,1)[:,:ka]
        td=np.take_along_axis(d,ti,1); tv=vals[ti]
        lo=-td/max(temp,1e-6); lo-=lo.max(1,keepdims=True)
        w=np.exp(lo); w/=w.sum(1,keepdims=True)+1e-10
        if ka>=3:
            wn=w/(w.sum(1,keepdims=True)+1e-8); c0=np.sum(wn*tv,1,keepdims=True)
            mad=np.maximum(np.sum(wn*np.abs(tv-c0),1,keepdims=True),1e-6)
            tv=np.clip(tv,c0-clip*mad,c0+clip*mad)
        res[b:e]=np.sum(w*tv,1)/np.maximum(w.sum(1),1e-8)
    return res

PAPER={"homesite-insurance":(0.9652,"TARS TabM",0.9662),
       "ecom-offers":(0.6373,"TARS FT-T",0.6372),
       "homecredit-default":(0.8697,"XGBoost",0.8670)}
rows=[]
for ds,(auc_paper,rival,auc_rival) in PAPER.items():
    R=json.load(open(f"analysis/{ds}/classification.json"))
    cfg=R["results"]["tree_leaf"]; W=cfg["weights"]; K=cfg["knn_params"]
    Y=np.load(f"tabred/data/{ds}/Y.npy").astype(np.float32)
    te=np.load(f"tabred/data/{ds}/split-default/test_idx.npy"); Y_test=Y[te]; N=len(Y_test)
    priors={}
    for nm in ("xgb","catboost","lightgbm"):
        p=f"cache/{nm}/{ds}/{nm}_predictions_test_raw.npy"
        if os.path.exists(p) and nm in W: priors[nm]=np.load(p).astype(np.float32)
    emb=np.load(f"cache/tree_leaf/{ds}/leaf_embeddings_test.npy").astype(np.float32)
    ws=sum(W[n] for n in priors); p_ens=sum((W[n]/ws)*priors[n] for n in priors)

    l1=p_ens.copy(); D=emb.shape[1]; bs=int(K["buf_size"])
    lk=np.zeros((MAX_BUF,D),np.float32); lv=np.zeros(MAX_BUF,np.float32); lp=ls=0
    sk=np.zeros((bs,D),np.float32); sv=np.zeros(bs,np.float32); sp=ss=0
    for st in range(0,N,64):
        en=min(st+64,N); b=emb[st:en]
        if ls>=100:
            lc=knn_batch(b,lk[:ls],lv[:ls],int(K["online_k"]),K["long_temp"],K["knn_clip"])
            if ss>=min(int(K["online_k"]),5):
                sc=knn_batch(b,sk[:ss],sv[:ss],min(int(K["online_k"]),ss),K["short_temp"],K["knn_clip"])
                corr=K["dual_mix"]*sc+(1-K["dual_mix"])*lc
            else: corr=lc
            l1[st:en]-=K["online_scale"]*corr
        for j in range(en-st):
            i=st+j; v=p_ens[i]-Y_test[i]
            lk[lp]=emb[i]; lv[lp]=v; lp=(lp+1)%MAX_BUF; ls=min(ls+1,MAX_BUF)
            sk[sp]=emb[i]; sv[sp]=v; sp=(sp+1)%bs; ss=min(ss+1,bs)
    pred=np.clip(l1,0,1)
    np.save(f"{OUT}/{ds}_pred.npy",pred); np.save(f"{OUT}/{ds}_y.npy",Y_test)

    auc,var=delong(Y_test,pred); se=np.sqrt(var)
    m,n=int((Y_test==1).sum()),int((Y_test==0).sum())
    gap=auc_paper-auc_rival
    print(f"=== {ds} ===")
    print(f"  rebuilt AUC {auc:.6f}   (saved in results json: {cfg['auc']:.6f}; paper: {auc_paper})")
    print(f"  n={N}  n_pos={m}  n_neg={n}")
    print(f"  DeLong SE = {se:.5f}   95% CI = [{auc-1.96*se:.4f}, {auc+1.96*se:.4f}]")
    print(f"  gap vs {rival} ({auc_rival}) = {gap:+.4f}  ->  {abs(gap)/se:.2f} x SE\n")
    rows.append(dict(dataset=ds,n=N,n_pos=m,n_neg=n,auc_rebuilt=float(auc),
                     auc_results_json=cfg["auc"],auc_paper=auc_paper,delong_se=float(se),
                     ci=[float(auc-1.96*se),float(auc+1.96*se)],rival=rival,auc_rival=auc_rival,
                     gap=float(gap),gap_over_se=float(abs(gap)/se),
                     pred_file=f"{OUT}/{ds}_pred.npy"))
json.dump(rows,open("results/delong_auc_ci.json","w"),indent=2)
print("saved results/delong_auc_ci.json  +  per-sample predictions in", OUT)
