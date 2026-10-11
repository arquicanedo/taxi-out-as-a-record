"""B4: the learned provenance gate for the 2026 RANKING set.

s230 built it, s232 measured it against the FULL hand-built chain on validation and it won by
16.690 s -- against a baseline CLAUDE.md says is handicapped on extremes (raw Pc rather than
v29's inverted P; 46 features against the ranking build's 49), so that figure is an UPPER
bound. This produces the artefact so the board can settle it.

Trained on all 2,084,659 labelled 2025 departures with the 49-feature ranking spec, matching
how every shipped component was built. Calibrated probabilities, no class weighting -- balanced
weights reach 87,016x and kill it (s230). Seeds aggregated by GEOMETRIC mean, because arithmetic
averaging shrinks near-binary probabilities toward the middle while the mixture components sit
40,000 s apart (CLAUDE.md).
"""
import os, sys, time, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import harness as H
SP=os.environ['DC26_WORK']
SEEDS=int(os.environ.get('SEEDS','8')); EPOCHS=int(os.environ.get('EPOCHS','14'))
THR=7200; FLOOR,CAP,DAY=60.,200000.,86400.
cj=json.load(open(f'{SP}/v15_cols_jul19.json')); F_,CATS=cj['F'],cj['CATS']
NUMF=[x for x in F_ if x not in CATS]
tr=pd.read_parquet(f'{SP}/v15_tr_jul19.parquet'); rk=pd.read_parquet(f'{SP}/v15_rk_jul19.parquet')
print(f'tr {len(tr):,}  rk {len(rk):,}   {len(NUMF)} numeric + {len(CATS)} cat')
y=tr.y.values.astype(float)
sd_t=tr.sched_delay.values.astype(float); lb_t=tr.mvt_minus_lobt.values.astype(float)
def classify(yy,s,l):
    k=np.zeros(len(yy),dtype=np.int64); e=yy>=THR
    a=np.abs(yy-s)<=600; b=np.isfinite(l)&(np.abs(yy-l)<=600); r=(yy-DAY>=0)&(yy-DAY<=7200)
    k[e]=4; k[e&a]=1; k[e&b&~a]=2; k[e&r]=3; return k
kt=classify(y,sd_t,lb_t)
names=['ordinary','SCHED copy','LOBT stale','date error','other extreme']
print('\nprovenance classes on all 2025:')
for i,n in enumerate(names): print(f'  {i} {n:<16}{int((kt==i).sum()):>10,}')
cards=[];CT=[];CR=[]
for x in CATS:
    u=pd.Index(tr[x].astype(str).unique()); cards.append(len(u)+1)
    CT.append(pd.Index(u).get_indexer(tr[x].astype(str))+1)
    CR.append(pd.Index(u).get_indexer(rk[x].astype(str))+1)
CT=np.clip(np.vstack(CT).T,0,None); CR=np.clip(np.vstack(CR).T,0,None)
Xt=tr[NUMF].to_numpy(np.float64); Xr=rk[NUMF].to_numpy(np.float64)
med=np.nanmedian(Xt,0); Mt=np.isnan(Xt).astype(np.float32); Mr=np.isnan(Xr).astype(np.float32)
Xt=np.where(np.isnan(Xt),med,Xt); Xr=np.where(np.isnan(Xr),med,Xr)
q1,q9=np.percentile(Xt,1,0),np.percentile(Xt,99,0); sc=np.where(q9-q1<1e-9,1,q9-q1)
Xt=np.hstack([np.clip((Xt-q1)/sc,-5,5),Mt]).astype(np.float32)
Xr=np.hstack([np.clip((Xr-q1)/sc,-5,5),Mr]).astype(np.float32)
Xtt,Ctt,ktt=H.to_dev(torch.tensor(Xt),torch.tensor(CT,dtype=torch.long),torch.tensor(kt))
Xrr,Crr=H.to_dev(torch.tensor(Xr),torch.tensor(CR,dtype=torch.long))
print(f'  features {Xt.shape[1]}   resident {Xtt.element_size()*Xtt.nelement()/2**30:.2f} GB on {H.DEV}')
Ps=[]; t0=time.time()
for s_ in range(SEEDS):
    m=H.Trunk(Xt.shape[1],cards,out=5).to(H.DEV)
    H.train(m,Xtt,Ctt,[ktt],lambda o,t:F.cross_entropy(o[:,:5],t[0]),epochs=EPOCHS,seed=s_)
    Ps.append(torch.softmax(H.infer(m,Xrr,Crr),1).numpy().astype(np.float64))
    print(f'  seed {s_}: mean P {Ps[-1].mean(0).round(6)}  [{time.time()-t0:.0f}s]',flush=True)
P=np.exp(np.mean(np.log(np.clip(np.stack(Ps),1e-12,1)),0)); P/=P.sum(1,keepdims=True)
print(f'\ngeometric-mean gate over {SEEDS} seeds. mean P by class: {P.mean(0).round(6)}')
for i,n in enumerate(names):
    print(f'  {n:<16} rows with P>0.5: {int((P.argmax(1)==i).sum()):>7,}')
pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,
              **{f'P{i}':P[:,i] for i in range(5)}}).to_parquet(f'{SP}/b4_gate_rk.parquet',index=False)
print(f'\nsaved {SP}/b4_gate_rk.parquet   total {time.time()-t0:.0f}s')
