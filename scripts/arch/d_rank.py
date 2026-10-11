"""Build architecture D for the 2026 RANKING set. s230 -> a shippable component.

Trains on all 2,084,659 labelled 2025 departures (the ranking build trains on all of 2025,
CLAUDE.md) with the 49-feature ranking spec, and predicts the 344,841 scored rows.

The window is built the same way on both sides, and it is legitimate on the scored set because
`ranking.parquet` carries 344,693 ARRIVALS with TAXITIME_SEC_mvt present on 100% of them
(median taxi-in 478 s) alongside the 344,841 departures whose labels are blanked. Neighbour
departure labels are never read.
"""
import os, sys, time, glob, json, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn
import harness as H
SP=os.environ['DC26_WORK']; DATA=os.environ['DC26_DATA']
K=16; NF=6; SEEDS=int(os.environ.get('SEEDS','8')); EPOCHS=int(os.environ.get('EPOCHS','14'))
US=10**6

def windows(mv, tgt_ids, tag):
    """mv: all movements (dep+arr) with t, apt, arr, taxi, rwy, sdn. tgt_ids: rows to build for."""
    cache=f'{SP}/arch_d_win_{tag}_K{K}.npz'
    if os.path.exists(cache):
        z=np.load(cache); assert len(z['ids'])==len(tgt_ids) and (z['ids']==tgt_ids).all()
        print(f'  loaded {cache}'); return z['W'],z['M']
    W=np.zeros((len(tgt_ids),K,NF),np.float32); M=np.zeros((len(tgt_ids),K),np.float32)
    off=np.array([o for o in range(-(K//2),K//2+1) if o!=0][:K])
    pos=pd.Series(np.arange(len(tgt_ids)),index=tgt_ids)
    t0=time.time()
    for apt,g in mv.groupby('apt',sort=False):
        g=g.sort_values('t'); n=len(g)
        tt=g.t.values.astype(np.float64); arr=g.arr.values.astype(np.float32)
        taxi=g.taxi.values; rwy=pd.factorize(g.rwy.values)[0]; sdn=g.sdn.values
        obs=np.where(arr>0,taxi,np.nan)                 # ONLY arrival taxi is observable
        r=pos.reindex(g.MVT_ID_mvt.values).values
        keep=np.where(~np.isnan(r))[0]
        if not len(keep): continue
        J=keep[:,None]+off[None,:]; ok=(J>=0)&(J<n); Jc=np.clip(J,0,n-1)
        ot=obs[Jc]
        feat=np.stack([np.clip((tt[Jc]-tt[keep][:,None])/1800.,-4,4), arr[Jc],
                       np.where(np.isnan(ot),0.,np.clip(ot/1800.,0,8)),
                       (~np.isnan(ot)).astype(np.float32),
                       (rwy[Jc]==rwy[keep][:,None]).astype(np.float32),
                       np.clip(sdn[Jc]/3600.,-6,6)],-1).astype(np.float32)*ok[...,None]
        rows=r[keep].astype(int); W[rows]=feat; M[rows]=ok.astype(np.float32)
    # DELL-PERF 3(d) only, not 3(c): see d_window.py, D is shipped and stays fp32.
    np.savez(cache,W=W,M=M,ids=np.asarray(tgt_ids))
    print(f'  built {tag}: {W.shape} filled {M.mean():.1%} [{time.time()-t0:.0f}s]')
    return W,M

def stream(df, phase_col='PHASE_mvt'):
    d=df.copy()
    d['t']=d.MVT_TIME_UTC_mvt.astype('int64')//US
    d['apt']=d.ADEP_mvt.where(d[phase_col]=='DEP',d.ADES_mvt).astype(str)
    d['arr']=(d[phase_col]=='ARR').astype(np.float32)
    d['sdn']=(d.t-d.SCHED_TIME_UTC_mvt.astype('int64')//US).astype(np.float32)
    d['rwy']=d.RUNWAY_mvt.astype(str); d['taxi']=d.TAXITIME_SEC_mvt.astype(float)
    return d

C=['MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt',
   'TAXITIME_SEC_mvt','RUNWAY_mvt']
print('building 2025 movement stream...')
m25=stream(pd.concat([pq.read_table(f,columns=C).to_pandas()
    for f in sorted(glob.glob(f'{DATA}/training_2025-*.parquet'))],ignore_index=True))
assert 1.735e9<np.median(m25.t)<1.77e9, 'train epoch not 2025'
print('building 2026 movement stream...')
m26=stream(pq.read_table(f'{DATA}/ranking.parquet',columns=C).to_pandas())
assert 1.76e9<np.median(m26.t)<1.80e9, 'ranking epoch not 2026'
print(f'  2025 {len(m25):,} movements ({int(m25.arr.sum()):,} arr)   '
      f'2026 {len(m26):,} ({int(m26.arr.sum()):,} arr)')
print(f'  2026 arrival taxi non-null: {m26[m26.arr>0].taxi.notna().mean():.1%}')

cj=json.load(open(f'{SP}/v15_cols_jul19.json')); F,CATS=cj['F'],cj['CATS']
NUMF=[x for x in F if x not in CATS]
tr=pd.read_parquet(f'{SP}/v15_tr_jul19.parquet'); rk=pd.read_parquet(f'{SP}/v15_rk_jul19.parquet')
print(f'\nfeature frames: tr {len(tr):,}  rk {len(rk):,}   {len(NUMF)} numeric + {len(CATS)} cat')
Wt,Mt=windows(m25,tr.MVT_ID_mvt.values,'tr25')
Wr,Mr=windows(m26,rk.MVT_ID_mvt.values,'rk26')
print(f'  window filled: train {Mt.mean():.1%}  ranking {Mr.mean():.1%}')

ext=(tr.y>=7200).values; trn=tr[~ext]; Wtn=Wt[~ext]; Mtn=Mt[~ext]
nt=trn.naive.values.astype(float); nr=rk.naive.values.astype(float)
bt=np.where(~np.isnan(nt),nt,trn.apt_hr_naive.values.astype(float))
br=np.where(~np.isnan(nr),nr,rk.apt_hr_naive.values.astype(float))
g=np.nanmedian(bt); bt=np.nan_to_num(bt,nan=g); br=np.nan_to_num(br,nan=g)
cards=[];CT=[];CR=[]
for x in CATS:
    u=pd.Index(trn[x].astype(str).unique()); cards.append(len(u)+1)
    CT.append(pd.Index(u).get_indexer(trn[x].astype(str))+1)
    CR.append(pd.Index(u).get_indexer(rk[x].astype(str))+1)
CT=np.clip(np.vstack(CT).T,0,None); CR=np.clip(np.vstack(CR).T,0,None)
Xt=trn[NUMF].to_numpy(np.float64); Xr=rk[NUMF].to_numpy(np.float64)
med=np.nanmedian(Xt,0); MKt=np.isnan(Xt).astype(np.float32); MKr=np.isnan(Xr).astype(np.float32)
Xt=np.where(np.isnan(Xt),med,Xt); Xr=np.where(np.isnan(Xr),med,Xr)
q1,q9=np.percentile(Xt,1,0),np.percentile(Xt,99,0); sc=np.where(q9-q1<1e-9,1,q9-q1)
Xt=np.hstack([np.clip((Xt-q1)/sc,-5,5),MKt]).astype(np.float32)
Xr=np.hstack([np.clip((Xr-q1)/sc,-5,5),MKr]).astype(np.float32)
rt=(trn.y.values-bt).astype(np.float32); rmn,rs=rt.mean(),rt.std()
print(f'  train rows {len(Xt):,}  features {Xt.shape[1]}  residual mean {rmn:.1f} sd {rs:.1f}')

class WindowNet(nn.Module):
    def __init__(s,nf,cards,nfw,h=384,dh=128,heads=4):
        super().__init__()
        s.emb=nn.ModuleList([nn.Embedding(int(c),min(24,(int(c)+1)//2)) for c in cards])
        e=sum(min(24,(int(c)+1)//2) for c in cards)
        s.stem=nn.Sequential(nn.Linear(nf+e,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(.15))
        s.item=nn.Sequential(nn.Linear(nfw,dh),nn.GELU(),nn.Linear(dh,dh),nn.GELU())
        s.q=nn.Linear(h,dh); s.att=nn.MultiheadAttention(dh,heads,batch_first=True)
        s.out=nn.Sequential(nn.Linear(h+dh,256),nn.GELU(),nn.BatchNorm1d(256),
                            nn.Dropout(.10),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x,c,w,m):
        z=s.stem(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(s.emb)],1))
        it=s.item(w); pad=(m<0.5)
        pad=torch.where(pad.all(1,keepdim=True),torch.zeros_like(pad),pad)
        a,_=s.att(s.q(z).unsqueeze(1),it,it,key_padding_mask=pad)
        return s.out(torch.cat([z,a.squeeze(1)],1))

Xtt,Ctt,Wtt,Mtt,rr=H.to_dev(torch.tensor(Xt),torch.tensor(CT,dtype=torch.long),
    torch.tensor(Wtn),torch.tensor(Mtn),torch.tensor((rt-rmn)/rs))
Xrr,Crr,Wrr,Mrr=H.to_dev(torch.tensor(Xr),torch.tensor(CR,dtype=torch.long),
    torch.tensor(Wr),torch.tensor(Mr))
print(f'  resident on {H.DEV}: {sum(t.element_size()*t.nelement() for t in (Xtt,Wtt))/2**30:.2f} GB')
ps=[]; t0=time.time()
for seed in range(SEEDS):
    torch.manual_seed(seed); mdl=WindowNet(Xt.shape[1],cards,NF).to(H.DEV)
    opt=torch.optim.AdamW(mdl.parameters(),lr=2e-3,weight_decay=1e-5)
    N=len(Xtt); B=16384; sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,
        total_steps=EPOCHS*(N//B+1))
    for ep in range(EPOCHS):
        mdl.train(); perm=torch.randperm(N,device=H.DEV)
        for i in range(0,N,B):
            j=perm[i:i+B]
            if len(j)<2: continue
            opt.zero_grad()
            nn.functional.mse_loss(mdl(Xtt[j],Ctt[j],Wtt[j],Mtt[j]).squeeze(1),rr[j]).backward()
            opt.step(); sch.step()
    mdl.eval()
    with torch.no_grad():
        o=torch.cat([mdl(Xrr[i:i+65536],Crr[i:i+65536],Wrr[i:i+65536],Mrr[i:i+65536]).squeeze(1).float().cpu()
                     for i in range(0,len(Xrr),65536)]).numpy().astype(np.float64)
    ps.append(br+(o*rs+rmn))
    print(f'  seed {seed}: ranking pred median {np.median(ps[-1]):.0f}s  [{time.time()-t0:.0f}s]',flush=True)
pred=np.mean(ps,0)
print(f'\nD on the ranking set: median {np.median(pred):.0f}s  p1 {np.percentile(pred,1):.0f}  '
      f'p99 {np.percentile(pred,99):.0f}  min {pred.min():.0f}  max {pred.max():.0f}')
pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'d_window':pred}).to_parquet(
    f'{SP}/d_window_rk.parquet',index=False)
print(f'\nsaved {SP}/d_window_rk.parquet   total {time.time()-t0:.0f}s')
# diagnostic only, AFTER the output is written: it reads two other stages' files, and until 2026-09-23 it ran
# first, so a missing v55_trees.parquet discarded the whole GPU run (clean-room find)
if os.path.exists(f'{SP}/v55_trees.parquet') and os.path.exists(f'{SP}/v15_net_jul19.parquet'):
    sh=pd.read_parquet(f'{SP}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(rk.MVT_ID_mvt)
    nn_=pd.read_parquet(f'{SP}/v15_net_jul19.parquet').set_index('MVT_ID_mvt').reindex(rk.MVT_ID_mvt)
    old=0.2*sh.lgbm_opdi.values+0.4*nn_.mlp.values+0.4*sh.perapt_opdi.values
    print(f'  shipped bulk median {np.median(old):.0f}s   corr(D, shipped bulk) '
          f'{np.corrcoef(pred,old)[0,1]:.4f}')
