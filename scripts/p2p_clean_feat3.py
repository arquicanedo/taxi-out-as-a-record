"""p2p step 2 (clean-room member): VERBATIM copy of scripts/p1c_features3.py with three changes only -- the raw frame is
trackC's p1c_raw.parquet (read-only), outputs go to the p2P scratch directory with a _$SEL suffix, and
SEL=train keeps the 2025 departures (src<12, exactly as the original) while SEL=rank keeps the 2026 ranking file
(src==12), so the ranking-side features are built by the same code as the window-side ones."""
"""Track C: third feature batch -- takeoff-flow interruptions (ground stops / storms).
For each departure: time since, and takeoffs since, the end of the last airport-level (and
runway-level) takeoff gap longer than G; rolling max takeoff gap in the previous 2 h; same for
landings. Observables only."""
import pandas as pd, numpy as np, time
t0=time.time()
import os; SEL=os.environ.get('SEL','rank'); RAW=os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackC/'; OUT=os.environ.get('P2P_OUT',os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')+'/clean/'; os.makedirs(OUT,exist_ok=True)
df=pd.read_parquet(RAW+'p1c_raw.parquet',columns=['MVT_ID_mvt','src','dep','apt','RUNWAY_mvt','t_mvt'])
df=(df[df.src<12] if SEL=='train' else df[df.src==12]).reset_index(drop=True)
dep=df[df.dep].reset_index(drop=True); arr=df[~df.dep].reset_index(drop=True)
H=pd.DataFrame({'MVT_ID_mvt':dep.MVT_ID_mvt.values})
def flow_feats(qkey,tq,rkey,tr,prefix,self_in_ref):
    out={f'{prefix}_since_g{G}':np.full(len(tq),np.nan) for G in (600,1200,1800)}
    out.update({f'{prefix}_nsince_g{G}':np.full(len(tq),np.nan) for G in (600,1200,1800)})
    out[f'{prefix}_maxgap120']=np.full(len(tq),np.nan)
    qk=pd.Series(qkey).fillna('NA').values; rk=pd.Series(rkey).fillna('NA').values
    for k in np.unique(qk):
        m=np.where(qk==k)[0]; T=np.sort(tr[rk==k])
        if len(T)<2: continue
        g=np.r_[np.nan,np.diff(T)]
        # position of each query: last reference event with time <= tq (self included when same set)
        pos=np.searchsorted(T,tq[m],'right')-1
        ok=pos>=0; pos_c=np.maximum(pos,0)
        for G in (600,1200,1800):
            idx=np.where(g>G,np.arange(len(T)),-1); last=np.maximum.accumulate(idx)
            lj=last[pos_c]; good=ok&(lj>=0)
            out[f'{prefix}_since_g{G}'][m]=np.where(good,tq[m]-T[np.maximum(lj,0)],np.nan)
            out[f'{prefix}_nsince_g{G}'][m]=np.where(good,pos_c-lj,np.nan)
        s=pd.Series(np.nan_to_num(g,nan=0.0),index=pd.to_datetime(T,unit='s'))
        rmax=s.rolling('7200s').max().values
        out[f'{prefix}_maxgap120'][m]=np.where(ok,rmax[pos_c],np.nan)
    return out
t=dep.t_mvt.values
for k,v in flow_feats(dep.apt.values,t,dep.apt.values,t,'fa',True).items(): H[k]=v
drk=(dep.apt+'_'+dep.RUNWAY_mvt.fillna('NA')).values
for k,v in flow_feats(drk,t,drk,t,'fr',True).items(): H[k]=v
for k,v in flow_feats(dep.apt.values,t,arr.apt.values,arr.t_mvt.values,'fl',False).items(): H[k]=v
print(H.describe().T[['count','mean','50%','max']].round(1).to_string())
H.to_parquet(OUT+f'feat3_{SEL}.parquet',index=False)
print('done',round(time.time()-t0,1))
