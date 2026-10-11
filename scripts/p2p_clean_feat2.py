"""p2p step 2 (clean-room member): VERBATIM copy of scripts/p1c_features2.py with three changes only -- the raw frame is
trackC's p1c_raw.parquet (read-only), outputs go to the p2P scratch directory with a _$SEL suffix, and
SEL=train keeps the 2025 departures (src<12, exactly as the original) while SEL=rank keeps the 2026 ranking file
(src==12), so the ranking-side features are built by the same code as the window-side ones."""
"""Track C: second feature batch (runway sequencing, queue-between-clocks counts, stand-area
congestion, inbound lateness). Observables and labelled arrivals only."""
import pandas as pd, numpy as np, time
t0=time.time()
import os; SEL=os.environ.get('SEL','rank'); RAW=os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackC/'; OUT=os.environ.get('P2P_OUT',os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')+'/clean/'; os.makedirs(OUT,exist_ok=True)
df=pd.read_parquet(RAW+'p1c_raw.parquet')
df=(df[df.src<12] if SEL=='train' else df[df.src==12]).reset_index(drop=True)
dep=df[df.dep].reset_index(drop=True); arr=df[~df.dep].reset_index(drop=True)
F=pd.read_parquet(OUT+f'feat1_{SEL}.parquet',columns=['MVT_ID_mvt'])
assert (F.MVT_ID_mvt.values==dep.MVT_ID_mvt.values).all()
G=pd.DataFrame({'MVT_ID_mvt':dep.MVT_ID_mvt.values})
t=dep.t_mvt.values
def groups(keys):
    k=pd.Series(keys).astype('string').fillna('NA').values
    u,inv=np.unique(k,return_inverse=True); return u,inv
def per_key(qkey,rkey):
    uq,iq=groups(qkey); ur,ir=groups(rkey)
    rmap={u:i for i,u in enumerate(ur)}
    return uq,iq,ur,ir,rmap
def gap_prev_next(qkey,tq,rkey,tr,self_same):
    """time since previous and until next event of reference set with same key (excluding self if same set)"""
    uq,iq,ur,ir,rmap=per_key(qkey,rkey)
    gp=np.full(len(tq),np.nan); gn=np.full(len(tq),np.nan)
    order=np.argsort(ir,kind='stable'); irs=ir[order]; bounds=np.searchsorted(irs,np.arange(len(ur)+1))
    for qi,u in enumerate(uq):
        if u not in rmap: continue
        ri=rmap[u]; T=np.sort(tr[order[bounds[ri]:bounds[ri+1]]])
        m=np.where(iq==qi)[0]; x=tq[m]
        a=np.searchsorted(T,x,'left'); b=np.searchsorted(T,x,'right')
        # previous strictly before (for same set, skip self: use a-1 where T[a]==x is self)
        pv=np.where(a>0,x-T[np.maximum(a-1,0)],np.nan)
        nx=np.where(b<len(T),T[np.minimum(b,len(T)-1)]-x,np.nan)
        gp[m]=pv; gn[m]=nx
    return gp,gn
def count_between(qkey,lo_t,hi_t,rkey,tr):
    uq,iq,ur,ir,rmap=per_key(qkey,rkey)
    out=np.full(len(lo_t),np.nan)
    order=np.argsort(ir,kind='stable'); irs=ir[order]; bounds=np.searchsorted(irs,np.arange(len(ur)+1))
    for qi,u in enumerate(uq):
        if u not in rmap: continue
        ri=rmap[u]; T=np.sort(tr[order[bounds[ri]:bounds[ri+1]]])
        m=np.where(iq==qi)[0]; lo=lo_t[m]; hi=hi_t[m]
        ok=np.isfinite(lo)&np.isfinite(hi)
        c=np.searchsorted(T,np.where(ok,hi,0),'left')-np.searchsorted(T,np.where(ok,lo,0),'left')
        out[m]=np.where(ok,c,np.nan)
    return out
drk=(dep.apt+'_'+dep.RUNWAY_mvt.fillna('NA')).values; ark=(arr.apt+'_'+arr.RUNWAY_mvt.fillna('NA')).values
G['rw_gap_prev_dep'],G['rw_gap_next_dep']=gap_prev_next(drk,t,drk,t,True)
G['rw_gap_prev_arr'],G['rw_gap_next_arr']=gap_prev_next(drk,t,ark,arr.t_mvt.values,False)
G['ap_gap_prev_dep'],G['ap_gap_next_dep']=gap_prev_next(dep.apt.values,t,dep.apt.values,t,True)
print('gaps',round(time.time()-t0,1),flush=True)
# queue-between-clocks: takeoffs (same apt / same runway) between our clock X and our takeoff
for nm,lo in [('sch',dep.t_sch.values),('aobt',dep.t_aobt.values),('eobt',dep.t_eobt.values)]:
    lo2=np.where(lo<t,lo,np.nan)
    G['q_ap_'+nm]=count_between(dep.apt.values,lo2,t,dep.apt.values,t)
    G['q_rw_'+nm]=count_between(drk,lo2,t,drk,t)
    G['qa_rw_'+nm]=count_between(drk,lo2,t,ark,arr.t_mvt.values)
print('queues',round(time.time()-t0,1),flush=True)
# stand-area (stand prefix) congestion
stp=(dep.apt+'_'+dep.STAND_mvt.fillna('NA').str.extract(r'^([A-Za-z]*\d?)',expand=False).fillna('NA')).values
def win_count(qkey,tq,rkey,tr,lo,hi,self_excl):
    uq,iq,ur,ir,rmap=per_key(qkey,rkey)
    out=np.zeros(len(tq))
    order=np.argsort(ir,kind='stable'); irs=ir[order]; bounds=np.searchsorted(irs,np.arange(len(ur)+1))
    for qi,u in enumerate(uq):
        if u not in rmap: continue
        ri=rmap[u]; T=np.sort(tr[order[bounds[ri]:bounds[ri+1]]])
        m=np.where(iq==qi)[0]; x=tq[m]
        c=np.searchsorted(T,x+hi,'left')-np.searchsorted(T,x+lo,'left')
        if self_excl: c=c-1
        out[m]=c
    return out
G['stp_nd_pm15']=win_count(stp,t,stp,t,-900,900,True)
G['stp_ns_pm15']=win_count(stp,dep.t_sch.values,stp,dep.t_sch.values,-900,900,True)
astp=(arr.apt+'_'+arr.STAND_mvt.fillna('NA').str.extract(r'^([A-Za-z]*\d?)',expand=False).fillna('NA')).values
G['stp_nain_pm15']=win_count(stp,t,astp,arr.t_blk.values,-900,900,False)
print('stand area',round(time.time()-t0,1),flush=True)
# inbound (last arrival in-block at same stand before our SCHED) lateness and its clocks
q=pd.DataFrame({'_i':np.arange(len(dep)),'key':(dep.apt+'_'+dep.STAND_mvt.fillna('NA')).values,'t':dep.t_sch.values})
A=pd.DataFrame({'key':(arr.apt+'_'+arr.STAND_mvt.fillna('NA')).values,'t':arr.t_blk.values,'tin':arr.t_blk.values,'asch':arr.t_sch.values,'aland':arr.t_mvt.values})
A=A.dropna(subset=['t']).sort_values('t'); q=q.sort_values('t')
m=pd.merge_asof(q,A,on='t',by='key',direction='backward').sort_values('_i')
G['ib_sch_m_in']=dep.t_sch.values-m.tin.values          # planned turnaround slack at stand
G['ib_late']=m.tin.values-m.asch.values                  # inbound in-block lateness vs its schedule
G['ib_mvt_m_in']=t-m.tin.values
print(G.describe().T[['count','mean','50%']].round(1).to_string())
G.to_parquet(OUT+f'feat2_{SEL}.parquet',index=False)
print('done',round(time.time()-t0,1))
