"""p2p step 2 (clean-room member): VERBATIM copy of the track C research script p1c_features.py (not in the
release; this file is self-contained) with three changes only -- the raw frame is
trackC's p1c_raw.parquet (read-only), outputs go to the p2P scratch directory with a _$SEL suffix, and
SEL=train keeps the 2025 departures (src<12, exactly as the original) while SEL=rank keeps the 2026 ranking file
(src==12), so the ranking-side features are built by the same code as the window-side ones."""
"""Track C clean-room feature builder (2025 only). Reads the raw frame built by p1c_load.py.
Every feature is formable from ranking.parquet-style columns: departure observables plus
LABELLED ARRIVALS; no departure label enters any feature (target encodings are done later,
from training months only)."""
import pandas as pd, numpy as np, time
from zoneinfo import ZoneInfo
t0=time.time()
import os; SEL=os.environ.get('SEL','rank'); RAW=os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackC/'; OUT=os.environ.get('P2P_OUT',os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')+'/clean/'; os.makedirs(OUT,exist_ok=True)
df=pd.read_parquet(RAW+'p1c_raw.parquet')
df=(df[df.src<12] if SEL=='train' else df[df.src==12]).reset_index(drop=True)          # 2025 only
dep=df[df.dep].reset_index(drop=True)
arr=df[~df.dep].reset_index(drop=True)
print('dep',len(dep),'arr',len(arr),flush=True)
F=pd.DataFrame({'MVT_ID_mvt':dep.MVT_ID_mvt.values,'src':dep.src.values,'y':dep.y.values,'apt':dep.apt.values})
t=dep.t_mvt.values
# ---- clock differences
for k in ['sch','lobt','iobt','eobt','aobt']:
    F['d_'+k]=t-dep['t_'+k].values
F['aobt_m_eobt']=dep.t_aobt.values-dep.t_eobt.values
F['lobt_m_sch']=dep.t_lobt.values-dep.t_sch.values
F['eobt_m_lobt']=dep.t_eobt.values-dep.t_lobt.values
F['iobt_m_lobt']=dep.t_iobt.values-dep.t_lobt.values
F['fly1']=dep.t_arvt1.values-dep.t_eobt.values     # planned off-block to arrival
F['fly3']=dep.t_arvt3.values-t                     # takeoff to actual arrival
F['arvt3_m_arvt1']=dep.t_arvt3.values-dep.t_arvt1.values
F['plan_minus_act']=F['fly1']-(dep.t_arvt3.values-dep.t_aobt.values)
F['mvt_sec']=t%60
F['aobt_sec']=dep.t_aobt.values%60
F['sch_sec']=dep.t_sch.values%300
# ---- local time
lh=np.full(len(dep),np.nan); wd=np.full(len(dep),np.nan)
TZ={'EDDF':'Europe/Berlin','EDDM':'Europe/Berlin','EGLL':'Europe/London','EHAM':'Europe/Amsterdam','LEBL':'Europe/Madrid','LEMD':'Europe/Madrid','LFPG':'Europe/Paris','LIRF':'Europe/Rome','LSZH':'Europe/Zurich','LTFM':'Europe/Istanbul'}
for a,z in TZ.items():
    m=(dep.apt==a).values
    ts=pd.to_datetime(t[m],unit='s',utc=True).tz_convert(z)
    lh[m]=ts.hour+ts.minute/60.0; wd[m]=ts.dayofweek
    tss=pd.to_datetime(dep.t_sch.values[m],unit='s',utc=True).tz_convert(z)
    F.loc[m,'sch_lh']=tss.hour+tss.minute/60.0
F['loc_hour']=lh; F['wday']=wd
# ---- categoricals (codes over all 2025 departures)
def code(s):
    c=pd.Categorical(s.astype('string').fillna('NA'))
    return c.codes.astype('int32'), len(c.categories)
cat_src={'c_apt':dep.apt,'c_stand':dep.apt+'_'+dep.STAND_mvt.fillna('NA'),'c_rwy':dep.apt+'_'+dep.RUNWAY_mvt.fillna('NA'),
 'c_type':dep.AIRCRAFT_TYPE_mvt,'c_wtc':dep.WK_TBL_CAT_flt,'c_mseg':dep.MARKET_SEGMENT_flt,'c_ftype':dep.FLIGHT_TYPE_flt,
 'c_op':dep.AIRCRAFT_OPERATOR_flt,'c_cs3':dep.CALLSIGN_flt.str[:3],'c_fl2':dep.FLIGHT_mvt.str.extract(r'^([A-Z0-9]{2})',expand=False),
 'c_ades':dep.ADES_mvt,'c_stpre':dep.apt+'_'+dep.STAND_mvt.fillna('NA').str.extract(r'^([A-Za-z]*\d?)',expand=False).fillna('NA'),
 'c_rule':dep.FLIGHT_RULE_mvt,'c_adesfiled_diff':(dep.ADES_FILED_flt!=dep.ADES_mvt).astype('string'),
 'c_typematch':(dep.AIRCRAFT_TYPE_flt!=dep.AIRCRAFT_TYPE_mvt).astype('string')}
for k,s in cat_src.items():
    F[k],n=code(s); print(k,n,flush=True)
# high-card keys kept as strings for target encoding later
F['k_flight']=dep.apt+'_'+dep.FLIGHT_mvt.fillna('NA')
F['k_ssr']=dep.apt+'_'+dep.STAND_mvt.fillna('NA')+'_'+dep.RUNWAY_mvt.fillna('NA')
F['k_cs']=dep.apt+'_'+dep.CALLSIGN_flt.fillna('NA')
F['k_opst']=dep.apt+'_'+dep.AIRCRAFT_OPERATOR_flt.fillna('NA')+'_'+dep.STAND_mvt.fillna('NA')
F['k_dest']=dep.apt+'_'+dep.ADES_mvt.fillna('NA')
print('cats done',round(time.time()-t0,1),flush=True)
# ---- traffic counts
def counts(key_q,tq,key_r,tr,lo,hi,self_excl):
    out=np.zeros(len(tq),dtype='float32')
    kq=pd.Series(key_q).astype('string').values; kr=pd.Series(key_r).astype('string').values
    order_r=np.argsort(kr,kind='stable')
    kr_s=kr[order_r]; tr_s=tr[order_r]
    uk,starts=np.unique(kr_s,return_index=True); ends=np.r_[starts[1:],len(kr_s)]
    idx={k:(s,e) for k,s,e in zip(uk,starts,ends)}
    for k in np.unique(kq):
        m=np.where(kq==k)[0]
        if k not in idx: continue
        s,e=idx[k]; T=np.sort(tr_s[s:e])
        tq_k=tq[m]
        c=np.searchsorted(T,tq_k+hi,side='left')-np.searchsorted(T,tq_k+lo,side='left')
        if self_excl and lo<=0<hi: c=c-1
        out[m]=c
    return out
ta=arr.t_mvt.values
for lo,hi in [(-600,0),(-1200,0),(-1800,-600),(-3600,0),(0,600),(0,1200),(600,1800)]:
    F[f'nd_{lo//60}_{hi//60}']=counts(dep.apt.values,t,dep.apt.values,t,lo,hi,True)
    F[f'na_{lo//60}_{hi//60}']=counts(dep.apt.values,t,arr.apt.values,ta,lo,hi,False)
for lo,hi in [(-900,0),(0,900),(-1800,1800)]:
    F[f'ndr_{lo//60}_{hi//60}']=counts((dep.apt+dep.RUNWAY_mvt.fillna('')).values,t,(dep.apt+dep.RUNWAY_mvt.fillna('')).values,t,lo,hi,True)
    F[f'nar_{lo//60}_{hi//60}']=counts((dep.apt+dep.RUNWAY_mvt.fillna('')).values,t,(arr.apt+arr.RUNWAY_mvt.fillna('')).values,ta,lo,hi,False)
# scheduled demand around our schedule time, and flights scheduled before us still not airborne at our MVT
ts_=dep.t_sch.values
F['ns_pm15']=counts(dep.apt.values,ts_,dep.apt.values,ts_,-900,900,True)
print('counts done',round(time.time()-t0,1),flush=True)
# ---- windowed means of other departures' MVT-AOBT and arrivals' taxi-in (labelled arrivals are available in 2026)
def wmean(key_q,tq,key_r,tr,vr,lo,hi,self_excl,vq=None):
    out=np.full(len(tq),np.nan,dtype='float64'); cnt=np.zeros(len(tq))
    kq=pd.Series(key_q).astype('string').values; kr=pd.Series(key_r).astype('string').values
    for k in np.unique(kq):
        m=np.where(kq==k)[0]; mr=np.where((kr==k)&np.isfinite(vr))[0]
        if len(mr)==0: continue
        o=np.argsort(tr[mr]); T=tr[mr][o]; V=vr[mr][o]; C=np.r_[0,np.cumsum(V)]
        a=np.searchsorted(T,tq[m]+lo,'left'); b=np.searchsorted(T,tq[m]+hi,'left')
        s=C[b]-C[a]; n=(b-a).astype(float)
        if self_excl:
            sv=vq[m]; ok=np.isfinite(sv)&(lo<=0)&(0<hi)
            s=np.where(ok,s-np.nan_to_num(sv),s); n=np.where(ok,n-1,n)
        out[m]=np.where(n>0,s/np.maximum(n,1),np.nan); cnt[m]=n
    return out,cnt
dA=np.clip(F['d_aobt'].values,0,3600)
arr_tx=np.clip(arr.y.values,0,3600)
for lo,hi,nm in [(-1800,1800,'30'),(-3600,3600,'60'),(-10800,10800,'180')]:
    F['md_aobt_'+nm],_=wmean(dep.apt.values,t,dep.apt.values,t,dA,lo,hi,True,dA)
    F['ma_taxi_'+nm],F['na_lab_'+nm]=wmean(dep.apt.values,t,arr.apt.values,ta,arr_tx,lo,hi,False)
F['md_aobt_rwy_60'],_=wmean((dep.apt+dep.RUNWAY_mvt.fillna('')).values,t,(dep.apt+dep.RUNWAY_mvt.fillna('')).values,t,dA,-3600,3600,True,dA)
# arrivals' in-block minus landing lateness vs schedule (how late the day is running)
arr_late=np.clip(arr.t_blk.values-arr.t_sch.values,-3600,7200)
F['ma_late_60'],_=wmean(dep.apt.values,t,arr.apt.values,ta,arr_late,-3600,3600,False)
dsch=np.clip(F['d_sch'].values,-3600,10800)
F['md_dsch_60'],_=wmean(dep.apt.values,t,dep.apt.values,t,dsch,-3600,3600,True,dsch)
print('wmeans done',round(time.time()-t0,1),flush=True)
# ---- stand-linked: arrivals' in-block at the same stand (labelled arrivals), departures from same stand
F['_i']=np.arange(len(F))
q=pd.DataFrame({'_i':F['_i'].values,'key':(dep.apt+'_'+dep.STAND_mvt.fillna('NA')).values,'t':t,'tsch':ts_,'typ':dep.AIRCRAFT_TYPE_mvt.fillna('').values,'op':dep.AIRCRAFT_OPERATOR_flt.fillna('').values})
A=pd.DataFrame({'key':(arr.apt+'_'+arr.STAND_mvt.fillna('NA')).values,'tin':arr.t_blk.values,'tland':arr.t_mvt.values,'atyp':arr.AIRCRAFT_TYPE_mvt.fillna('').values,'aop':arr.AIRCRAFT_OPERATOR_flt.fillna('').values,'asch':arr.t_sch.values,'ataxi':arr.y.values})
A=A.dropna(subset=['tin']).sort_values('tin'); q=q.sort_values('t')
m1=pd.merge_asof(q,A.rename(columns={'tin':'t'}).assign(tin=lambda x:x['t']),on='t',by='key',direction='backward',allow_exact_matches=True)
m1=m1.sort_values('_i')
F['st_gap_in1']=t-m1.tin.values
F['st_in1_typ_same']=(m1.atyp.values==q.sort_values('_i').typ.values).astype('float32')
F['st_in1_op_same']=(m1.aop.values==q.sort_values('_i').op.values).astype('float32')
F['st_in1_taxi']=m1.ataxi.values
F['st_sch_m_in1']=ts_-m1.tin.values
# second-last arrival in-block at stand: shift within key
A2=A.copy(); A2['tin_prev']=A2.groupby('key').tin.shift(1); A2['atyp_prev']=A2.groupby('key').atyp.shift(1); A2['aop_prev']=A2.groupby('key').aop.shift(1)
m2=pd.merge_asof(q,A2[['key','tin','tin_prev','atyp_prev','aop_prev']].rename(columns={'tin':'t'}),on='t',by='key',direction='backward').sort_values('_i')
F['st_gap_in2']=t-m2.tin_prev.values
F['st_in2_typ_same']=(m2.atyp_prev.values==q.sort_values('_i').typ.values).astype('float32')
F['st_in2_op_same']=(m2.aop_prev.values==q.sort_values('_i').op.values).astype('float32')
# next arrival in-block at stand after our takeoff
m3=pd.merge_asof(q,A.rename(columns={'tin':'t'}).assign(tin=lambda x:x['t'])[['key','t','tin']],on='t',by='key',direction='forward',allow_exact_matches=False).sort_values('_i')
F['st_gap_next_in']=m3.tin.values-t
# departures from the same stand: previous and next takeoffs
Dq=q[['key','t','_i']].copy()
Dq=Dq.sort_values(['key','t']); Dq['tprev']=Dq.groupby('key').t.shift(1); Dq['tnext']=Dq.groupby('key').t.shift(-1)
Dq=Dq.sort_values('_i')
F['st_gap_dprev']=t-Dq.tprev.values
F['st_gap_dnext']=Dq.tnext.values-t
print('stand links done',round(time.time()-t0,1),flush=True)
# ---- same flight number: previous day's same-flight observables (no labels)
Fq=pd.DataFrame({'_i':F['_i'].values,'k':F['k_flight'].values,'t':t,'st':(dep.STAND_mvt.fillna('')).values,'daobt':F['d_aobt'].values})
Fq=Fq.sort_values(['k','t']); g=Fq.groupby('k')
Fq['pf_gap']=Fq.t-g.t.shift(1); Fq['pf_same_stand']=(Fq.st==g.st.shift(1)).astype('float32'); Fq['pf_daobt']=g.daobt.shift(1)
Fq=Fq.sort_values('_i')
F['pf_gap']=Fq.pf_gap.values; F['pf_same_stand']=Fq.pf_same_stand.values; F['pf_daobt']=Fq.pf_daobt.values
F.drop(columns=['_i'],inplace=True)
F['t_mvt']=t; F['t_sch']=ts_
F['day']=((t+0)//86400).astype('int64')
print(F.shape, round(time.time()-t0,1))
F.to_parquet(OUT+f'feat1_{SEL}.parquet',index=False)
print(F.describe().T[['count','mean','50%']].round(2).to_string())
