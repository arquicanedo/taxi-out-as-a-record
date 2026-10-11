"""Track E re-test arms. Each arm returns (tr, FB, extra) for p1e_fit.py.

Every added feature is formable from ranking.parquet columns: take-off time, runway, airport,
AOBT_3 (through `naive`), wake category, operator, stand, flight designator. No label, no BLOCK,
no neighbour label. `naive` is the shipped column (MVT - AOBT_3, NaN outside [0, 7200]).

  s28    the s28 bottleneck-intensity set, re-implemented from scripts/surface_pressure.py
         (per airport, backward 30/90-min windows on take-off time, inclusive of the row itself)
  rwyq   the observable proxy s257 never tried for the runway-queue axis: the NEIGHBOURS' naive
         on the same (airport, runway), previous and next by take-off time, and its deviation
         from the airport-hour median, plus centred +-15 min means excluding the row itself
  s28rwyq  both
  idcat  s421's three identity columns as native categoricals (op x apt, stand x apt, carrier x apt)
  tuned  s417's 09-11 tuned hyperparameter set (tune_lgb_best.json) on the pooled model. RESEARCH ONLY: no release
         script produces that json, and v100 uses idcat and rwyq only
"""
import os, glob, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
D=os.environ['DC26_DATA']; W=os.environ['DC26_WORK']

def _secs(s):
    t=pd.to_datetime(s,utc=True)
    v=(t-pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds().values.astype(np.float64)
    return v

def s28_feats(tr):
    t=_secs(tr.MVT_TIME_UTC_mvt); assert 1.735e9<np.nanmedian(t)<1.80e9,'units'
    nv=tr.naive.values.astype(float)
    isH=(tr.WK_TBL_CAT_flt.astype(str).values=='H').astype(float)
    nlong=(np.nan_to_num(nv,nan=0.)>=1800.).astype(float); nHl=nlong*isH
    apt=tr.apt.astype(str).values
    out={k:np.full(len(tr),np.nan) for k in
         [f'{tag}_{kind}_{wt}' for tag in ('long','Hlong','H') for kind in ('n','share') for wt in ('30m','90m')]
         +['local_naive_mean','naive_vs_local']}
    for a in np.unique(apt):
        ix=np.where(apt==a)[0]; o=ix[np.argsort(t[ix],kind='stable')]; tt=t[o]
        for col,tag in ((nlong,'long'),(nHl,'Hlong'),(isH,'H')):
            cs=np.concatenate([[0],np.cumsum(col[o])])
            for w,wt in ((1800,'30m'),(5400,'90m')):
                hi=np.searchsorted(tt,tt,'right'); lo=np.searchsorted(tt,tt-w,'left'); n=np.maximum(hi-lo,1)
                out[f'{tag}_n_{wt}'][o]=cs[hi]-cs[lo]; out[f'{tag}_share_{wt}'][o]=(cs[hi]-cs[lo])/n
        v=nv[o]; med=np.nanmedian(v) if np.isfinite(v).any() else 900.
        cs=np.concatenate([[0],np.cumsum(np.where(np.isfinite(v),v,med))])
        hi=np.searchsorted(tt,tt,'right'); lo=np.searchsorted(tt,tt-1800,'left'); n=np.maximum(hi-lo,1)
        out['local_naive_mean'][o]=(cs[hi]-cs[lo])/n
    out['naive_vs_local']=nv-out['local_naive_mean']
    out['is_H']=isH
    return pd.DataFrame(out,index=tr.index)

def rwyq_feats(tr):
    t=_secs(tr.MVT_TIME_UTC_mvt); assert 1.735e9<np.nanmedian(t)<1.80e9,'units'
    nv=tr.naive.values.astype(float); ah=tr.apt_hr_naive.values.astype(float)
    dev=nv-ah
    key=(tr.apt.astype(str)+'|'+tr.RUNWAY_mvt.astype(str)).values
    apt=tr.apt.astype(str).values
    F={k:np.full(len(tr),np.nan) for k in ('rq_p1','rq_p2','rq_p3','rq_n1','rq_n2','rq_dev_p3','rq_dev_n2',
                                           'rq_dev_c15','rq_cnt_c15','rq_dt_p1','rq_dt_n1','aq_dev_c15')}
    def centred(o,tt,x,w,self_excl=True):
        ok=np.isfinite(x[o]).astype(float); xv=np.where(np.isfinite(x[o]),x[o],0.)
        cs=np.concatenate([[0],np.cumsum(xv)]); cn=np.concatenate([[0],np.cumsum(ok)])
        hi=np.searchsorted(tt,tt+w,'right'); lo=np.searchsorted(tt,tt-w,'left')
        s=cs[hi]-cs[lo]; n=cn[hi]-cn[lo]
        if self_excl: s=s-xv; n=n-ok
        m=np.where(n>0,s/np.maximum(n,1),np.nan)
        return m,n
    for k in np.unique(key):
        ix=np.where(key==k)[0]; o=ix[np.argsort(t[ix],kind='stable')]; tt=t[o]; v=nv[o]; dv=dev[o]; m=len(o)
        for lag,nm in ((1,'rq_p1'),(2,'rq_p2'),(3,'rq_p3')):
            a=np.full(m,np.nan); a[lag:]=v[:-lag] if m>lag else a[lag:]; F[nm][o]=a
        for lag,nm in ((1,'rq_n1'),(2,'rq_n2')):
            a=np.full(m,np.nan); a[:-lag]=v[lag:] if m>lag else a[:-lag]; F[nm][o]=a
        a=np.full(m,np.nan)
        if m>1: a[1:]=tt[1:]-tt[:-1]
        F['rq_dt_p1'][o]=a
        a=np.full(m,np.nan)
        if m>1: a[:-1]=tt[1:]-tt[:-1]
        F['rq_dt_n1'][o]=a
        # mean deviation of the previous 3 and next 2 (NaN-aware)
        P=np.full((m,3),np.nan)
        for lag in (1,2,3):
            if m>lag: P[lag:,lag-1]=dv[:-lag]
        F['rq_dev_p3'][o]=np.nanmean(np.where(np.isfinite(P),P,np.nan),1) if m else np.nan
        Nn=np.full((m,2),np.nan)
        for lag in (1,2):
            if m>lag: Nn[:-lag,lag-1]=dv[lag:]
        F['rq_dev_n2'][o]=np.nanmean(Nn,1)
        mm,nn=centred(o,tt,dev,900.)
        F['rq_dev_c15'][o]=mm; F['rq_cnt_c15'][o]=nn
    for a_ in np.unique(apt):
        ix=np.where(apt==a_)[0]; o=ix[np.argsort(t[ix],kind='stable')]; tt=t[o]
        mm,_=centred(o,tt,dev,900.)
        F['aq_dev_c15'][o]=mm
    return pd.DataFrame(F,index=tr.index)

def idcat_feats(tr):
    C=['MVT_ID_mvt','PHASE_mvt','FLIGHT_mvt','AIRCRAFT_OPERATOR_flt','STAND_mvt','ADEP_mvt']
    meta=pd.concat([pq.read_table(f,columns=C).to_pandas() for f in sorted(glob.glob(f'{D}/training_*.parquet'))],
                   ignore_index=True); meta=meta[meta.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt')
    M=meta.set_index('MVT_ID_mvt').reindex(tr.MVT_ID_mvt.values)
    assert M.ADEP_mvt.notna().mean()>0.999, f'join coverage {M.ADEP_mvt.notna().mean():.4f}'
    S=lambda c: M[c].astype(str).values
    out=pd.DataFrame({'op_apt':S('ADEP_mvt').astype(object)+'|'+S('AIRCRAFT_OPERATOR_flt'),
                      'stand_apt':S('ADEP_mvt').astype(object)+'|'+S('STAND_mvt'),
                      'car_apt':S('ADEP_mvt').astype(object)+'|'+pd.Series(S('FLIGHT_mvt')).str[:3].values},
                     index=tr.index)
    return out

def apply(arm,tr,FB,TRN,WIN,y):
    FB=list(FB); extra={}; n0=len(FB)
    if arm in ('s28','s28rwyq'):
        f=s28_feats(tr)
        for c in f.columns:
            if c not in tr.columns: tr[c]=f[c].values; FB.append(c)
    if arm in ('rwyq','s28rwyq'):
        f=rwyq_feats(tr)
        for c in f.columns: tr[c]=f[c].values; FB.append(c)
    if arm=='idcat':
        f=idcat_feats(tr)
        for c in f.columns: tr[c]=f[c].values; FB.append(c)
        extra['cats']=list(f.columns)
    if arm=='tuned':
        p=json.load(open(f'{W}/tune_lgb_best.json'))['params']
        extra['params']={k:v for k,v in p.items() if k!='objective'}
    for c in FB[n0:]:
        if c in tr.columns and tr[c].dtype.kind=='f':
            v=tr[c].values
            print(f'    feat {c:<18} coverage {np.isfinite(v).mean():6.1%}  median {np.nanmedian(v) if np.isfinite(v).any() else float("nan"):>10.2f}',flush=True)
    return tr,FB,extra
