"""DC2026 submission generator v6: LightGBM bulk model.

Fixes carried over the v3 recipe:
  * NO y<7200 training filter   (v1-v3 trained without any extreme examples)
  * NO prediction clip at 7200  (v1-v3 could not emit a value above 2h at all)
  * fixed timestamp units and surface counts (via features.py)
  * two-stage mixture for the tail
  * destination encodings, explicit aobt_null
v6 change over v5: the bulk model is LightGBM, not sklearn HistGradientBoosting.
Measured on the Jan+July3 split with the extreme classifier frozen and shared, so
the difference is attributable to the bulk learner alone:
  overall  347.15 -> 345.04      EGLL  293.66 -> 283.50 (10.2 s)
  fit time    115 s ->     79 s
Nothing else changed. No candidate column from the s37 audit earned inclusion;
the best, a runway-configuration switch detector, was -0.23 s, inside the noise
floor. `brew install libomp` (2026-09-02) is what made LightGBM loadable at all.

v5 changes over v4, all measured on held-out months:
  * E[y|extreme] = sched_delay, not a group mean.  LIRF copies SCHED_TIME into
    BLOCK_TIME when it has no actual off-block, so y == sched_delay exactly on
    82% of all extremes.  Worth +24.6 s seed-averaged on the Jan+July3 split,
    and +42.7 s projected on the ranking set's larger 2026 cell.
  * normal model retargeted on the residual y - baseline (naive, or apt_hr_naive
    where AOBT_3 is null), which a GBM cannot express in piecewise constants.
  * 3-seed averaging on both the regressor and the classifier.  Single-seed
    comparisons here carry about +-9 s of noise.
  * soft mixture, not hard assignment.  Hard beat soft on one seed and lost
    under three, which is what squared loss predicts for a calibrated P.
"""
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, glob, time, sys
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import features
D=features.DATA; U=features.USE+['BLOCK_TIME_UTC_mvt']; t0=time.time()
EXT_T=7200          # tail threshold; NOT a filter — only defines the mixture component
# CLIP_HI removed (s64): it was the s50 bug, dead since v9 but still readable
# here as if it were policy. The prediction guard lives in v14_combine.py.

def link(raw, dep):
    raw=raw.copy(); raw['apt']=raw.ADEP_mvt.where(raw.PHASE_mvt=='DEP',raw.ADES_mvt)
    a=raw[(raw.PHASE_mvt=='ARR')&raw.BLOCK_TIME_UTC_mvt.notna()&raw.STAND_mvt.notna()][
        ['apt','STAND_mvt','BLOCK_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','TAXITIME_SEC_mvt']].copy()
    a.columns=['apt','STAND_mvt','in_blk','a_type','a_taxiin']
    a['k']=a.apt+'|'+a.STAND_mvt.astype(str); dep=dep.copy(); dep['k']=dep.apt+'|'+dep.STAND_mvt.astype(str)
    m=pd.merge_asof(dep.sort_values('MVT_TIME_UTC_mvt'),a.sort_values('in_blk').drop(columns=['apt','STAND_mvt']),
                    left_on='MVT_TIME_UTC_mvt',right_on='in_blk',by='k',direction='backward',allow_exact_matches=False)
    m['since_inblk']=(m.MVT_TIME_UTC_mvt-m.in_blk).dt.total_seconds()
    m.loc[(m.since_inblk<0)|(m.since_inblk>86400),'since_inblk']=np.nan
    m['type_match']=(m.a_type==m.AIRCRAFT_TYPE_mvt).astype(float)
    m['prev_taxi_in']=m.a_taxiin.astype(float)
    m['aobt_null']=m.AOBT_3_flt.isna().astype(float)
    # record-structure signals: always available, never blanked
    m['sched_delay']=(m.MVT_TIME_UTC_mvt-m.SCHED_TIME_UTC_mvt).dt.total_seconds()
    m['sched_delay_log']=np.sign(m.sched_delay)*np.log1p(np.abs(m.sched_delay))
    m['mvt_minus_lobt']=(m.MVT_TIME_UTC_mvt-m.LOBT_flt).dt.total_seconds()
    m['mvt_minus_iobt']=(m.MVT_TIME_UTC_mvt-m.IOBT_flt).dt.total_seconds()
    m['mvt_minus_eobt']=(m.MVT_TIME_UTC_mvt-m.EOBT_1_flt).dt.total_seconds()
    return m

tr_raw=pd.concat([pq.read_table(f,columns=U).to_pandas() for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True).drop_duplicates('MVT_ID_mvt')
rk_raw=pq.read_table(f'{D}/ranking.parquet',columns=U).to_pandas()
tr=link(tr_raw,features.build(tr_raw)); rk=link(rk_raw,features.build(rk_raw))
tr=tr.dropna(subset=['TAXITIME_SEC_mvt']); tr['y']=tr.TAXITIME_SEC_mvt.astype(float)
tr=tr[tr.y>0]                                   # lower bound only. NO upper filter.
print(f'built. train {len(tr):,} (extremes kept: {(tr.y>=EXT_T).sum()})  rank {len(rk):,}  [{time.time()-t0:.0f}s]')

def te(k,n,q=50):
    gm=tr.y.mean(); g=tr.groupby(k,observed=True).y.agg(['mean','size']); sm=(g['mean']*g['size']+gm*q)/(g['size']+q)
    for df in (tr,rk): df[n]=df.set_index(k).index.map(sm).astype(float)
te(['apt','STAND_mvt','RUNWAY_mvt'],'te_stand_rwy'); te(['apt','STAND_mvt'],'te_stand'); te(['apt','RUNWAY_mvt'],'te_rwy')
te(['apt','AIRCRAFT_OPERATOR_flt'],'te_op'); te(['apt'],'te_apt'); te(['ADES_mvt'],'te_des',100)
te(['apt','ADES_mvt'],'te_apt_des',100); te(['apt','aobt_null'],'te_apt_null',20)
CATS=['apt','RUNWAY_mvt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt','FLIGHT_TYPE_flt','AIRCRAFT_TYPE_mvt','dest_region']
for c in CATS:
    kp=tr[c].value_counts().index[:250]
    for df in (tr,rk): df[c]=df[c].where(df[c].isin(kp),'OTHER')
    u=pd.api.types.CategoricalDtype(sorted(set(tr[c].dropna())|set(rk[c].dropna()))); tr[c]=tr[c].astype(u); rk[c]=rk[c].astype(u)
F=['naive','aobt_minus_eobt','aobt_minus_sched','lobt_minus_iobt','filed_block','flown_block','hour','dow','month','doy',
   'hr_sin','hr_cos','N_push','N_off','dep_900','dep_1800','dep_3600','arr_900','arr_1800','rwy_900','rwy_gap',
   'apt_day_naive','apt_day_naive_p90','apt_hr_naive','naive_vs_day','te_stand_rwy','te_stand','te_rwy','te_op','te_apt',
   'since_inblk','type_match','prev_taxi_in','te_des','te_apt_des','aobt_null','te_apt_null',
   'sched_delay','sched_delay_log','mvt_minus_lobt','mvt_minus_iobt','mvt_minus_eobt']+CATS


# ---- in-period arrival norms (s47). Arrivals are labelled in 2025 AND 2026, so
# keying by the row's own year-month makes these self-updating rather than stale.
import pyarrow.parquet as _pq
_U=['PHASE_mvt','ADES_mvt','MVT_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt','TAXITIME_SEC_mvt','MVT_ID_mvt']
_a=pd.concat([tr_raw[_U] if set(_U)<=set(tr_raw.columns) else tr_raw.reindex(columns=_U),
              rk_raw[_U] if set(_U)<=set(rk_raw.columns) else rk_raw.reindex(columns=_U)],ignore_index=True)
_a=_a[(_a.PHASE_mvt=='ARR')&_a.TAXITIME_SEC_mvt.notna()&(_a.TAXITIME_SEC_mvt>0)&(_a.TAXITIME_SEC_mvt<7200)].copy()
_a['apt']=_a.ADES_mvt.astype(str)
_a['ym']=_a.MVT_TIME_UTC_mvt.dt.year*100+_a.MVT_TIME_UTC_mvt.dt.month
def _norm(keys,minn,name):
    g=_a.groupby(keys,observed=True).TAXITIME_SEC_mvt.agg(['median','size']).reset_index()
    return g[g['size']>=minn].rename(columns={'median':name})[keys+[name]]
_ns=_norm(['apt','STAND_mvt','ym'],10,'arr_stand_mo')
_nr=_norm(['apt','RUNWAY_mvt','ym'],30,'arr_rwy_mo')
_na=_norm(['apt','ym'],50,'arr_apt_mo')
for _df in (tr,rk):
    _df['ym']=_df.MVT_TIME_UTC_mvt.dt.year*100+_df.MVT_TIME_UTC_mvt.dt.month
tr=tr.merge(_ns,on=['apt','STAND_mvt','ym'],how='left').merge(_nr,on=['apt','RUNWAY_mvt','ym'],how='left').merge(_na,on=['apt','ym'],how='left')
rk=rk.merge(_ns,on=['apt','STAND_mvt','ym'],how='left').merge(_nr,on=['apt','RUNWAY_mvt','ym'],how='left').merge(_na,on=['apt','ym'],how='left')
INP=['arr_stand_mo','arr_rwy_mo','arr_apt_mo']
print('in-period arrival norms — train nulls: '+str({c:f'{100*tr[c].isna().mean():.1f}%' for c in INP})
      +'  rank nulls: '+str({c:f'{100*rk[c].isna().mean():.1f}%' for c in INP}),flush=True)
print('  ranking ym values: '+str(sorted(rk.ym.unique()))+'   arrival pool ym: '+str(sorted(_a.ym.unique())[:3])+'...'+str(sorted(_a.ym.unique())[-2:]),flush=True)
F_BULK=F+INP
tr.to_parquet(features.wk('v8s_tr'),index=False); rk.to_parquet(features.wk('v8s_rk'),index=False)
import json; json.dump({'F':F,'F_BULK':F_BULK,'CATS':CATS},open(features.wk('v8s_cols','json'),'w'))
print(f'saved: train {len(tr):,}  rank {len(rk):,}  feats {len(F)}',flush=True)
