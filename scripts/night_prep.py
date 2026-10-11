"""Build the validation frames once. Every variant tonight loads these."""
import pandas as pd, numpy as np, glob, os, pyarrow.parquet as pq, warnings; warnings.filterwarnings('ignore')
SP=os.environ.get("DC26_SP", os.environ["DC26_WORK"])
# produced by scripts/newcols.py -> scripts/derived_ops.py (DC26_TMP, default /tmp)
d=pd.read_parquet(os.path.join(os.environ.get('DC26_TMP','/tmp'),'dc26_derived.parquet'))
d['aobt_null']=d.AOBT_3_flt.isna().astype(float)
d['sched_delay']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['sched_delay_log']=np.sign(d.sched_delay)*np.log1p(np.abs(d.sched_delay))
d['mvt_minus_lobt']=(d.MVT_TIME_UTC_mvt-d.LOBT_flt).dt.total_seconds()
d['mvt_minus_iobt']=(d.MVT_TIME_UTC_mvt-d.IOBT_flt).dt.total_seconds()
d['mvt_minus_eobt']=(d.MVT_TIME_UTC_mvt-d.EOBT_1_flt).dt.total_seconds()
d=d.dropna(subset=['TAXITIME_SEC_mvt']); d['y']=d.TAXITIME_SEC_mvt.astype(float); d=d[d.y>0]
d['month']=d.MVT_TIME_UTC_mvt.dt.month
# copy-event labels (the quantities the equation actually needs)
d['copy_s']=((d.y-d.sched_delay).abs()<=60).astype(int)
d['copy_l']=((d.y-d.mvt_minus_lobt).abs()<=60).astype(int)
d['mat_s']=(d.copy_s.astype(bool)&(d.sched_delay>=1800)).astype(int)   # materially different from a normal taxi
d['mat_l']=(d.copy_l.astype(bool)&(d.mvt_minus_lobt>=1800)).astype(int)
# The validation window MIRRORS the ranking set's composition, so the split rule
# is a property of the ranking extract, not a modelling choice.
#   j3  : Jan all-10 + Jul {EDDF,EGLL,EHAM}. Matches the ORIGINAL 215,876-row
#         ranking set, where July carried only three reporting airports. Every
#         number recorded in day-one-findings up to s89 was measured on this.
#   all : Jan all-10 + Jul all-10. Matches the 2026-09-04 re-extract's 344,841
#         rows, where July carries all ten. Use this going forward.
J3=['EDDF','EGLL','EHAM']
_SPLIT=os.environ.get('DC26_SPLIT','j3')
assert _SPLIT in ('j3','all'), f'DC26_SPLIT={_SPLIT!r}'
V=(d.month==1)|((d.month==7)&(d.apt.isin(J3) if _SPLIT=='j3' else True))
print(f'split={_SPLIT}')
TAGSP='' if _SPLIT=='j3' else '_'+_SPLIT
tr,va=d[~V].copy(),d[V].copy()
def te(k,n,q=50):
    gm=tr.y.mean(); g=tr.groupby(k,observed=True).y.agg(['mean','size']); sm=(g['mean']*g['size']+gm*q)/(g['size']+q)
    for f in (tr,va): f[n]=f.set_index(k).index.map(sm).astype(float)
te(['apt','STAND_mvt','RUNWAY_mvt'],'te_stand_rwy'); te(['apt','STAND_mvt'],'te_stand'); te(['apt','RUNWAY_mvt'],'te_rwy')
te(['apt','AIRCRAFT_OPERATOR_flt'],'te_op'); te(['apt'],'te_apt'); te(['ADES_mvt'],'te_des',100)
te(['apt','ADES_mvt'],'te_apt_des',100); te(['apt','aobt_null'],'te_apt_null',20)
CATS=['apt','RUNWAY_mvt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt','FLIGHT_TYPE_flt','AIRCRAFT_TYPE_mvt','dest_region']
for c in CATS:
    kp=tr[c].value_counts().index[:250]
    for f in (tr,va): f[c]=f[c].where(f[c].isin(kp),'OTHER')
    u=pd.api.types.CategoricalDtype(sorted(set(tr[c].dropna())|set(va[c].dropna())))
    tr[c]=tr[c].astype(u); va[c]=va[c].astype(u)
F=['naive','aobt_minus_eobt','aobt_minus_sched','lobt_minus_iobt','filed_block','flown_block','hour','dow','month','doy',
 'hr_sin','hr_cos','N_push','N_off','dep_900','dep_1800','dep_3600','arr_900','arr_1800','rwy_900','rwy_gap',
 'apt_day_naive','apt_day_naive_p90','apt_hr_naive','naive_vs_day','te_stand_rwy','te_stand','te_rwy','te_op','te_apt',
 'te_des','te_apt_des','aobt_null','te_apt_null','sched_delay','sched_delay_log','mvt_minus_lobt','mvt_minus_iobt','mvt_minus_eobt']+CATS
tr.to_parquet(f'{SP}/night_tr{TAGSP}.parquet',index=False); va.to_parquet(f'{SP}/night_va{TAGSP}.parquet',index=False)
import json; json.dump({'F':F,'CATS':CATS},open(f'{SP}/night_cols{TAGSP}.json','w'))
print(f'train {len(tr):,}  val {len(va):,}  feats {len(F)}')
print(f'  val extremes(y>=2h) {int((va.y>=7200).sum())}')
print(f'  train copy_s {int(tr.copy_s.sum()):,}  copy_l {int(tr.copy_l.sum()):,}  mat_s {int(tr.mat_s.sum()):,}  mat_l {int(tr.mat_l.sum()):,}')
print(f'  val   copy_s {int(va.copy_s.sum()):,}  copy_l {int(va.copy_l.sum()):,}  mat_s {int(va.mat_s.sum()):,}  mat_l {int(va.mat_l.sum()):,}')
