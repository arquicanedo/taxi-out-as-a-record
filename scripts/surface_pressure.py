"""Bottleneck intensity: not how many aircraft are present, but how many are
CURRENTLY STRUGGLING -- observable at inference through `naive`, crossed with
wake category. Rush-hour backup, per airport, in a rolling window."""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq, glob
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import features
D=features.DATA; U=features.USE+['BLOCK_TIME_UTC_mvt']
raw=pd.concat([pq.read_table(f,columns=U).to_pandas() for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True).drop_duplicates('MVT_ID_mvt')
raw['apt']=raw.ADEP_mvt.where(raw.PHASE_mvt=='DEP',raw.ADES_mvt)
d=pd.read_parquet(os.path.join(os.environ.get('DC26_TMP', '/tmp'), 'dc26_feat_fixed.parquet'))
a=raw[(raw.PHASE_mvt=='ARR')&raw.BLOCK_TIME_UTC_mvt.notna()&raw.STAND_mvt.notna()][['apt','STAND_mvt','BLOCK_TIME_UTC_mvt','AIRCRAFT_TYPE_mvt','TAXITIME_SEC_mvt']].copy()
a.columns=['apt','STAND_mvt','in_blk','a_type','a_taxiin']
a['k']=a.apt+'|'+a.STAND_mvt.astype(str); d['k']=d.apt+'|'+d.STAND_mvt.astype(str)
d=pd.merge_asof(d.sort_values('MVT_TIME_UTC_mvt'),a.sort_values('in_blk').drop(columns=['apt','STAND_mvt']),
    left_on='MVT_TIME_UTC_mvt',right_on='in_blk',by='k',direction='backward',allow_exact_matches=False)
d['since_inblk']=(d.MVT_TIME_UTC_mvt-d.in_blk).dt.total_seconds()
d.loc[(d.since_inblk<0)|(d.since_inblk>86400),'since_inblk']=np.nan
d['type_match']=(d.a_type==d.AIRCRAFT_TYPE_mvt).astype(float); d['prev_taxi_in']=d.a_taxiin.astype(float)
d['aobt_null']=d.AOBT_3_flt.isna().astype(float)
d['is_H']=(d.WK_TBL_CAT_flt=='H').astype(float)

# ---- bottleneck intensity, rolling on take-off time, per airport ----
d['toff']=d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('float64')
LONG=1800.0                      # "struggling" threshold on the OBSERVABLE naive
d['nlong']=(d.naive.fillna(0)>=LONG).astype(float)
d['nH_long']=d.nlong*d.is_H
out=[]
for apt,g in d.groupby('apt',sort=False):
    g=g.sort_values('toff').copy(); t=g.toff.values
    for col,tag in [('nlong','long'),('nH_long','Hlong'),('is_H','H')]:
        cs=np.concatenate([[0],np.cumsum(g[col].values)])
        for w,wt in ((1800,'30m'),(5400,'90m')):
            hi=np.searchsorted(t,t,'right'); lo=np.searchsorted(t,t-w,'left')
            n=np.maximum(hi-lo,1)
            g[f'{tag}_n_{wt}']=cs[hi]-cs[lo]
            g[f'{tag}_share_{wt}']=(cs[hi]-cs[lo])/n
    # my own naive relative to the local congestion level
    cs=np.concatenate([[0],np.cumsum(g.naive.fillna(g.naive.median()).values)])
    hi=np.searchsorted(t,t,'right'); lo=np.searchsorted(t,t-1800,'left'); n=np.maximum(hi-lo,1)
    g['local_naive_mean']=(cs[hi]-cs[lo])/n
    g['naive_vs_local']=g.naive-g.local_naive_mean
    out.append(g)
d=pd.concat(out,ignore_index=True)
B=[c for c in d.columns if c.startswith(('long_','Hlong_','H_n','H_share'))]+['local_naive_mean','naive_vs_local','is_H']
d2=d.dropna(subset=['TAXITIME_SEC_mvt']); yy=d2.TAXITIME_SEC_mvt.astype(float)
print('bottleneck features, correlation with taxi time:')
for c in B: print(f'   {c:<20} {yy.corr(d2[c]):+.3f}')
d=d.dropna(subset=['TAXITIME_SEC_mvt']); d['y']=d.TAXITIME_SEC_mvt.astype(float); d=d[d.y>0]
J3=['EDDF','EGLL','EHAM']; VAL=(d.month==1)|((d.month==7)&d.apt.isin(J3))
tr,va=d[~VAL].copy(),d[VAL].copy()
def te(k,n,q=50):
    gm=tr.y.mean(); g=tr.groupby(k,observed=True).y.agg(['mean','size']); sm=(g['mean']*g['size']+gm*q)/(g['size']+q)
    for df in (tr,va): df[n]=df.set_index(k).index.map(sm).astype(float)
te(['apt','STAND_mvt','RUNWAY_mvt'],'te_stand_rwy'); te(['apt','STAND_mvt'],'te_stand'); te(['apt','RUNWAY_mvt'],'te_rwy')
te(['apt','AIRCRAFT_OPERATOR_flt'],'te_op'); te(['apt'],'te_apt'); te(['ADES_mvt'],'te_des',100)
te(['apt','ADES_mvt'],'te_apt_des',100); te(['apt','aobt_null'],'te_apt_null',20)
CATS=['apt','RUNWAY_mvt','MARKET_SEGMENT_flt','WK_TBL_CAT_flt','FLIGHT_TYPE_flt','AIRCRAFT_TYPE_mvt','dest_region']
for c in CATS:
    kp=tr[c].value_counts().index[:250]
    for df in (tr,va): df[c]=df[c].where(df[c].isin(kp),'OTHER')
    u=pd.api.types.CategoricalDtype(sorted(set(tr[c].dropna())|set(va[c].dropna()))); tr[c]=tr[c].astype(u); va[c]=va[c].astype(u)
BASE=['naive','aobt_minus_eobt','aobt_minus_sched','lobt_minus_iobt','filed_block','flown_block','hour','dow','month','doy',
   'hr_sin','hr_cos','N_push','N_off','dep_900','dep_1800','dep_3600','arr_900','arr_1800','rwy_900','rwy_gap',
   'apt_day_naive','apt_day_naive_p90','apt_hr_naive','naive_vs_day','te_stand_rwy','te_stand','te_rwy','te_op','te_apt',
   'since_inblk','type_match','prev_taxi_in','te_des','te_apt_des','aobt_null','te_apt_null']+CATS
rmse=lambda p,q: float(np.sqrt(np.mean((np.asarray(p)-np.asarray(q))**2))); yv=va.y.values
R=lambda s=0: HistGradientBoostingRegressor(learning_rate=0.06,max_iter=600,max_leaf_nodes=127,min_samples_leaf=100,
   l2_regularization=1.0,categorical_features='from_dtype',early_stopping=True,validation_fraction=0.1,n_iter_no_change=40,random_state=s)
Cf=lambda s=0: HistGradientBoostingClassifier(learning_rate=0.06,max_iter=400,max_leaf_nodes=63,min_samples_leaf=50,
   l2_regularization=1.0,categorical_features='from_dtype',early_stopping=True,validation_fraction=0.1,n_iter_no_change=40,random_state=s)
def mixture(F):
    ext=(tr.y>=7200).values
    clf=Cf().fit(tr[F],ext.astype(int)); Pv=clf.predict_proba(va[F])[:,1]; Pt=clf.predict_proba(tr[F])[:,1]
    Pv=IsotonicRegression(out_of_bounds='clip').fit(Pt,ext.astype(float)).predict(Pv)
    mn=R().fit(tr[~ext][F],tr[~ext].y).predict(va[F])
    gm=tr[ext].y.mean(); g=tr[ext].groupby(['apt','aobt_null'],observed=True).y.agg(['mean','size'])
    g=g[g['size']>=5]['mean']; me=pd.Series(va.set_index(['apt','aobt_null']).index.map(g)).astype(float).fillna(gm).values
    p=Pv*me+(1-Pv)*mn
    lng=(yv>=1800)
    return rmse(p,yv), rmse(p[lng],yv[lng])
a1,l1=mixture(BASE); a2,l2=mixture(BASE+B)
print()
print(f'  {"MIXTURE, current":<40}{a1:8.1f}   [>=30m rows {l1:7.0f}]')
print(f'  {"MIXTURE + bottleneck intensity":<40}{a2:8.1f}   [>=30m rows {l2:7.0f}]')
print(f'\n  gain {a1-a2:+.1f} s ({100*(1-a2/a1):+.2f}%)')
