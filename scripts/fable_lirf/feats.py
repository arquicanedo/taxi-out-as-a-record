import os
import pandas as pd, numpy as np
from base import *
def build():
    d=pd.read_parquet('lirf_dep_feat.parquet')
    df=pd.read_parquet('lirf.parquet')
    a=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')].copy()
    a['acopy']=(a.BLOCK_TIME_UTC_mvt-a.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
    a['car']=a.FLIGHT_mvt.str[:3]; a['day']=a.MVT_TIME_UTC_mvt.dt.floor('D'); a['mon']=a.MVT_TIME_UTC_mvt.dt.month; a['yr']=a.MVT_TIME_UTC_mvt.dt.year
    cd=a.groupby(['car','day']).acopy.agg(['mean','size']).rename(columns={'mean':'car_day_arr','size':'car_day_arr_n'}).reset_index()
    d=d.merge(cd,on=['car','day'],how='left')
    # carrier-month arrival copy rate (same month, observable in 2026 since arrivals labelled)
    d['yr']=d.MVT_TIME_UTC_mvt.dt.year
    cm=a.groupby(['car','yr','mon']).acopy.agg(['mean','size']).rename(columns={'mean':'car_mon_arr','size':'car_mon_arr_n'}).reset_index()
    d=d.merge(cm,on=['car','yr','mon'],how='left')
    # carrier-day join-null delayed count, carrier-day delayed count
    g=d.groupby(['car','day']).agg(car_day_cf=('cfc','sum'),car_day_del=('del','sum')).reset_index()
    d=d.merge(g,on=['car','day'],how='left')
    d['dest2']=d.ADES_mvt.str[:2]; d['stand1']=d.STAND_mvt.str[:1]
    d['pat_bad']=(d.pat!='icao').astype(int)
    d['prev_dep_jn']=0
    for c in ['dest2','stand1','RUNWAY_mvt','car']:
        d[c+'_cat']=d[c].astype('category').cat.codes
    return d
def carrier_rates(train, te, keys=('car',), M=5):
    """cross-fitted carrier rates from train rows: copy rate overall (delayed), coin-flip copy rate, jn|copy, jn|real"""
    out={}
    dl=train[train['del']]
    g=dl.groupby(list(keys))['copy'].agg(['sum','size']); gp=dl['copy'].mean()
    cf=train[train.cfc].groupby(list(keys))['copy'].agg(['sum','size']); gc=train[train.cfc]['copy'].mean()
    q=dl[dl['copy']].groupby(list(keys))['jn'].agg(['sum','size']); gq=dl[dl['copy']].jn.mean()
    r=dl[~dl['copy']].groupby(list(keys))['jn'].agg(['sum','size']); gr=dl[~dl['copy']].jn.mean()
    allc=train.groupby(list(keys))['copy'].agg(['sum','size']); ga=train['copy'].mean()
    def sh(tab,g,k):
        if k in tab.index: s,n=tab.loc[k]; return (s+g*M)/(n+M)
        return g
    ks=te[list(keys)].itertuples(index=False,name=None) if len(keys)>1 else te[keys[0]]
    ks=list(ks)
    P=np.array([sh(g,gp,k) for k in ks]); Q=np.array([sh(q,gq,k) for k in ks]); R=np.array([sh(r,gr,k) for k in ks])
    out['car_del_copy']=P; out['car_cf_copy']=np.array([sh(cf,gc,k) for k in ks]); out['car_q']=Q; out['car_r']=R
    out['car_bayes']=P*Q/(P*Q+(1-P)*R); out['car_all_copy']=np.array([sh(allc,ga,k) for k in ks])
    return pd.DataFrame(out,index=te.index)
