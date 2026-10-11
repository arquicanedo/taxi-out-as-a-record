import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep()
d['del']=d.ms>=3600; d['cfc']=d.jn&d['del']
day=d.groupby('day').agg(n=('ms','size'),n_del=('del','sum'),n_cf=('cfc','sum')).reset_index()
day['frac_del']=day.n_del/day.n
d=d.merge(day,on='day')
# carrier-day DLA discipline from join-present delayed rows
jp=d[~d.jn&(d.ms>=1800)].copy(); jp['dla']=jp.ls>=900
cd=jp.groupby(['car','day']).dla.agg(['mean','size']).rename(columns={'mean':'car_day_dla','size':'car_day_dla_n'}).reset_index()
d=d.merge(cd,on=['car','day'],how='left')
t=d[d.set=='train']; cf=t[t.cfc].copy()
print('coin-flip rows', len(cf))
cf['b_ncf']=pd.cut(cf.n_cf,[0,1,3,9,30,100]); cf['b_frac']=pd.cut(cf.frac_del,[-.01,.02,.05,.1,.2,1])
print(cf.groupby('b_ncf',observed=True)['copy'].agg(['mean','size']).to_string())
print(cf.groupby('b_frac',observed=True)['copy'].agg(['mean','size']).to_string())
cf['b_dla']=pd.cut(cf.car_day_dla,[-.01,.001,.5,.999,1.01])
print(cf.groupby('b_dla',observed=True)['copy'].agg(['mean','size']).to_string(), '\n missing', cf.car_day_dla.isna().sum())
print(cf.groupby(['b_frac','b_dla'],observed=True)['copy'].agg(['mean','size']).unstack('b_dla').to_string())
# 2026 same features
r=d[d.set=='rank']; rc=r[r.cfc]
print('2026 coin-flip rows', len(rc)); print(rc.assign(b_frac=pd.cut(rc.frac_del,[-.01,.02,.05,.1,.2,1]),b_dla=pd.cut(rc.car_day_dla,[-.01,.001,.5,.999,1.01])).groupby(['mon','b_frac','b_dla'],observed=True).size().to_string())
print('frac_del by yr-mon:', d.groupby([d.MVT_TIME_UTC_mvt.dt.year,'mon']).frac_del.mean().round(3).to_dict())
d.to_parquet('lirf_dep_feat.parquet',index=False)
