import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
a=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')].copy()
a['copy']=(a.BLOCK_TIME_UTC_mvt-a.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
a['car']=a.FLIGHT_mvt.str[:3]; a['mon']=a.MVT_TIME_UTC_mvt.dt.month; a['yr']=a.MVT_TIME_UTC_mvt.dt.year
a['sm']=(a.MVT_TIME_UTC_mvt-a.SCHED_TIME_UTC_mvt).dt.total_seconds()
print('arrival copy rate by year-month'); print(a.groupby(['yr','mon'])['copy'].agg(['mean','size']).to_string())
# arrival copy where it matters: late arrivals (landing > sched in-block by > 30 min): copy means BLOCK=SCHED before landing!
late=a[a.sm>1800]
print('late arrivals (MVT > SCHED+30m) copy rate by yr-mon'); print(late.groupby(['yr','mon'])['copy'].agg(['mean','size']).to_string())
# carrier-level comparison Jan/Jul 2025 vs 2026
g=a[a.mon.isin([1,7])].groupby(['car','yr','mon'])['copy'].agg(['mean','size']).unstack(['yr','mon'])
g=g[(g[('size',2026,1)].fillna(0)+g[('size',2026,7)].fillna(0))>=30]
print(g.round(3).sort_values(('size',2026,7),ascending=False).head(60).to_string())
# departures 2025 by car-month vs arrivals: correlation of copy rates
dep=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')&(df.set=='train')].copy()
dep['copy']=(dep.BLOCK_TIME_UTC_mvt-dep.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
dep['car']=dep.FLIGHT_mvt.str[:3]; dep['mon']=dep.MVT_TIME_UTC_mvt.dt.month
dd=dep.groupby(['car','mon'])['copy'].agg(['mean','size']).rename(columns={'mean':'dep_rate','size':'dep_n'})
aa=a[a.yr==2025].groupby(['car','mon'])['copy'].agg(['mean','size']).rename(columns={'mean':'arr_rate','size':'arr_n'})
j=dd.join(aa,how='inner'); j=j[(j.dep_n>=30)&(j.arr_n>=30)]
print('carrier-month dep vs arr copy-rate corr', j.dep_rate.corr(j.arr_rate), 'n',len(j))
print(j.describe().to_string())
