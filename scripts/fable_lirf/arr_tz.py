import os
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
a=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')].copy()
a['yin']=(a.BLOCK_TIME_UTC_mvt-a.MVT_TIME_UTC_mvt).dt.total_seconds()
loc=a.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome'); a['off']=loc.map(lambda t: t.utcoffset().total_seconds())
a['car']=a.FLIGHT_mvt.str[:3]; a['yr']=a.MVT_TIME_UTC_mvt.dt.year; a['mon']=a.MVT_TIME_UTC_mvt.dt.month
a['tz']=((a.yin+a.off)>=60)&((a.yin+a.off)<=2500)&(a.yin<-1000)
a['jn']=a.LOBT_flt.isna()
print('arrival TZ rows', a.tz.sum(), 'by yr', a.groupby('yr').tz.sum().to_dict())
print(a[a.tz].groupby(['yr','mon']).size().to_string())
print('by carrier (yr):'); print(pd.crosstab(a[a.tz].car,a[a.tz].yr).to_string() if a.tz.sum() else 'none')
print('arrival TZ rows join-null rate', a[a.tz].jn.mean())
print('yin quantiles', a.yin.quantile([0,.0001,.001,.01]).to_dict())
# other negative taxi-in rows
neg=a[(a.yin<-300)&~a.tz]; print('other negative taxi-in rows', len(neg), neg.groupby('yr').size().to_dict())
print(neg[['yr','FLIGHT_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','yin']].head(20).to_string())
# 2026 departures cell A by carrier
from base import *
d=load_lirf_dep(); r=d[d.set=='rank']; cA=r[r.jn&(r.ms<1800)]
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'}); cA=cA.merge(sub,on='MVT_ID_mvt')
print('2026 cell A rows', len(cA)); print(cA[['mon','FLIGHT_mvt','car','ADES_mvt','ms','off','pred','cross']].to_string())
