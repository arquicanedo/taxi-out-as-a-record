import os
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
r=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')&(df.set=='rank')].copy()
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt',how='left'); print('2026 LIRF deps',len(r),'pred null',r.pred.isna().sum())
r['jn']=r.LOBT_flt.isna(); r['ms']=(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt).dt.total_seconds()
r['mon']=r.MVT_TIME_UTC_mvt.dt.month
r['msb']=pd.cut(r.ms,[-1e9,1800,3600,7200,14400,1e9],labels=['<30m','30-60m','1-2h','2-4h','>4h'])
loc=r.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome'); sl=r.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome')
r['cross']=(loc.dt.date>sl.dt.date)
r['wimp']=(r.pred-900)/(r.ms-900)
print(r.groupby(['mon','jn','msb'],observed=True).agg(n=('pred','size'),pred_med=('pred','median'),ms_med=('ms','median'),wimp_med=('wimp','median'),cross=('cross','sum')).to_string())
print('pred>=7200:', (r.pred>=7200).sum(), 'pred>=40000', (r.pred>=40000).sum())
print(r[r.pred>=20000][['mon','FLIGHT_mvt','ADES_mvt','STAND_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','ms','pred','jn','cross']].sort_values('pred',ascending=False).to_string())
# 2025 same cells for comparison
t=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')&(df.set=='train')&(df.src.isin(['2025-01','2025-07']))].copy()
t['jn']=t.LOBT_flt.isna(); t['ms']=(t.MVT_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds(); t['mon']=t.MVT_TIME_UTC_mvt.dt.month
t['msb']=pd.cut(t.ms,[-1e9,1800,3600,7200,14400,1e9],labels=['<30m','30-60m','1-2h','2-4h','>4h'])
t['copy']=(t.BLOCK_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
print(t.groupby(['mon','jn','msb'],observed=True).agg(n=('y','size'),copy=('copy','mean')).to_string())
