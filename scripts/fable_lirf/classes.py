import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); t=d[d.set=='train']
print('train LIRF deps', len(t)); print(t.cls.value_counts())
t['msb']=pd.cut(t.ms,[-1e9,1800,3600,7200,14400,1e9],labels=['<30m','30-60m','1-2h','2-4h','>4h'])
print(pd.crosstab([t.jn,t.msb],t.cls))
print('\nOTHER rows (join-null):'); o=t[(t.cls=='other')&t.jn]
print(o[['src','FLIGHT_mvt','ADES_mvt','STAND_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','y','bs','ms','cross']].sort_values('y').to_string())
print('\nROLL rows:'); print(t[t.roll][['src','FLIGHT_mvt','ADES_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','y','bs','ms','ml','cross','jn','hour']].to_string())
# cross>0 rows: class distribution by jn and ms
c=t[t.cross>0]
print('\ndate-crossing rows', len(c)); print(pd.crosstab([c.jn,pd.cut(c.ms,[-1e9,3600,7200,14400,30000,1e9])],c.cls))
t.to_parquet('lirf_train_cls.parquet',index=False)
