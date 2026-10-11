import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
d=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')&(df.set=='train')].copy()
d['bs']=(d.BLOCK_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['jn']=d.LOBT_flt.isna()
d['copy']=d.bs.abs()<=60
loc=d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome')
d['off']=loc.apply(lambda t: t.utcoffset().total_seconds())
d['tz']=((d.y-d.off)>=200)&((d.y-d.off)<=3000)&(d.bs<-1500)
print('TZ-class rows', d.tz.sum(), 'by month', d[d.tz].groupby('src').size().to_dict())
print('by jn', d[d.tz].groupby('jn').size().to_dict())
print(d[d.tz].groupby(d.FLIGHT_mvt.str[:3]).size().sort_values(ascending=False).head(15).to_dict())
print(d[d.tz][['src','FLIGHT_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','y','bs','ms','off','jn']].head(40).to_string())
# how common: among join-null with ms<1800, class split
jn_small=d[d.jn&(d.ms<1800)]
print('join-null ms<1800: n',len(jn_small),'copy',jn_small['copy'].mean(),'tz',jn_small.tz.mean(), 'y median', jn_small.y.median())
print(jn_small.groupby('tz').y.describe().to_string())
# generally: rows with bs < -1500 (block before sched by >25min): class
neg=d[d.bs<-1500]
print('bs<-1500 rows', len(neg), 'tz', neg.tz.sum(), 'jn', neg.jn.sum())
print(neg.groupby('jn').agg(n=('y','size'),tz=('tz','mean'),y_med=('y','median'),ms_med=('ms','median')).to_string())
# window SSE of tz rows (needs newW)
w=pd.read_parquet('win_lirf.parquet')
m=w.merge(d[['MVT_ID_mvt','tz','off']],on='MVT_ID_mvt')
print('window tz rows', m.tz.sum(), 'SSE', m.se[m.tz].sum(), 'of total', m.se.sum())
print(m[m.tz][['src','FLIGHT_mvt','MVT_TIME_UTC_mvt','y','newW','res','bs','ms','off']].to_string())
