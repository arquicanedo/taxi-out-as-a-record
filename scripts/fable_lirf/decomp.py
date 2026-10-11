import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
w=pd.read_parquet('win_lirf.parquet')
w['jn']=w.LOBT_flt.isna()
w['ext']=w.y>=7200
w['msb']=pd.cut(w.ms,[-1e9,600,1800,3600,7200,14400,1e9],labels=['<10m','10-30m','30-60m','1-2h','2-4h','>4h'])
w['ls']=(w.LOBT_flt-w.SCHED_TIME_UTC_mvt).dt.total_seconds()
g=w.groupby(['jn','msb','copy'],observed=True).agg(n=('se','size'),SSE=('se','sum'),y_med=('y','median'),newW_med=('newW','median'),ms_med=('ms','median'))
g['SSE_pct']=100*g.SSE/w.se.sum()
print(g.to_string())
print('\nTOTAL SSE', w.se.sum())
# oracle: perfect copy knowledge; non-copy rows get min(newW, 2500) (crude taxi) 
taxi_typ=np.where(w.newW<2500,w.newW,1000.0)
orc=np.where(w['copy'],w.ms,taxi_typ)
print('oracle perfect-copy SSE', ((w.y-orc)**2).sum(), 'removed', w.se.sum()-((w.y-orc)**2).sum())
# oracle restricted: only on rows where ms>=1800
m=w.ms>=1800
o2=np.where(m,orc,w.newW)
print('oracle on ms>=1800 rows SSE removed', w.se.sum()-((w.y-o2)**2).sum(), 'rows',m.sum())
# join-present LOBT-SCHED distribution
jp=w[~w.jn]
print('LOBT-SCHED quantiles (join-present):', jp.ls.quantile([0,.01,.1,.5,.9,.99,.999,1]).round(0).to_dict())
print('join-null ms quantiles:', w[w.jn].ms.quantile([0,.05,.25,.5,.75,.95,1]).round(0).to_dict())
# copy rate vs ls band among join-present with ms>=1800
jp2=jp[jp.ms>=1800].copy()
jp2['lsb']=pd.cut(jp2.ls,[-1e9,-1,300,1800,3600,7200,1e9])
print(jp2.groupby(['lsb'],observed=True).agg(n=('se','size'),copy=('copy','mean'),SSE=('se','sum')).to_string())
# residual on copy rows: what did the model predict relative to ms? implied weight
w['wimp']=(w.newW-900)/(w.ms-900)
c=w[(w.ms>=1800)]
print(c.groupby(['jn','copy']).wimp.describe().to_string())
