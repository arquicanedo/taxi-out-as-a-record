import os
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',200)
df=pd.read_parquet('lirf.parquet')
dep=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')&(df.set=='train')].copy()
z=np.load(os.path.join(os.environ['DC26_WORK'], 'adsblol', 'cand', 'v121_window.npz'))
w=pd.DataFrame({'MVT_ID_mvt':z['win_ids'],'newW':z['newW']})
print('window ids unique', w.MVT_ID_mvt.is_unique, len(w))
win=dep.merge(w,on='MVT_ID_mvt',how='inner')
print('LIRF window rows', len(win), 'months', win.src.value_counts().to_dict())
print('LIRF Jan+Jul deps in train', dep.src.isin(['2025-01','2025-07']).sum())
win['res']=win.y-win.newW; win['se']=win.res**2
print('LIRF window RMSE', np.sqrt(win.se.mean()), 'SSE', win.se.sum())
ext=win.y>=7200
print('ordinary rows', (~ext).sum(), 'SSE', win.se[~ext].sum(), 'RMSE', np.sqrt(win.se[~ext].mean()))
print('extreme rows', ext.sum(), 'SSE', win.se[ext].sum())
# top 50 by se
win['bs']=(win.BLOCK_TIME_UTC_mvt-win.SCHED_TIME_UTC_mvt).dt.total_seconds()
win['ms']=(win.MVT_TIME_UTC_mvt-win.SCHED_TIME_UTC_mvt).dt.total_seconds()
win['ml']=(win.MVT_TIME_UTC_mvt-win.LOBT_flt).dt.total_seconds()
win['ma']=(win.MVT_TIME_UTC_mvt-win.AOBT_3_flt).dt.total_seconds()
win['me']=(win.MVT_TIME_UTC_mvt-win.EOBT_1_flt).dt.total_seconds()
win['mi']=(win.MVT_TIME_UTC_mvt-win.IOBT_flt).dt.total_seconds()
win['op']=win.AIRCRAFT_OPERATOR_flt.str[:6]
win['copy']=win.bs.abs()<=60
cols=['MVT_ID_mvt','src','FLIGHT_mvt','op','ADES_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','y','newW','res','bs','ms','ml','ma','me','mi','copy']
top=win.sort_values('se',ascending=False).head(50)
print(top[cols].to_string())
print('\nTOP ORDINARY')
topo=win[~ext].sort_values('se',ascending=False).head(50)
print(topo[cols].to_string())
win.to_parquet('win_lirf.parquet',index=False)
