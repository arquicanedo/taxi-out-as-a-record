import os
import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); win=load_window(d)
m=pd.read_parquet('dep_destarr.parquet')[['MVT_ID_mvt','a_LOBT_flt','a_AOBT_3_flt','a_IOBT_flt','a_EOBT_1_flt','a_CALLSIGN_flt','a_set']]
t=d[d.set=='train'].merge(m,on='MVT_ID_mvt',how='left')
t['a_ml']=(t.MVT_TIME_UTC_mvt-t.a_LOBT_flt).dt.total_seconds(); t['a_ma']=(t.MVT_TIME_UTC_mvt-t.a_AOBT_3_flt).dt.total_seconds()
jn=t[t.jn&t.a_LOBT_flt.notna()].copy()
print('join-null train rows with recovered LOBT', len(jn), 'of', t.jn.sum())
jn['far']=jn.a_ml>TOL
print(pd.crosstab(jn.far,jn.cls))
print('far rows: y vs a_ma'); f=jn[jn.far]; print((f.y-f.a_ma).describe())
print('near rows (a_ml<=TOL): y - ms'); n=jn[~jn.far]; print((n.y-n.ms).describe()); print(n[['src','FLIGHT_mvt','y','ms','a_ml','a_ma','cls']].to_string())
# window lever
w=win.merge(m,on='MVT_ID_mvt',how='left')
w['a_ml']=(w.MVT_TIME_UTC_mvt-w.a_LOBT_flt).dt.total_seconds(); w['a_ma']=(w.MVT_TIME_UTC_mvt-w.a_AOBT_3_flt).dt.total_seconds()
sel=w.jn&w.a_LOBT_flt.notna()
print('window join-null rows with recovery', sel.sum())
pred=w.newW.copy()
far=sel&(w.a_ml>TOL); near=sel&(w.a_ml<=TOL)
pred[far]=np.where(w.a_ma[far].notna(), w.a_ma[far], 900.0)
pred[near]=w.ms[near]
report('dest-arrival recovery (far->AOBT, near->copy)', w, pred)
print(w[sel][['src','FLIGHT_mvt','ADES_mvt','y','newW','ms','a_ml','a_ma','cls']].assign(pred=pred[sel]).to_string())
# 2026 exposure
r=d[d.set=='rank'].merge(m,on='MVT_ID_mvt',how='left')
r['a_ml']=(r.MVT_TIME_UTC_mvt-r.a_LOBT_flt).dt.total_seconds(); r['a_ma']=(r.MVT_TIME_UTC_mvt-r.a_AOBT_3_flt).dt.total_seconds()
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt')
s=r[r.jn&r.a_LOBT_flt.notna()]
print('2026 join-null LIRF rows with recovered LOBT', len(s), 'of', r.jn.sum())
s=s.assign(far=s.a_ml>TOL, newpred=np.where(s.a_ml>TOL, np.where(s.a_ma.notna(),s.a_ma,900.0), s.ms))
print(s[['mon','FLIGHT_mvt','ADES_mvt','ms','pred','a_ml','a_ma','far','newpred']].to_string())
print('2026 sum (pred-newpred)^2 =', ((s.pred-s.newpred)**2).sum())
