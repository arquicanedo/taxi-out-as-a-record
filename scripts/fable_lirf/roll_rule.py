import os
import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); t=d[d.set=='train'].copy(); win=load_window(d)
t['taxi_est']=np.where(t.roll,t.y-86400,t.y)
t['rdelay']=t.ms-t.taxi_est   # real delay for real/roll rows
rr=t[t.cls.isin(['real','roll'])&(t.ms>=14400)]
rr['band']=pd.cut(rr.rdelay,[-1e9,30000,39600,43200,46800,50400,1e9])
print('real vs roll by real delay band (ms>=4h):'); print(pd.crosstab([rr.jn,rr.band],rr.cls))
print(rr[rr.rdelay>=39600][['src','FLIGHT_mvt','jn','cross','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','y','ms','rdelay','cls','hour']].sort_values('rdelay').to_string())
# copies in the same ms range
cp=t[(t.ms>=39600)]
print('all rows ms>=11h:'); print(pd.crosstab([cp.jn,pd.cut(cp.ms,[0,43200,46800,50400,1e9])],cp.cls))
# window lever: jn & ms>=45000: pred = ms + P*(87300-ms); P cross-fitted by month from jn & ms>=45000 rows
pred=win.newW.copy()
for mon in [1,7]:
    tr=t[(t.mon!=mon)&t.jn&(t.ms>=45000)]
    P=(tr.cls=='roll').mean(); print(mon,'P_roll train', round(P,3), 'n', len(tr), tr.cls.value_counts().to_dict())
    idx=(win.mon==mon)&win.jn&(win.ms>=45000)
    pred[idx]=win.ms[idx]+P*(86400+900-win.ms[idx])
report('cell B: jn & ms>=45000 copy/roll mixture', win, pred)
w=win[win.jn&(win.ms>=45000)]; print(w[['src','FLIGHT_mvt','y','newW','ms','cls']].assign(pred=pred[w.index]).to_string())
# variant: ms in [30000,43200): roll impossible -> cap newW at ms when newW>ms
pred2=win.newW.copy(); idx=win.jn&(win.ms>=30000)&(win.ms<43200)&(win.newW>win.ms); pred2[idx]=win.ms[idx]
report('cell B2: jn & 30000<=ms<43200 cap at ms', win, pred2)
print(win[win.jn&(win.ms>=30000)&(win.ms<43200)][['src','FLIGHT_mvt','y','newW','ms','cls']].to_string())
# 2026 exposure
r=d[d.set=='rank']; sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'}); r=r.merge(sub,on='MVT_ID_mvt')
s=r[r.jn&(r.ms>=30000)]; print('2026 jn & ms>=30000 rows', len(s)); print(s[['mon','FLIGHT_mvt','pat','ms','pred','cross']].sort_values('ms').to_string())
