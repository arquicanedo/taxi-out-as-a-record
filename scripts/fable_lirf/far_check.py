import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250)
d=load_lirf_dep(); m=pd.read_parquet('dep_destarr.parquet')[['MVT_ID_mvt','a_LOBT_flt','a_AOBT_3_flt']]
t=d[d.set=='train'].merge(m,on='MVT_ID_mvt'); t['a_ml']=(t.MVT_TIME_UTC_mvt-t.a_LOBT_flt).dt.total_seconds(); t['a_ma']=(t.MVT_TIME_UTC_mvt-t.a_AOBT_3_flt).dt.total_seconds()
jn=t[t.jn&t.a_LOBT_flt.notna()]
far=jn[jn.a_ml>TOL]
print('far rows', len(far), 'class', far.cls.value_counts().to_dict())
print(far[far.cls!='real'][['src','FLIGHT_mvt','y','ms','a_ml','a_ma','bs','cls']].to_string())
# what does L2 far rule cost on the 2 copies vs gain on reals (full year, against a naive 'ms' or 'mid' baseline is not available; show y vs a_ma errors)
e=far.y-far.a_ma.fillna(1000); print('far: (y - a_ma) abs quantiles', e.abs().quantile([.5,.9,.99,1]).to_dict(), 'SSE', (e**2).sum())
near=jn[jn.a_ml<=TOL]; pr=np.where(near.ms>near.a_ml+TOL, near.ms, 1000+near.off); e=near.y-pr
print('near rows', len(near), 'class', near.cls.value_counts().to_dict(), '(y-pred) abs quantiles', pd.Series(e).abs().quantile([.5,.9,1]).to_dict(), 'SSE', (e**2).sum())
print(near.assign(pred=pr,err=e)[['src','FLIGHT_mvt','y','ms','a_ml','a_ma','off','cls','pred','err']].to_string())
# consistency: join-present rows -- does a_ml (from the arrival row) equal ml? 
jp=t[~t.jn&t.a_LOBT_flt.notna()]; print('join-present: a_LOBT==LOBT', (jp.a_LOBT_flt==jp.LOBT_flt).mean(), 'n', len(jp))
# 2026 estimates = window gain * (2026 moved / window moved)
est={'L2':(5.8696e8,6.1231e8,2.4474e8),'L3':(1.6398e8,1.0482e8,5.538e7),'L5':(4.7846e8,2.3615e8,2.0046e8),'L5x':(2.5402e8,None,2.0046e8)}
for k,(g,mw,m26) in est.items():
    if mw: print(f'{k}: window gain {g:.3e}, moved {mw:.3e}, ratio {g/mw:.2f}; 2026 moved {m26:.3e} -> est 2026 gain {g*m26/mw:.3e}')
