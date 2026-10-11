"""Export lever L5 (the coin-flip copy classifier) for the 2026 LIRF rows, exactly as levers.py applies it: rows that are
join-null with 3600 <= ms < 14400 (the classifier's cell), not recoverable through the destination arrival (L2), with
mix = taxi_hat + p_clf * (ms - taxi_hat); taxi_hat is the median 2025 real taxi by 3-hour bucket. Writes l5_2026.parquet."""
import pandas as pd, numpy as np
from base import *; from feats import *
d = build(); t = d[d.set == 'train'].copy(); t['hb'] = t.hour // 3
r = d[d.set == 'rank'].copy(); r['hb'] = r.hour // 3
m = pd.read_parquet('dep_destarr.parquet')[['MVT_ID_mvt', 'a_LOBT_flt', 'a_AOBT_3_flt']]
r = r.merge(m, on='MVT_ID_mvt', how='left'); r['a_ml'] = (r.MVT_TIME_UTC_mvt - r.a_LOBT_flt).dt.total_seconds()
c26 = pd.read_parquet('cf_clf_2026.parquet')[['MVT_ID_mvt', 'p_clf']]; r = r.merge(c26, on='MVT_ID_mvt', how='left')
med = t[t.cls == 'real'].groupby('hb').y.median(); r['taxi_hat'] = r.hb.map(med)
sel = r.jn & r.a_ml.notna() & r.FLIGHT_mvt.notna()
cf = r.p_clf.notna() & (r.ms < 14400) & ~sel
r['mix'] = r.taxi_hat + r.p_clf * (r.ms - r.taxi_hat)
o = r[cf][['MVT_ID_mvt', 'mon', 'ms', 'p_clf', 'taxi_hat', 'mix']]
print(f'L5 2026 rows {len(o)} (Jan {int((o.mon == 1).sum())}, Jul {int((o.mon == 7).sum())}); mean p_clf {o.p_clf.mean():.3f}; taxi_hat by hb {med.round(0).to_dict()}')
o.to_parquet('l5_2026.parquet', index=False); print('wrote l5_2026.parquet')
