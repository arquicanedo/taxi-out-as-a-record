"""Export lever L3 (the local-time record class) for the 2026 LIRF rows, exactly as levers.py applies it: join-null rows
with ms < 1800 not recoverable through L2 (cell A); class rates (real / local-time / copy) counted on all 2025 cell-A
rows per group (BEL, AEA, AEZ, AEE, LZB, ETH vs the rest); prediction p_real*1000 + p_tz*(1000+offset) + p_copy*clip(ms).
Writes l3_2026.parquet."""
import pandas as pd, numpy as np
from base import *; from feats import *
d = build(); t = d[d.set == 'train'].copy(); r = d[d.set == 'rank'].copy()
m = pd.read_parquet('dep_destarr.parquet')[['MVT_ID_mvt', 'a_LOBT_flt']]
r = r.merge(m, on='MVT_ID_mvt', how='left'); r['a_ml'] = (r.MVT_TIME_UTC_mvt - r.a_LOBT_flt).dt.total_seconds()
grp = lambda c: 'tzcar' if c in ('BEL', 'AEA', 'AEZ', 'AEE', 'LZB', 'ETH') else 'rest'
t['grp'] = t.car.map(grp); r['grp'] = r.car.map(grp)
sel = r.jn & r.a_ml.notna() & r.FLIGHT_mvt.notna(); cellA = r.jn & (r.ms < 1800) & ~sel
r['l3'] = np.nan
for g_ in ['tzcar', 'rest']:
    s_ = t[t.jn & (t.ms < 1800) & (t.grp == g_)]; p_tz = (s_.cls == 'tz').mean(); p_copy = (s_.cls == 'copy').mean(); p_real = 1 - p_tz - p_copy
    idx = cellA & (r.grp == g_); r.loc[idx, 'l3'] = p_real * 1000 + p_tz * (1000 + r.off[idx]) + p_copy * r.ms[idx].clip(lower=60)
    print(f'  {g_}: 2025 cell rows {len(s_)}, p_real {p_real:.3f} p_tz {p_tz:.3f} p_copy {p_copy:.3f}; 2026 rows {int(idx.sum())}')
o = r[cellA][['MVT_ID_mvt', 'mon', 'car', 'grp', 'ms', 'off', 'l3']]; o.to_parquet('l3_2026.parquet', index=False)
print(f'wrote l3_2026.parquet: {len(o)} rows')
