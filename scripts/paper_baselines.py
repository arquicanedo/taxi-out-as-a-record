"""Baselines and per-airport accuracy for the paper (v2, section 6.0), on the labelled validation window (January and
July 2025, 344,336 departures). Every number the section adds beyond v1 comes from this script, and
analysis/paper-v2/added.tsv names it as the source.

  naive          MVT - AOBT_3, falling back to the airport-hour median of that difference, then to its global median
  airport median the median 2025 taxi-out of the airport, from the ten months outside the window
  base model     v100 on the window, rebuilt exactly as scripts/adsb_price.py does
  final model    v126's window vector (cand/v126_window.npz, scripts/paper_final_window_v126.py on top of
                 paper_final_window.py's v121 vector); FINAL=v125, v124, v121 or v114 gives the earlier final models
All clipped to [60, 200,000] s as the submissions are. Intervals: 95% day-cluster bootstrap (62 days, 2,000 draws).

  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/paper_baselines.py
"""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'adsb_stage_exp.py')
exec(open(os.path.join(HERE, 'adsb_price.py')).read().split('# ---------------- 2. ADS-B features')[0])
W = os.environ['DC26_WORK']
FINAL = os.environ.get('FINAL', 'v126')
z = np.load(f'{I.CAND}/{FINAL}_window.npz' if FINAL in ('v121', 'v124', 'v125', 'v126') else f'{I.CAND}/{FINAL}_adsb.npz'); assert (z['win_ids'] == ctx['ids']).all()
v114w = z['newW'].astype(float)   # the final model's window vector (name kept from the first version of this script)
print(f'final model: {FINAL}')
ids = ctx['ids']
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))])
T = T[T.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt')
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds()
T['naive'] = sec(T.MVT_TIME_UTC_mvt) - sec(T.AOBT_3_flt)
T['hour'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.floor('h'); T['mon'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.month
Wn = T.set_index('MVT_ID_mvt').reindex(ids)
assert np.array_equal(Wn.TAXITIME_SEC_mvt.values.astype(float), y), 'window labels'
hr_med = Wn.groupby(['ADEP_mvt', 'hour']).naive.transform('median'); g_med = Wn.naive.median()
naive = cl(Wn.naive.fillna(hr_med).fillna(g_med).values.astype(float))
out = T[~T.mon.isin([1, 7])]; apt_med = out.groupby('ADEP_mvt').TAXITIME_SEC_mvt.median()
aptm = cl(Wn.ADEP_mvt.map(apt_med).values.astype(float))
P = {'naive MVT - AOBT_3': naive, 'airport median': aptm, 'base model': v100w.astype(float), 'final model': v114w}
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
allm = np.ones(len(y), bool)
print(f'window rows {len(y):,}, ordinary {int(O.sum()):,}, extremes {int(ext.sum())}; AOBT_3 present {Wn.AOBT_3_flt.notna().mean():.1%}')
print(f'\n{"RMSE, s":22s}' + ''.join(f'{k:>20s}' for k in P))
for nm, m in (('all rows', allm), ('ordinary rows', O)):
    print(f'{nm:22s}' + ''.join(f'{rm(p, m):20.3f}' for p in P.values()))
aw = ctx['apt'].astype(str)
print(f'\nordinary rows, by airport\n{"airport":8s}{"rows":>8s}' + ''.join(f'{k:>20s}' for k in P))
for a in sorted(set(aw)):
    m = O & (aw == a); print(f'{a:8s}{int(m.sum()):8d}' + ''.join(f'{rm(p, m):20.1f}' for p in P.values()))
# day-cluster bootstrap of RMSE and of the base -> final difference
rng = np.random.default_rng(7); days = ctx['cluster']; k = days.max() + 1; B = 2000
for nm, m in (('all rows', allm), ('ordinary rows', O)):
    e_b = np.bincount(days[m], weights=((v100w - y) ** 2)[m], minlength=k); e_f = np.bincount(days[m], weights=((v114w - y) ** 2)[m], minlength=k)
    n = np.bincount(days[m], minlength=k); rb, rf, dd = [], [], []
    for _ in range(B):
        i = rng.integers(0, k, k); nn = n[i].sum(); b = np.sqrt(e_b[i].sum() / nn); f = np.sqrt(e_f[i].sum() / nn)
        rb.append(b); rf.append(f); dd.append(b - f)
    q = lambda a: np.percentile(a, [2.5, 97.5])
    print(f'\n{nm}: base {rm(v100w, m):.3f} [{q(rb)[0]:.3f}, {q(rb)[1]:.3f}]  final {rm(v114w, m):.3f} [{q(rf)[0]:.3f}, {q(rf)[1]:.3f}]  '
          f'base - final {rm(v100w, m) - rm(v114w, m):.3f} [{q(dd)[0]:.3f}, {q(dd)[1]:.3f}]  (95%, {k} days)')
