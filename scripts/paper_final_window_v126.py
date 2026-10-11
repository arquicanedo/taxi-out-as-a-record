"""The final model (v126) on the labelled window, for the paper (Sections 4.9, 5.3, 6) and MODELS.md.

v124 = v121 + the flight-plan join rule (v122: the bound clip at every airport and LIRF's recovered-plan decisive branch)
+ the coin-flip classifier (v123) + the recovered-plan near branch (v124). Replayed on January and July 2025 on top of
v121's window vector (cand/v121_window.npz, scripts/paper_final_window.py):
  L1   join-present rows, every airport: clip onto [MVT - LOBT - 3606, MVT - LOBT + 3606] (no parameter)
  L2   LIRF join-null rows whose plan is recovered from the destination arrival, took off > 3606 s after its LOBT:
       y = MVT - AOBT_3 of the arrival row (no parameter)
  L5   LIRF join-null, 3600 <= ms < 14400, not recovered: 0.5 * (taxi_hat + P (ms - taxi_hat)) + 0.5 * current, with P the
       classifier's leave-one-month-out probability (scripts/fable_lirf/cf_clf2.py -> win_cf_pred.parquet) and taxi_hat the
       median real LIRF taxi by local 3-hour bucket over the ten other 2025 months
  near LIRF join-null, recovered, took off within 3606 s of its LOBT: raise to MVT - LOBT + 3606; where ms exceeds that,
       move toward clip(ms) at the copy rate counted on the other eleven 2025 months
  L3   (v125) LIRF join-null, ms < 1800, not recovered: the class mixture p_real * 1000 + p_tz * (1000 + offset) +
       p_copy * clip(ms), class rates per carrier group counted on the other eleven 2025 months
  floor (v126) LIRF join-null, clean identifier (not CHAR(8)), 30000 <= ms < 46800: clip(ms); 46800 <= ms < 86000:
       max(prediction, clip(ms)). No parameter is fitted; the band edges were read off all of 2025, window months included
The classifier and the two counted rates leave out only the row's own month (eleven months); taxi_hat leaves out both
window months (ten); L1 and L2 learn nothing.
Writes cand/v124_window.npz, v125_window.npz and v126_window.npz (win_ids, newW) and prints each step's window gain with day-clustered 95% intervals.
  uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python scripts/paper_final_window_v124.py
"""
import contextlib, glob, io, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'adsb_stage_exp.py')
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_price.py')).read().split('# ---------------- 2. ADS-B features')[0])
FD = os.environ.get('FABLE_DIR', os.path.join(os.environ['DC26_WORK'], 'fable_lirf'))
z = np.load(f'{I.CAND}/v121_window.npz'); assert (z['win_ids'] == ctx['ids']).all(); v121w = z['newW'].astype(float)
ids = ctx['ids']; TOL = 3606.0
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))]).drop_duplicates('MVT_ID_mvt')
E = pd.Timestamp('1970-01-01', tz='UTC')
for c, k in (('MVT_TIME_UTC_mvt', 'mvt'), ('SCHED_TIME_UTC_mvt', 'sch'), ('LOBT_flt', 'lobt'), ('AOBT_3_flt', 'aobt')):
    T[k] = (pd.to_datetime(T[c], utc=True) - E).dt.total_seconds()
assert 1.735e9 < T.mvt.median() < 1.77e9
T['mon'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.month
loc = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert('Europe/Rome'); T['hb'] = loc.dt.hour // 3
dep, arr = T[T.PHASE_mvt.eq('DEP')], T[T.PHASE_mvt.eq('ARR')]
L = dep[dep.ADEP_mvt.eq('LIRF') & dep.FLIGHT_mvt.notna()][['MVT_ID_mvt', 'FLIGHT_mvt', 'ADES_mvt', 'mvt', 'sch', 'lobt', 'mon', 'TAXITIME_SEC_mvt']]
A = arr[['FLIGHT_mvt', 'ADES_mvt', 'mvt', 'lobt', 'aobt']].rename(columns={'mvt': 'a_mvt', 'lobt': 'a_lobt', 'aobt': 'a_aobt'})
M = L.merge(A, on=['FLIGHT_mvt', 'ADES_mvt'], how='inner'); M = M[(M.a_mvt - M.mvt).between(900, 21600)]
M = M.sort_values('a_mvt').drop_duplicates('MVT_ID_mvt'); M['ml'] = M.mvt - M.a_lobt; M['ms'] = M.mvt - M.sch
M['y'] = M.TAXITIME_SEC_mvt.astype(float); M['iscp'] = (M.y - M.ms).abs() <= 60
R = M[M.lobt.isna() & M.a_lobt.notna()].set_index('MVT_ID_mvt')
W = dep.set_index('MVT_ID_mvt').reindex(ids)
apt = ctx['apt'].astype(str); mvt, lobt = W.mvt.values, W.lobt.values; ms = ctx['vsd']
days = ctx['cluster']; k = days.max() + 1; rg = np.random.default_rng(11); idx = [rg.integers(0, k, k) for _ in range(2000)]
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2))); allm = np.ones(len(y), bool)
def step(nm, p0, p1):
    g = (p0 - y) ** 2 - (p1 - y) ** 2; gd = np.bincount(days, weights=g, minlength=k); bs = [gd[i].sum() for i in idx]
    e0 = np.bincount(days[O], weights=((p0 - y) ** 2)[O], minlength=k); e1 = np.bincount(days[O], weights=((p1 - y) ** 2)[O], minlength=k)
    n = np.bincount(days[O], minlength=k); do = [np.sqrt(e0[i].sum() / n[i].sum()) - np.sqrt(e1[i].sum() / n[i].sum()) for i in idx]
    print(f'  {nm:34s} rows {int((p1 != p0).sum()):5d}  SSE removed {g.sum():+.4e} [{np.percentile(bs, 2.5):+.3e}, {np.percentile(bs, 97.5):+.3e}]  '
          f'ordinary RMSE gain {rm(p0, O) - rm(p1, O):+.3f} [{np.percentile(do, 2.5):+.3f}, {np.percentile(do, 97.5):+.3f}]  all {rm(p1, allm):.3f}')
    return p1
p = v121w.copy()
# L1
lo = np.maximum(mvt - lobt - TOL, 60); hi = mvt - lobt + TOL; jp = np.isfinite(lo) & np.isfinite(hi)
assert int(((y < mvt - lobt - TOL) | (y > hi))[jp].sum()) == 0, 'a window label outside the join bound'
q = p.copy(); q[jp] = np.clip(np.rint(np.clip(p[jp], lo[jp], hi[jp])), 60, 2e5); p = step('L1 join-bound clip', p, q)
# L2 decisive branch
Rw = R.reindex(ids); rec = Rw.ml.notna().values & (apt == 'LIRF')
far = rec & (Rw.ml.values > TOL) & np.isfinite(Rw.a_aobt.values)
q = p.copy(); q[far] = np.clip(np.rint(mvt[far] - Rw.a_aobt.values[far]), 60, 2e5); p = step('L2 recovered plan, decisive branch', p, q)
v122w = p.copy()
# L5 classifier
cf = pd.read_parquet(os.path.join(FD, 'win_cf_pred.parquet')).set_index('MVT_ID_mvt').reindex(ids)
cls = pd.read_parquet(os.path.join(FD, 'lirf_train_cls.parquet'))
th = {mo: cls[(cls.mon != mo) & ~cls.mon.isin([1, 7]) & (cls.cls == 'real')].groupby(cls.hour // 3).y.median() for mo in (1, 7)}
mon = ctx['month'].astype(int); hb = (pd.to_datetime(W.MVT_TIME_UTC_mvt, utc=True).dt.tz_convert('Europe/Rome').dt.hour // 3).values
taxi_hat = np.array([th[mo].get(h, np.nan) if mo in th else np.nan for mo, h in zip(mon, hb)])
cell = (apt == 'LIRF') & np.isnan(lobt) & (ms >= 3600) & (ms < 14400) & ~rec & np.isfinite(cf.p_clf.values)
mix = taxi_hat + cf.p_clf.values * (ms - taxi_hat)
q = p.copy(); q[cell] = np.clip(np.rint(0.5 * mix[cell] + 0.5 * p[cell]), 60, 2e5); p = step('L5 coin-flip classifier', p, q)
v123w = p.copy()
# near branch
near = rec & (Rw.ml.values <= TOL); fl = Rw.ml.values + TOL
q = p.copy(); up = near & (q < fl); q[up] = np.clip(np.rint(fl[up]), 60, 2e5)
hd = near & (ms > fl)
for mo in (1, 7):
    o = M[M.lobt.isna() & M.a_lobt.notna() & (M.ml <= TOL) & (M.ms > M.ml + TOL) & (M.mon != mo)]; r = float(o.iscp.mean())
    s = hd & (mon == mo); q[s] = np.clip(np.rint(r * np.clip(ms[s], 60, 2e5) + (1 - r) * q[s]), 60, 2e5)
    print(f'    near-branch copy rate outside month {mo}: {r:.3f} on {len(o)} rows; window rows hedged {int(s.sum())}')
p = step('near branch: floor and hedge', p, q)
v124w = p.copy()
np.savez(f'{I.CAND}/v124_window.npz', win_ids=ids, newW=v124w, v122W=v122w, v123W=v123w)
# L3, the local-time class (v125)
off = np.array([3600.0 if mo == 1 else 7200.0 for mo in mon])
car = W.FLIGHT_mvt.astype(str).str[:3].values; grp = np.where(np.isin(car, ['BEL', 'AEA', 'AEZ', 'AEE', 'LZB', 'ETH']), 'tzcar', 'rest')
cA = (apt == 'LIRF') & np.isnan(lobt) & (ms < 1800) & ~rec
tr = cls[cls.jn & (cls.ms < 1800)].copy(); tr['grp'] = np.where(tr.car.isin(['BEL', 'AEA', 'AEZ', 'AEE', 'LZB', 'ETH']), 'tzcar', 'rest')
q = p.copy()
for mo in (1, 7):
    for g_ in ('tzcar', 'rest'):
        s_ = tr[(tr.mon != mo) & (tr.grp == g_)]; pt = (s_.cls == 'tz').mean(); pc = (s_.cls == 'copy').mean(); pr = 1 - pt - pc
        sel = cA & (mon == mo) & (grp == g_)
        q[sel] = np.clip(np.rint(pr * 1000 + pt * (1000 + off[sel]) + pc * np.clip(ms[sel], 60, 2e5)), 60, 2e5)
p = step('L3 local-time class', p, q)
print(f'  final (v125 on the window): all {rm(p, allm):.3f}, ordinary {rm(p, O):.3f}; v124 all {rm(v124w, allm):.3f}, ordinary {rm(v124w, O):.3f}; '
      f'v121 all {rm(v121w, allm):.3f}, ordinary {rm(v121w, O):.3f}')
v125w = p.copy()
np.savez(f'{I.CAND}/v125_window.npz', win_ids=ids, newW=v125w, v122W=v122w, v123W=v123w, v124W=v124w)
# the day-scale floor on clean identifiers (v126)
c8 = W.FLIGHT_mvt.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d').values | W.FLIGHT_mvt.astype(str).str.match(r'^[A-Z]{2}([A-Z])\1').values
band = (apt == 'LIRF') & np.isnan(lobt) & ~c8 & (ms >= 30000) & (ms < 86000)
q = p.copy(); lo_ = band & (ms < 46800); hi_ = band & (ms >= 46800)
q[lo_] = np.clip(np.rint(ms[lo_]), 60, 2e5); q[hi_] = np.clip(np.maximum(q[hi_], np.rint(ms[hi_])), 60, 2e5)
p = step('floor, clean identifiers', p, q)
print(f'  final (v126 on the window): all {rm(p, allm):.3f}, ordinary {rm(p, O):.3f}; v125 all {rm(v125w, allm):.3f}, ordinary {rm(v125w, O):.3f}')
np.savez(f'{I.CAND}/v126_window.npz', win_ids=ids, newW=p, v122W=v122w, v123W=v123w, v124W=v124w, v125W=v125w)
print(f'  wrote {I.CAND}/v124_window.npz, v125_window.npz and v126_window.npz')
