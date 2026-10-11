"""The final model (v121) on the labelled window, for the paper (v2, sections 5.3, 6.1 and 6.4) and MODELS.md.

v121 = v117 (the trajectory stage with its blend weight refitted outside LIRF) + more seeds in three members (v118) + two
record rules read from labelled arrivals (v120, v121). On the window (January and July 2025):
  v117      its cross-fitted window vector, saved by scripts/adsb_price_v117.py (cand/v117_adsb.npz, 'newW')
  v118      no window counterpart: the extra seeds were trained for the scored side only, so the window vector is v117's
  v120      scripts/v120_build.py's gate (a month in which THY's early arrivals at LIRF are >= 90% copied on >= 10) is
            checked on the window months; it opens on none, so v120 changes no window row
  v121      scripts/v121_build.py's rule replayed with everything it learns from labelled departures (follower classes,
            follow rates, class hedge, ms support) taken from the ten 2025 months outside the window, and outage spans
            read from the window months' own arrivals, as the 2026 build reads them from the 2026 arrivals. The window's
            base is the validation version, which lacks the submitted chain's carrier cell (scripts/v96_build.py: eleven
            LIRF carriers committed to the copy); on the 2026 side those rows already sit at the copy and the rule
            cannot move them, so the replay leaves those eleven carriers out (V96_OUT=0 keeps them, for comparison)
Writes cand/v121_window.npz (win_ids, newW) for paper_baselines.py, paper_window_ladder.py and paper_pushback_split.py.

  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/paper_final_window.py
"""
import contextlib, glob, io, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'adsb_stage_exp.py')
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_price.py')).read().split('# ---------------- 2. ADS-B features')[0])
z = np.load(f'{I.CAND}/v117_adsb.npz'); assert (z['win_ids'] == ctx['ids']).all(); v117w = z['newW'].astype(float)
ids = ctx['ids']; N_RUN, MIN_INSIDE = 5, 10
V96 = {'TKJ', 'ISR', 'TWB', 'LAV', 'QFA', 'UPS', 'HAT', 'TGZ', 'FIE', 'ANE', 'ENT'} if os.environ.get('V96_OUT', '1') == '1' else set()
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))])
T = T.drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
for c in ('MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'): T[c] = pd.to_datetime(T[c], utc=True)
T['car'] = T.FLIGHT_mvt.astype(str).str[:3]; T['mon'] = T.MVT_TIME_UTC_mvt.dt.month
T['ms'] = (T.MVT_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds()
T['cp'] = (T.BLOCK_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
A = T[T.PHASE_mvt.eq('ARR') & T.ADES_mvt.eq('LIRF') & (T.ms < -600)].sort_values('MVT_TIME_UTC_mvt')
Dp = T[T.PHASE_mvt.eq('DEP') & T.ADEP_mvt.eq('LIRF')]
WM = [1, 7]
# v120's gate on the window months
thy = A[A.car.eq('THY') & A.mon.isin(WM)].groupby('mon').cp.agg(['size', 'mean'])
print('v120 gate, THY early arrivals at LIRF in the window months: ' + '  '.join(f'm{m} {int(r["size"])} at {r["mean"]:.1%}' for m, r in thy.iterrows())
      + f'  -> opens on {int(((thy["size"] >= 10) & (thy["mean"] >= 0.9)).sum())} month(s)')
# v121's rule: spans from all 2025 arrivals (as the 2026 build reads its own year), classes from the ten other months
spans = []
for car, g in A.groupby('car'):
    c = g.cp.values; t = g.MVT_TIME_UTC_mvt.values; i = 0
    while i < len(c):
        if c[i]:
            j = i
            while j + 1 < len(c) and c[j + 1]: j += 1
            if j - i + 1 >= N_RUN: spans.append((car, t[i], t[j]))
            i = j + 1
        else: i += 1
SP = pd.DataFrame(spans, columns=['car', 'start', 'end'])
def inside(df, sp):
    m = np.zeros(len(df), bool); t = df.MVT_TIME_UTC_mvt.dt.tz_localize(None).values; cr = df.car.values
    for _, s in sp.iterrows(): m |= (cr == s.car) & (t >= s.start) & (t <= s.end)
    return m
Dtr = Dp[~Dp.mon.isin(WM)]; intr = inside(Dtr, SP)
per = pd.DataFrame({'car': Dtr.car.values[intr], 'cp': Dtr.cp.values[intr]}).groupby('car').cp.agg(['size', 'mean'])
per = per[per['size'] >= MIN_INSIDE]; fol = per['mean'] >= 0.5; PI = float(fol.mean())
I10 = pd.DataFrame({'car': Dtr.car.values[intr], 'cp': Dtr.cp.values[intr]})
qF = float(I10.cp[I10.car.isin(per.index[fol])].mean()); qN = float(I10.cp[I10.car.isin(per.index[~fol])].mean())
HW = PI * qF + (1 - PI) * qN; lo, hi = np.quantile(Dtr.ms.values[intr], [.01, .99])
seen = set(SP.car[~pd.to_datetime(SP.start).dt.month.isin(WM) | ~pd.to_datetime(SP.end).dt.month.isin(WM)])
print(f'v121 rule, learned on the ten other months: {len(per)} span carriers, {int(fol.sum())} follow (PI {PI:.3f}), '
      f'followers {qF:.3f}, others {qN:.3f}, class hedge {HW:.3f}; ms support {lo:.0f}..{hi:.0f}')
Rw = Dp[Dp.mon.isin(WM)].set_index('MVT_ID_mvt').reindex(ids); ok = Rw.MVT_TIME_UTC_mvt.notna().values
inw = np.zeros(len(ids), bool); inw[ok] = inside(Rw[ok].assign(car=Rw.car[ok]), SP)
bulk = cl(np.rint(M @ wF + SHW)); Ak = cl(ctx['vsd']); ms = ctx['vsd']
with np.errstate(divide='ignore', invalid='ignore'): p = np.clip((v117w - bulk) / (Ak - bulk), 0, 1)
tgt = np.full(len(ids), np.nan); cars = Rw.car.fillna('').values
for cr in np.unique(cars[inw]):
    k = inw & (cars == cr)
    if cr in per.index and fol.get(cr, False): tgt[k] = per.loc[cr, 'mean']
    elif cr not in seen: tgt[k] = HW
fire = inw & ~np.isin(cars, list(V96)) & np.isfinite(tgt) & (ms >= lo) & (ms <= hi) & np.isfinite(p) & (p < tgt)
fin = v117w.copy(); fin[fire] = cl(np.rint(tgt[fire] * Ak[fire] + (1 - tgt[fire]) * bulk[fire]))
d = fin - v117w; cpw = np.abs(y - ms) <= 60
print(f'  window departures inside a window span: {int(inw.sum())}; fired {int(fire.sum())} '
      f'({", ".join(f"{c} {int((fire & (cars == c)).sum())}" for c in np.unique(cars[fire]))}); copies among them {int(cpw[fire].sum())}')
rm = lambda q, m: float(np.sqrt(np.mean((q[m] - y[m]) ** 2))); allm = np.ones(len(y), bool)
G = float(((v117w - y) ** 2 - (fin - y) ** 2).sum()); Qw = float((d ** 2).sum())
days = ctx['cluster']; k = days.max() + 1; rng = np.random.default_rng(13)
g_day = np.bincount(days, weights=(v117w - y) ** 2 - (fin - y) ** 2, minlength=k)
bs = [g_day[rng.integers(0, k, k)].sum() for _ in range(2000)]
print(f'  window SSE removed {G:+.4e} [95% day-cluster {np.percentile(bs, 2.5):+.3e}, {np.percentile(bs, 97.5):+.3e}]; Q_w {Qw:.4e}; '
      f'-dSSE/Q {G / Qw if Qw else float("nan"):.3f}')
print(f'  RMSE all {rm(v117w, allm):.3f} -> {rm(fin, allm):.3f}; ordinary {rm(v117w, O):.3f} -> {rm(fin, O):.3f}')
if V96:
    np.savez(f'{I.CAND}/v121_window.npz', win_ids=ids, newW=fin); print(f'  wrote {I.CAND}/v121_window.npz')
# v119 (not kept): scripts/v119_build.py's never-joins cell replayed on the window, rate and ms support from the ten
# other months; "never joins" is read over 2025 and 2026 as the build reads it (a join is not a label)
c2 = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'LOBT_flt']
L2 = pd.concat([pq.read_table(f, columns=c2).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']])
L2 = L2[L2.PHASE_mvt.eq('DEP') & L2.ADEP_mvt.eq('LIRF')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
known = set(L2.FLIGHT_mvt[L2.LOBT_flt.notna()].astype(str))
garb = lambda s: s.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d') | s.astype(str).str.match(r'^[A-Z]{2}([A-Z])\1')
Dl = Dp.join(L2.LOBT_flt, on='MVT_ID_mvt')
cell = Dl.LOBT_flt.isna() & (Dl.ms >= 3600) & ~Dl.FLIGHT_mvt.astype(str).isin(known) & ~garb(Dl.FLIGHT_mvt)
pr = Dl[cell & ~Dl.mon.isin(WM)]; R9 = float(pr.cp.mean()); lo9, hi9 = np.quantile(pr.ms, [.01, .99])
cw = Dl[cell & Dl.mon.isin(WM)].MVT_ID_mvt; m9 = pd.Index(ids).isin(cw) & (ms >= lo9) & (ms <= hi9) & np.isfinite(p) & (p < R9)
f9 = v117w.copy(); f9[m9] = cl(np.rint(R9 * Ak[m9] + (1 - R9) * bulk[m9]))
g9 = float(((v117w - y) ** 2 - (f9 - y) ** 2).sum()); gd9 = np.bincount(days, weights=(v117w - y) ** 2 - (f9 - y) ** 2, minlength=k)
bs9 = [gd9[rng.integers(0, k, k)].sum() for _ in range(2000)]
print(f'v119 on the window: ten-month rate {R9:.3f} on {len(pr)} rows, support {lo9:.0f}..{hi9:.0f}; window cell {len(cw)}, fired {int(m9.sum())}, '
      f'copies {int(cpw[m9].sum())}; SSE removed {g9:+.4e} [{np.percentile(bs9, 2.5):+.3e}, {np.percentile(bs9, 97.5):+.3e}]; '
      f'ordinary RMSE {rm(v117w, O):.3f} -> {rm(f9, O):.3f}')
# ordinary-row RMSE gains with the ladder's day-cluster interval (paper_window_ladder.py), for Table 18
rg = np.random.default_rng(11); idx = [rg.integers(0, k, k) for _ in range(2000)]
def ci(p0, p1, m):
    e0 = np.bincount(days[m], weights=((p0 - y) ** 2)[m], minlength=k); e1 = np.bincount(days[m], weights=((p1 - y) ** 2)[m], minlength=k)
    n = np.bincount(days[m], minlength=k); d = [np.sqrt(e0[i].sum() / n[i].sum()) - np.sqrt(e1[i].sum() / n[i].sum()) for i in idx]
    return np.percentile(d, [2.5, 97.5])
for nm, q in (('v121 rule', fin), ('v119 cell', f9)):
    c = ci(v117w, q, O); print(f'{nm} on the window, ordinary rows: {rm(v117w, O) - rm(q, O):+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]; '
                                f'fired rows that are extremes {int((ext & (q != v117w)).sum())}')
# v121's window replay by carrier: which arm fires, and where the cost comes from
gain_row = (v117w - y) ** 2 - (fin - y) ** 2
for c in np.unique(cars[fire]):
    m = fire & (cars == c); arm = 'own 2025 rate' if c in per.index and fol.get(c, False) else 'class hedge'
    print(f'  {c}: {int(m.sum())} rows at w {np.nanmax(tgt[m]):.3f} ({arm}); copies {int(cpw[m].sum())} gain {gain_row[m & cpw].sum():+.3e}; '
          f'non-copies {int((m & ~cpw).sum())} gain {gain_row[m & ~cpw].sum():+.3e}')
print(f'  all fired: copies {gain_row[fire & cpw].sum():+.4e}, non-copies {gain_row[fire & ~cpw].sum():+.4e}')
