"""Numbers for the paper (v2, Section 4.8): a ground handler's data feed fails, and the arrivals show it.
Labelled 2025 departures and arrivals, and the 2026 arrivals the scored file labels. A schedule copy is a block time within
60 s of the scheduled time; on arrivals it is counted on early arrivals (landing > 600 s before schedule), the only ones on
which a copied in-block is distinguishable from a real one; on departures on delayed ones (>= 900 s) unless stated.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/paper_feed_outage.py
"""
import glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
fs = sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in fs]).drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
for c in ('MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'): T[c] = pd.to_datetime(T[c], utc=True)
T['car'] = T.FLIGHT_mvt.astype(str).str[:3]; T['yr'] = T.MVT_TIME_UTC_mvt.dt.year; T['mon'] = T.MVT_TIME_UTC_mvt.dt.month
T['ms'] = (T.MVT_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds()
T['cp'] = (T.BLOCK_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
dep = T[T.PHASE_mvt.eq('DEP')]; arr = T[T.PHASE_mvt.eq('ARR') & (T.ms < -600)]
L_dep = dep[dep.ADEP_mvt.eq('LIRF')]; L_arr = arr[arr.ADES_mvt.eq('LIRF')]
# 1. one carrier's switch, month by month
th_d = L_dep[(L_dep.car == 'THY') & (L_dep.yr == 2025)]; th_a = L_arr[L_arr.car == 'THY']
print('1. THY at LIRF: departures copied (all), early arrivals copied, by month')
for y, m in [(2025, k) for k in range(1, 13)] + [(2026, 1), (2026, 7)]:
    d = th_d[th_d.mon == m] if y == 2025 else None; a = th_a[(th_a.yr == y) & (th_a.mon == m)]
    print(f'   {y}-{m:02d}  departures {"" if d is None else f"{len(d):4d} copied {d.cp.mean():6.1%}"}   early arrivals {len(a):3d} copied {a.cp.mean():6.1%}')
nd = th_d[th_d.mon.isin([11, 12])]; print(f'   Nov+Dec 2025 departures: {int(nd.cp.sum())} of {len(nd)} copied ({nd.cp.mean():.1%})')
# 2. within-carrier co-movement, 2025, carrier-months with >= 15 of each
a = L_arr[L_arr.yr == 2025].groupby(['car', 'mon']).cp.agg(['size', 'mean']).rename(columns={'size': 'na', 'mean': 'ra'})
d = L_dep[(L_dep.yr == 2025) & (L_dep.ms >= 900)].groupby(['car', 'mon']).cp.agg(['size', 'mean']).rename(columns={'size': 'nd', 'mean': 'rd'})
m = a.join(d, how='inner'); m = m[(m.na >= 15) & (m.nd >= 15)].reset_index(); m = m.groupby('car').filter(lambda g: len(g) >= 6)
m = m.assign(da=m.ra - m.groupby('car').ra.transform('mean'), dd=m.rd - m.groupby('car').rd.transform('mean')); w = np.sqrt(m.na * m.nd)
r = np.average(m.da * m.dd, weights=w) / np.sqrt(np.average(m.da ** 2, weights=w) * np.average(m.dd ** 2, weights=w))
print(f'2. within-carrier month-to-month corr(arrival, delayed-departure copy rate) {r:.3f} over {m.car.nunique()} carriers, {len(m)} carrier-months')
# 3. outage spans (>= 5 consecutive early arrivals copied) and whether departures follow, 2025
A5 = L_arr[L_arr.yr == 2025].sort_values('MVT_TIME_UTC_mvt'); spans = []
for car, g in A5.groupby('car'):
    c = g.cp.values; t = g.MVT_TIME_UTC_mvt.values; i = 0
    while i < len(c):
        if c[i]:
            j = i
            while j + 1 < len(c) and c[j + 1]: j += 1
            if j - i + 1 >= 5: spans.append((car, t[i], t[j]))
            i = j + 1
        else: i += 1
D5 = L_dep[L_dep.yr == 2025]; tt = D5.MVT_TIME_UTC_mvt.dt.tz_localize(None).values; inside = np.zeros(len(D5), bool)
for car, s, e in spans: inside |= (D5.car.values == car) & (tt >= s) & (tt <= e)
cars_sp = set(c for c, _, _ in spans)
print(f'3. 2025 outage spans {len(spans)} over {len(cars_sp)} carriers; departures inside {int(inside.sum()):,} copied {D5.cp.values[inside].mean():.1%}; '
      f'same carriers outside {D5.cp.values[~inside & D5.car.isin(cars_sp).values].mean():.1%}')
per = pd.DataFrame({'car': D5.car.values[inside], 'cp': D5.cp.values[inside]}).groupby('car').cp.agg(['size', 'mean']); per = per[per['size'] >= 10]
fol = per['mean'] >= 0.5
print(f'   carriers with >= 10 departures inside: {len(per)}; follow (>= 50% copied) {int(fol.sum())}, range {per["mean"][fol].min():.0%}-{per["mean"][fol].max():.0%};'
      f' do not {int((~fol).sum())}, range {per["mean"][~fol].min():.0%}-{per["mean"][~fol].max():.0%}')
print('   followers: ' + ' '.join(per.index[fol]) + '   non-followers: ' + ' '.join(per.index[~fol]))
# 4. other airports
print('4. by airport: delayed departures copied 2025, carriers >= 50% copied (>= 30 delayed departures), early arrivals copied 2025 / 2026')
for ap in sorted(dep.ADEP_mvt.unique()):
    x = dep[(dep.ADEP_mvt == ap) & (dep.yr == 2025) & (dep.ms >= 900)]; pc = x.groupby('car').cp.agg(['size', 'mean'])
    y = arr[arr.ADES_mvt == ap]
    print(f'   {ap}  {x.cp.mean():6.1%}  {int(((pc["size"] >= 30) & (pc["mean"] >= 0.5)).sum()):3d}   {y.cp[y.yr == 2025].mean():5.1%} / {y.cp[y.yr == 2026].mean():5.1%}')
