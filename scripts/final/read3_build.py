"""Read 3: the four-month not-fitted model's own January and July rows, as a leaderboard file, and its pre-registration.

  uv run --with pandas --with numpy --with pyarrow python scripts/final/read3_build.py \
      --uniform <DC26_WORK>/quick-boat_final_uniform.parquet --old-data <2026-09-04 extract> --final-data <final view> \
      --out submissions/quick-boat_v130.parquet

The leaderboard scores January and July against the 2026-09-04 extract's labels, so the file takes the uniform model's
predictions for those 344,841 departures in that extract's submitting order and schema.

The expected score is v127's 247.7412 plus each group's expected change in squared error, computed from labelled rates
only, before any score is seen:
  day-scale rows   the LIRF bands of scripts/final/lirf_dayscale.py; each row's truth is a schedule copy, a date error
                   (86400 nd + 1,080) or an ordinary taxi (taken at the model's base) with the band's 2025 rates; the
                   expectation and the variance of dSSE follow, and the variance sets the band (rows independent)
  v96 rows         the eleven carriers: the copy with probability 0.9911, else an ordinary taxi of 1,013 s
  every other row  expected change 0: the base differs from v127's by re-drawn members and the re-jittered inputs; the
                   window cannot price it, so it is reported as its Q and left out of the expectation
"""
import argparse, hashlib, glob
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
for k in ('--uniform', '--old-data', '--final-data', '--out'): ap.add_argument(k, required=True)
a = ap.parse_args()
FLOOR, CAP, DAY, TAXI_ROLL, N, S127 = 60, 200000, 86400, 1080, 344841, 247.7412

tpl = pq.read_table(f'{a.old_data}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values; assert len(ids) == N
U = pq.read_table(a.uniform).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
u = U.reindex(ids).to_numpy(float); assert np.isfinite(u).all(), 'the uniform file must cover every January and July departure'
v = pq.read_table('submissions/quick-boat_v127.parquet').to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(ids).to_numpy(float)
t = pa.Table.from_pandas(pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': u.astype(tpl.schema.field('TAXITIME_SEC_mvt').type.to_pandas_dtype())}),
                         schema=tpl.schema, preserve_index=False)
pq.write_table(t, a.out); sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
d = u - v
print(f'wrote {a.out}  sha256 {sha}\nagainst v127: rows differing {int((d != 0).sum()):,}, RMS {np.sqrt(np.mean(d ** 2)):.2f} s, '
      f'>600 s {int((np.abs(d) > 600).sum())}, Q {float((d ** 2).sum()):.4e}')

# the populations, on the extract the leaderboard scores (labels of the scored rows are never read: they are blank)
C = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']
R = pq.read_table(f'{a.old_data}/ranking.parquet', columns=C).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(ids)
for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'): R[c] = pd.to_datetime(R[c], utc=True)
ms = (R.MVT_TIME_UTC_mvt - R.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
lirf = R.ADEP_mvt.eq('LIRF').to_numpy(); nolobt = R.LOBT_flt.isna().to_numpy()
loc = lambda s: s.dt.tz_convert('Europe/Rome').dt.normalize()
nd = ((loc(R.MVT_TIME_UTC_mvt) - loc(R.SCHED_TIME_UTC_mvt)).dt.days).to_numpy()
copy = np.clip(ms, FLOOR, CAP)
DS = lirf & nolobt & (ms >= 20000)
band = np.where(ms < 50000, 'A', np.where(nd >= 1, 'B', 'C'))
RATES = {'A': (48 / 50, 0.0, 2 / 50), 'B': (8 / 19, 11 / 19, 0.0), 'C': (49 / 51, 0.0, 2 / 51)}   # copy, date error, ordinary (2025)
E = V = 0.0
for i in np.where(DS)[0]:
    pc, pr, po = RATES[band[i]]; roll = np.clip(DAY * max(nd[i], 1) + TAXI_ROLL, FLOOR, CAP)
    outs = [(pc, copy[i]), (pr, roll), (po, v[i])]
    g = np.array([(u[i] - y) ** 2 - (v[i] - y) ** 2 for _, y in outs]); w = np.array([p for p, _ in outs])
    e = float((w * g).sum()); E += e; V += float((w * (g - e) ** 2).sum())
car = R.FLIGHT_mvt.astype(str).str[:3].to_numpy()
HOT = ['ANE', 'ENT', 'FIE', 'HAT', 'ISR', 'LAV', 'QFA', 'TGZ', 'TKJ', 'TWB', 'UPS']
S6 = lirf & (ms > 600) & np.isin(car, HOT) & ~DS
g6 = 0.9911 * ((u[S6] - copy[S6]) ** 2 - (v[S6] - copy[S6]) ** 2) + 0.0089 * ((u[S6] - 1013) ** 2 - (v[S6] - 1013) ** 2)
E6 = float(g6.sum())
rest = ~DS & ~S6
print(f'\nday-scale rows {int(DS.sum())} (A {int((DS & (band == "A")).sum())}, B {int((DS & (band == "B")).sum())}, C {int((DS & (band == "C")).sum())}): '
      f'expected dSSE {E:+.4e}, SD {np.sqrt(V):.4e}')
print(f'v96 rows {int(S6.sum())}: expected dSSE {E6:+.4e}')
print(f'every other row: Q {float((d[rest] ** 2).sum()):.4e} over {int((d[rest] != 0).sum()):,} rows, expectation 0 (not priced)')
SSE = N * S127 ** 2; tot = E + E6; sd = np.sqrt(V)
f = lambda x: np.sqrt((SSE + x) / N)
print(f'\nPREDICTION  {f(tot):.4f}   +-2 SD (day-scale only) [{f(tot - 2 * sd):.4f}, {f(tot + 2 * sd):.4f}]   '
      f'+-3 SD [{f(tot - 3 * sd):.4f}, {f(tot + 3 * sd):.4f}]')
