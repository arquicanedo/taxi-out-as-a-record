"""v122 = v121 with two consequences of the flight-plan join rule, both deterministic (no fitted parameter).

THE RULE (found by an independent agent session from the provided data only, 2026-10-03, and verified here): the extract
joins an airport departure to its Network Manager flight plan only when the recorded off-block lies within 3,606 s of the
plan's last off-block. Over all 2,062,577 join-present 2025 departures BLOCK - LOBT spans exactly [-3606, +3606].
  L1  join-present rows (all airports): y = MVT - BLOCK lies in [MVT - LOBT - 3606, MVT - LOBT + 3606]. Clip v121 onto
      that interval. Guaranteed: moving a prediction onto an interval that contains y removes at least its squared move.
  L2  LIRF join-null rows: the flight's own plan is usually on its ARRIVAL row at the destination when that is one of the
      ten airports (matched on FLIGHT_mvt + ADES, arrival 15 min to 6 h after take-off; on join-present rows the arrival's
      LOBT equals the departure's on 99.88%). Join-null says |BLOCK - LOBT| > 3606. If the flight took off more than 3,606
      s after that LOBT, the plan was never updated for the delay, so a schedule copy (BLOCK = SCHED ~ LOBT) would have
      joined: the record is real. 2025: 76 such rows, 0 copies, median |y - (MVT - AOBT_3)| 124 s. Prediction: the arrival
      row's MVT - AOBT_3 (the network's off-block for the same flight). The other branch (took off within 3,606 s of the
      LOBT) is 21 of 30 copies and is NOT applied here.
Window (Jan|Jul 2025, against v121's window vector): L2's branch removes 4.8724e8 on 21 rows, 0 copies; L1's bound is
violated by 0 of 339,046 window labels. CONTROL: FIRE=0 writes v121 back byte-identically.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v122_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V122_BASE'] if CLEAN else 'submissions/quick-boat_v121.parquet'
OUT = os.environ.get('V122_OUT', 'submissions/quick-boat_v122.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v122_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA121 = '7acc7069a8c3974cec476cb68e97c278d7fd2e473a12fef53bdc46cc3fbd8fa7'
SHA122 = '29cdf104e680abbeda21fc225df6e9cf4269e0c29061e2113fe5a5d7e41bf28e'   # recorded from the first run
TOL, FLOOR, CAP = 3606.0, 60.0, 200000.0
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA121, 'v121 on disk is not the shipped file'
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
        'LOBT_flt', 'AOBT_3_flt', 'TAXITIME_SEC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']])
T = T.drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
E = pd.Timestamp('1970-01-01', tz='UTC')
for c, k in (('MVT_TIME_UTC_mvt', 'mvt'), ('BLOCK_TIME_UTC_mvt', 'blk'), ('SCHED_TIME_UTC_mvt', 'sch'), ('LOBT_flt', 'lobt'), ('AOBT_3_flt', 'aobt')):
    T[k] = (pd.to_datetime(T[c], utc=True) - E).dt.total_seconds()
assert 1.735e9 < T.mvt.median() < 1.80e9, 'timestamp units'
T['yr'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.year
dep, arr = T[T.PHASE_mvt.eq('DEP')], T[T.PHASE_mvt.eq('ARR')]
# ---- the rule on labelled 2025
jp = dep[(dep.yr == 2025) & dep.lobt.notna()]; d = jp.blk - jp.lobt
print(f'  join-present 2025 departures {len(jp):,}: BLOCK - LOBT in [{d.min():.0f}, {d.max():.0f}]')
assert len(jp) == 2062577 and d.min() >= -TOL and d.max() <= TOL, 'the join rule changed'
# ---- L2's match and its labelled 2025 truth table
L = dep[dep.ADEP_mvt.eq('LIRF') & dep.FLIGHT_mvt.notna()][['MVT_ID_mvt', 'FLIGHT_mvt', 'ADES_mvt', 'mvt', 'sch', 'lobt', 'yr', 'TAXITIME_SEC_mvt']]
A = arr[['FLIGHT_mvt', 'ADES_mvt', 'mvt', 'lobt', 'aobt']].rename(columns={'mvt': 'a_mvt', 'lobt': 'a_lobt', 'aobt': 'a_aobt'})
M = L.merge(A, on=['FLIGHT_mvt', 'ADES_mvt'], how='inner'); M = M[(M.a_mvt - M.mvt).between(900, 21600)]
M = M.sort_values('a_mvt').drop_duplicates('MVT_ID_mvt')
c = M[M.lobt.notna() & M.a_lobt.notna()]; agree = float(np.mean(np.abs(c.a_lobt - c.lobt) < 1))
far = M.lobt.isna() & M.a_lobt.notna() & M.a_aobt.notna() & (M.mvt - M.a_lobt > TOL)
f25 = M[far & (M.yr == 2025)]; y25 = f25.TAXITIME_SEC_mvt.astype(float); cp25 = (y25 - (f25.mvt - f25.sch)).abs() <= 60
print(f'  arrival LOBT == departure LOBT on {agree:.2%} of {len(c):,} join-present matches; 2025 L2 rows {len(f25)}, copies {int(cp25.sum())}, '
      f'median |y - (MVT - AOBT_3)| {np.median(np.abs(y25 - (f25.mvt - f25.a_aobt))):.0f} s')
assert len(f25) == 76 and int(cp25.sum()) == 0, 'the L2 truth table changed'
# ---- 2026
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
R = dep[dep.yr == 2026].set_index('MVT_ID_mvt').reindex(ids)
out = inc.copy()
lo = np.maximum(R.mvt.values - R.lobt.values - TOL, FLOOR); hi = R.mvt.values - R.lobt.values + TOL
has = np.isfinite(lo) & np.isfinite(hi)
l1 = has & ((out < lo) | (out > hi))
if FIRE: out[l1] = np.clip(np.rint(np.clip(out[l1], lo[l1], hi[l1])), FLOOR, CAP)
F6 = M[far & (M.yr == 2026)].set_index('MVT_ID_mvt')
l2 = pd.Index(ids).isin(F6.index)
if FIRE:
    f = F6.reindex(ids[l2]); out[l2] = np.clip(np.rint(f.mvt.values - f.a_aobt.values), FLOOR, CAP)
d = out - inc; Q = float((d ** 2).sum())
print(f'  2026: L1 rows {int(l1.sum())} (by airport: ' + ', '.join(f'{a} {n}' for a, n in pd.Series(R.ADEP_mvt.values[l1]).value_counts().items())
      + f'); L2 rows {int(l2.sum())}; rows changed {int((d != 0).sum())}; Q_b {Q:.4e}')
if FIRE:
    q1 = float((d[l1] ** 2).sum()); q2 = float((d[l2] ** 2).sum()); N = len(ids); s0 = N * 225.9576 ** 2
    print(f'  L1 guaranteed floor {q1:.4e}; L2 moved {q2:.4e}; if both pay their Q: {np.sqrt((s0 - q1 - q2) / N):.4f}; '
          f'L1 alone: {np.sqrt((s0 - q1) / N):.4f}')
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA121), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v121 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA122:
    assert h == SHA122, f'v122 does not reproduce: {h}'; print('  sha256 matches the recorded v122')
