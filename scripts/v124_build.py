"""v124 = v123 with the other branch of the flight-plan join rule at Fiumicino (s586, s590).

A LIRF departure with no flight-plan join whose own plan is recovered from its arrival row at the destination (as in
v122) and which took off WITHIN 3,606 s of that plan's LOBT: join-null means |BLOCK - LOBT| > 3606; BLOCK above LOBT +
3606 would put the off-block after the take-off, so BLOCK < LOBT - 3606 and y > MVT - LOBT + 3606. Two consequences:
  floor  every such row's prediction is raised to MVT - LOBT + 3606 if it sits below (deterministic: y is above it)
  hedge  where ms exceeds that floor, a schedule copy is possible: 21 of 23 such 2025 rows were copies (the other two a
         date error and a real 4,868 s), so the prediction moves to r * clip(ms) + (1 - r) * floor-clipped v123,
         r = 21/23 counted on 2025 (asserted), not fitted to a score
Window (v121's vector): the hedge at the leave-month-out rate removed 5.2551e7 on 6 rows, 5 copies.
CONTROL: FIRE=0 writes v123 back byte-identically.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v124_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V124_BASE'] if CLEAN else 'submissions/quick-boat_v123.parquet'
OUT = os.environ.get('V124_OUT', 'submissions/quick-boat_v124.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v124_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA123 = '9f2f9f4bf8e6948d241ab53218da55b839f409179b39630c9ba51d252d0ece21'
SHA124 = 'a99fef2878dab2d2c457bf96f1a597cc8f6705c4bde35ddeae7ddb45f1ad70fd'   # recorded from the first run
TOL, FLOOR, CAP = 3606.0, 60.0, 200000.0
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA123, 'v123 on disk is not the shipped file'
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'TAXITIME_SEC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']])
T = T.drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
E = pd.Timestamp('1970-01-01', tz='UTC')
for c, k in (('MVT_TIME_UTC_mvt', 'mvt'), ('SCHED_TIME_UTC_mvt', 'sch'), ('LOBT_flt', 'lobt')):
    T[k] = (pd.to_datetime(T[c], utc=True) - E).dt.total_seconds()
assert 1.735e9 < T.mvt.median() < 1.80e9, 'timestamp units'
T['yr'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.year
dep, arr = T[T.PHASE_mvt.eq('DEP')], T[T.PHASE_mvt.eq('ARR')]
L = dep[dep.ADEP_mvt.eq('LIRF') & dep.FLIGHT_mvt.notna()][['MVT_ID_mvt', 'FLIGHT_mvt', 'ADES_mvt', 'mvt', 'sch', 'lobt', 'yr', 'TAXITIME_SEC_mvt']]
A = arr[['FLIGHT_mvt', 'ADES_mvt', 'mvt', 'lobt']].rename(columns={'mvt': 'a_mvt', 'lobt': 'a_lobt'})
M = L.merge(A, on=['FLIGHT_mvt', 'ADES_mvt'], how='inner'); M = M[(M.a_mvt - M.mvt).between(900, 21600)]
M = M.sort_values('a_mvt').drop_duplicates('MVT_ID_mvt')
M['ml'] = M.mvt - M.a_lobt; M['ms'] = M.mvt - M.sch; M['fl'] = M.ml + TOL
near = M.lobt.isna() & M.a_lobt.notna() & (M.ml <= TOL)
n25 = M[near & (M.yr == 2025)]; y = n25.TAXITIME_SEC_mvt.astype(float)
viol = int((y <= n25.fl).sum()); hc = n25.ms > n25.fl; cp = (y - n25.ms).abs() <= 60
r = float(cp[hc].mean())
print(f'  2025 near rows {len(n25)}: y below the floor on {viol}; with ms above the floor {int(hc.sum())}, copies {int(cp[hc].sum())} (r {r:.4f})')
assert viol == 0 and int(hc.sum()) == 23 and int(cp[hc].sum()) == 21, 'the labelled branch changed'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
n26 = M[near & (M.yr == 2026)].set_index('MVT_ID_mvt').reindex(ids)
m = n26.ml.notna().values
out = inc.copy()
fl = n26.fl.values; ms = np.clip(n26.ms.values, FLOOR, CAP)
up = m & (out < fl)
if FIRE: out[up] = np.clip(np.rint(fl[up]), FLOOR, CAP)
qf = float(((out - inc) ** 2).sum())
hd = m & (n26.ms.values > fl)
if FIRE: out[hd] = np.clip(np.rint(r * ms[hd] + (1 - r) * out[hd]), FLOOR, CAP)
d = out - inc; Q = float((d ** 2).sum()); N = len(ids); s0 = N * 223.5285 ** 2
print(f'  2026 near rows {int(m.sum())}; raised to the floor {int(up.sum())} (guaranteed floor {qf:.4e}); hedged {int(hd.sum())}; '
      f'rows changed {int((d != 0).sum())}; Q_b {Q:.4e}')
if FIRE:
    print(f'  floor alone: {np.sqrt((s0 - qf) / N):.4f}; gain = Q: {np.sqrt((s0 - Q) / N):.4f}')
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA123), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v123 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA124:
    assert h == SHA124, f'v124 does not reproduce: {h}'; print('  sha256 matches the recorded v124')
