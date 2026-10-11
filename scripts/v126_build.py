"""v126 = v125 with a floor on Fiumicino's day-scale departures without a flight plan (the second agent session's lever
F, restricted to clean identifiers; s595).

THE RULE, from labelled 2025: a LIRF departure without a flight-plan join delayed 30,000 to 86,000 s is never a real
off-block record: in 2025 every such row was a schedule copy (y within 60 s of ms) or a date error (y above ms), and no
row had y below ms - 60. Between 30,000 and 46,800 s no date error occurs either, so the record is the copy.
  30000 <= ms < 46800   prediction = clip(ms)
  46800 <= ms < 86000   prediction = max(prediction, clip(ms))
CHAR(8) identifiers (the airline designator's tail cycled to pad the flight number to eight characters, the v54 cell)
are LEFT OUT: their 2025 copy rate did not carry to 2026 (the board's optimum toward the copy for that cell was -0.08),
and one of them, RYRR388G, would carry 82% of the move.
CONTROL: FIRE=0 writes v125 back byte-identically.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v126_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V126_BASE'] if CLEAN else 'submissions/quick-boat_v125.parquet'
OUT = os.environ.get('V126_OUT', 'submissions/quick-boat_v126.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v126_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA125 = '64eb3556d3fa7c93c8b7f8b891b6f71bc800021e681fc01901cd841cd8ab9612'
SHA126 = 'd66768fe3504ddcc6ed0bbf9a0e97bfff5d9e353e73bc4e4a4b170a079b6a951'   # recorded from the first run
LO, MID, HI, FLOOR, CAP = 30000.0, 46800.0, 86000.0, 60.0, 200000.0
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA125, 'v125 on disk is not the shipped file'
char8 = lambda s: s.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d') | s.astype(str).str.match(r'^[A-Z]{2}([A-Z])\1')
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'LOBT_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']])
T = T[T.PHASE_mvt.eq('DEP') & T.ADEP_mvt.eq('LIRF') & T.LOBT_flt.isna()].drop_duplicates('MVT_ID_mvt').copy()
T['ms'] = (pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(T.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
T['yr'] = pd.to_datetime(T.MVT_TIME_UTC_mvt, utc=True).dt.year; T['c8'] = char8(T.FLIGHT_mvt).values
b = T[(T.yr == 2025) & (T.ms >= LO) & (T.ms < HI)]; y = b.TAXITIME_SEC_mvt.astype(float)
below = int((y < b.ms - 60).sum()); cp = int(((y - b.ms).abs() <= 60).sum()); lowroll = int(((y > b.ms + 60) & (b.ms < MID)).sum())
print(f'  2025 LIRF no-plan rows with {LO:.0f} <= ms < {HI:.0f}: {len(b)}, copies {cp}, above ms {len(b) - cp - below}, below ms {below}; '
      f'above ms with ms < {MID:.0f}: {lowroll}')
assert below == 0 and lowroll == 0, 'the labelled support changed'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
bb = pq.read_table(BASE).to_pandas(); assert (bb.MVT_ID_mvt.values == ids).all(); inc = bb.TAXITIME_SEC_mvt.values.astype(float)
R = T[T.yr == 2026].set_index('MVT_ID_mvt').reindex(ids)
ms = R.ms.values; c8 = R.c8.fillna(True).values.astype(bool)
cell = np.isfinite(ms) & (ms >= LO) & (ms < HI) & ~c8
out = inc.copy()
if FIRE:
    lo = cell & (ms < MID); hi = cell & (ms >= MID)
    out[lo] = np.clip(np.rint(ms[lo]), FLOOR, CAP)
    out[hi] = np.clip(np.maximum(out[hi], np.rint(ms[hi])), FLOOR, CAP)
d = out - inc; Q = float((d ** 2).sum()); N = len(ids); s0 = N * 222.9161 ** 2
print(f'  2026 clean-identifier rows in the band {int(cell.sum())}; rows changed {int((d != 0).sum())}; Q_b {Q:.4e}; '
      f'CHAR(8) rows left out {int((np.isfinite(ms) & (ms >= LO) & (ms < HI) & c8).sum())}')
if FIRE: print(f'  if every moved row is a copy (gain ~ Q): {np.sqrt((s0 - Q) / N):.4f}')
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA125), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v125 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA126:
    assert h == SHA126, f'v126 does not reproduce: {h}'; print('  sha256 matches the recorded v126')
