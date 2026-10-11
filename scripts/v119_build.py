"""v119 = v118 with Fiumicino's never-joining, normal-looking identifiers hedged toward the schedule copy (s572).

THE RULE, from labelled 2025 data only. A LIRF departure with no flight-plan join (LOBT null), delayed at least an hour
(ms >= 3600), whose FLIGHT_mvt never joins a flight plan on any LIRF departure in 2025 or 2026, and which is not a
doubled/garbled designator (the v54 cell, whose 2025 rate did not carry to 2026), copied the schedule on 83 of 87 rows in
2025 (95.4%), joins counted over 2025 and 2026 alike. Its non-copies were ordinary taxis (529-1,261 s) and one long one, no date errors.
On such 2026 rows inside the 2025 support of ms (q01..q99), where v118's implied copy weight is below the rate, the
prediction moves to rate * clip(MVT - SCHED) + (1 - rate) * bulk, the bulk being v118's nine-member blend (seeds chain).
Evidence class: a labelled prior, counted on the predicate's own rows (presubmit_guard.py --evidence labelled-prior).
PREDICTED 226.02 (expected gain 5.60e7 SSE at the 2025 rate); KILL at or above 226.3833. CONTROL: FIRE=0 writes v118.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v119_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
BASE = 'submissions/quick-boat_v118.parquet'
OUT = os.environ.get('V119_OUT', 'submissions/quick-boat_v119.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v119_FIRE0.parquet'))
SHA118 = '2f2079f70ba9526ac7ee93f131711e64ece209266f4d6cf6378b364e365428de'
SHA119 = '20aa2fd24dc82009518460b572d4f6f69aae8a29f08a59598b4c3196dc014f4a'   # recorded from the first run
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert sha(BASE) == SHA118, 'v118 on disk is not the shipped file'
garb = lambda s: s.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d') | s.astype(str).str.match(r'^[A-Z]{2}([A-Z])\1')
# ---- 2025: the predicate's labelled rate and ms support
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'LOBT_flt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
L = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))]).drop_duplicates('MVT_ID_mvt')
L = L[L.PHASE_mvt.eq('DEP') & L.ADEP_mvt.eq('LIRF')].copy()
L['ms'] = (pd.to_datetime(L.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(L.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
R = pq.read_table(f'{D}/ranking.parquet', columns=cols[:-1]).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
known = set(L.FLIGHT_mvt[L.LOBT_flt.notna()].astype(str)) | set(R.FLIGHT_mvt[(R.ADEP_mvt == 'LIRF') & R.LOBT_flt.notna()].astype(str))
pr = L[L.LOBT_flt.isna() & (L.ms >= 3600) & ~L.FLIGHT_mvt.astype(str).isin(known) & ~garb(L.FLIGHT_mvt)]
cp = np.abs(pr.TAXITIME_SEC_mvt.astype(float) - pr.ms) <= 60; RATE = float(cp.mean()); lo, hi = np.quantile(pr.ms, [.01, .99])
print(f'  2025 predicate: {len(pr)} rows, {int(cp.sum())} copies, rate {RATE:.4f}; ms support {lo:.0f}..{hi:.0f}')
assert len(pr) == 87 and int(cp.sum()) == 83, 'the labelled predicate changed'
# ---- 2026: the rows, the copy and bulk branches, v118's implied weight
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
mvt = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True); ms = (mvt - pd.to_datetime(R.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().values
fp = np.load(os.environ.get('DC26_SESSION', f'{W}/session') + '/p2P/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
Mr = pd.read_parquet(f'{W}/seeds/chain/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
B = np.clip(np.rint(Mr[FULL].to_numpy(float) @ wF + np.where(mvt.dt.month.values == 1, -8.778, -3.861)), 60, 200000); A = np.clip(ms, 60, 200000)
with np.errstate(divide='ignore', invalid='ignore'): p = np.clip((inc - B) / (A - B), 0, 1)
fire = ((R.ADEP_mvt.values == 'LIRF') & R.LOBT_flt.isna().values & (ms >= 3600) & ~R.FLIGHT_mvt.astype(str).isin(known).values
        & ~garb(R.FLIGHT_mvt).values & (ms >= lo) & (ms <= hi) & np.isfinite(p) & (p < RATE))
out = inc.copy()
if FIRE: out[fire] = np.clip(np.rint(RATE * A[fire] + (1 - RATE) * B[fire]), 60, 200000)
d = out - inc; Q = float((d ** 2).sum())
print(f'  2026 rows fired {int(fire.sum())}; rows changed {int((d != 0).sum())}; Q_b {Q:.4e}' + (f'; largest row share {float((d**2).max()/Q):.1%}' if Q else ''))
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == SHA118, 'FIRE=0 must reproduce v118'; print('  CONTROL: FIRE=0 reproduces v118 byte-identically')
elif SHA119:
    assert h == SHA119, f'v119 does not reproduce: {h}'; print('  sha256 matches the recorded v119')
