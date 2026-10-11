"""v120 = v118 with Turkish Airlines' January 2026 departures at Fiumicino moved to the schedule copy (s574).

THE MECHANISM. THY's handler stopped feeding LIRF's off-block system in late 2025: its departures copied the schedule
on 5-16% of rows January-September 2025, 56.5% in October, then 241 of 243 in November and December (99.2%, both
months 99.2%; the two others were 603-604 s taxis). Its early ARRIVALS switched with it (2-15% -> 46% -> 100%, 100%).
The scored file labels 2026 arrivals: THY's January 2026 early arrivals are 38 of 38 copied, so the regime persists.
v118, trained on all of 2025, gives these rows a copy weight of zero. Rule, fixed before pricing: THY departures at
LIRF in a 2026 month whose early arrivals are >= 90% copied (>= 10 of them), inside the Nov-Dec 2025 support of ms
(q01..q99), move to rate * clip(MVT - SCHED) + (1 - rate) * bulk, rate = the Nov-Dec 2025 departure copy rate.
Evidence class: a labelled prior counted on the predicate's own 2025 rows (presubmit_guard.py --evidence labelled-prior).
PREDICTED 225.97 (expected gain 6.44e7 SSE); KILL at or above 226.3833. CONTROL: FIRE=0 writes v118.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v120_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V120_BASE'] if CLEAN else 'submissions/quick-boat_v118.parquet'
OUT = os.environ.get('V120_OUT', 'submissions/quick-boat_v120.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v120_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA118 = '2f2079f70ba9526ac7ee93f131711e64ece209266f4d6cf6378b364e365428de'
SHA120 = '16e294f1b6a0f52c72e53db9e2115eabb040b6cac5394c5ddf13d25b57d6d507'   # recorded from the first run
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA118, 'v118 on disk is not the shipped file'
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
def load(fs):
    T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in fs]).drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
    for c in ('MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'): T[c] = pd.to_datetime(T[c], utc=True)
    T['mon'] = T.MVT_TIME_UTC_mvt.dt.month; T['ms'] = (T.MVT_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds()
    T['thy'] = T.FLIGHT_mvt.astype(str).str.startswith('THY'); return T
cp = lambda T: (T.BLOCK_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
# ---- 2025: the regime's departure copy rate and ms support (labelled)
T = load(sorted(glob.glob(f'{D}/training_2025-*.parquet')))
reg = T[T.PHASE_mvt.eq('DEP') & T.ADEP_mvt.eq('LIRF') & T.thy & T.mon.isin([11, 12])]
RATE = float(cp(reg).mean()); lo, hi = np.quantile(reg.ms, [.01, .99])
print(f'  2025 THY Nov+Dec departures: {len(reg)}, copies {int(cp(reg).sum())}, rate {RATE:.4f}; ms support {lo:.0f}..{hi:.0f}')
assert len(reg) == 243 and int(cp(reg).sum()) == 241, 'the labelled regime changed'
# ---- 2026: the regime gate from the labelled arrivals, month by month
R = load([f'{D}/ranking.parquet'])
arr = R[R.PHASE_mvt.eq('ARR') & R.ADES_mvt.eq('LIRF') & R.thy & ((R.SCHED_TIME_UTC_mvt - R.MVT_TIME_UTC_mvt).dt.total_seconds() > 600)]
g = arr.groupby('mon').apply(lambda x: pd.Series({'n': len(x), 'rate': float(cp(x).mean())}))
hot = [int(m) for m, r in g.iterrows() if r.n >= 10 and r.rate >= 0.9]
print('  2026 THY early arrivals copied by month: ' + '  '.join(f'm{int(m)} {int(r.n)} at {r.rate:.0%}' for m, r in g.iterrows()) + f'  -> regime months {hot}')
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
Rd = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(ids)
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
fp = np.load(os.environ['DC26_WF'] if CLEAN else os.environ.get('DC26_SESSION', f'{W}/session') + '/p2P/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
Mr = pd.read_parquet(os.environ['DC26_MEMBERS_RK'] if CLEAN else f'{W}/seeds/chain/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
mon = Rd.mon.values; ms = Rd.ms.values
B = np.clip(np.rint(Mr[FULL].to_numpy(float) @ wF + (pd.Series(mon).map({m: float(os.environ[f'SHIFT_{int(m)}']) for m in np.unique(mon[np.isfinite(mon)])}).fillna(float(os.environ['SHIFT_7'])).to_numpy(float) if CLEAN else np.where(mon == 1, -8.778, -3.861))), 60, 200000); A = np.clip(ms, 60, 200000)
fire = (Rd.ADEP_mvt.values == 'LIRF') & Rd.thy.values.astype(bool) & np.isin(mon, hot) & (ms >= lo) & (ms <= hi)
dropped = (Rd.ADEP_mvt.values == 'LIRF') & Rd.thy.values.astype(bool) & np.isin(mon, hot) & ~fire
out = inc.copy()
if FIRE: out[fire] = np.clip(np.rint(RATE * A[fire] + (1 - RATE) * B[fire]), 60, 200000)
d = out - inc; Q = float((d ** 2).sum())
print(f'  2026 rows fired {int(fire.sum())} (outside the support, left alone: {int(dropped.sum())}); changed {int((d != 0).sum())}; Q_b {Q:.4e}'
      + (f'; largest row share {float((d ** 2).max() / Q):.1%}' if Q else ''))
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA118), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v118 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA120:
    assert h == SHA120, f'v120 does not reproduce: {h}'; print('  sha256 matches the recorded v120')
