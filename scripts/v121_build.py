"""v121 = v120 with LIRF departures inside 2026 provider outages moved toward the schedule copy (s578, s579).

THE MECHANISM (the user's framing: provider outages, timed). A handler that stops feeding LIRF's system leaves its
carrier's early ARRIVALS with copied in-blocks; the scored file labels 2026 arrivals. An outage span is a run of >= 5
consecutive early arrivals of a carrier, all copied, from the first to the last. Whether the carrier's DEPARTURES follow
is a fixed carrier property, measured on 2025 (both directions labelled): inside their 2025 spans followers copy 91-100%,
non-followers (RYR, UAL, AAL, VLG, MSR, CHH, WMT, CCA, ...) 7-25%.
Rule, fixed before pricing, applied to 2026 LIRF departures inside a 2026 span of their carrier, inside the 2025
inside-span ms support (q01..q99), and only where v120's implied copy weight is below the target:
  a carrier with 2025 spans and >= 10 departures inside them that follows (inside copy rate >= 0.5): its own 2025 rate
  a carrier with no 2025 spans: the class hedge PI * qF + (1 - PI) * qN, PI = the share of 2025 span carriers (>= 10
     departures inside) that follow, qF / qN = the pooled inside copy rates of followers / non-followers
  everything else (non-followers, carriers with too few 2025 rows): untouched
moved to w * clip(MVT - SCHED) + (1 - w) * bulk. CONTROL: FIRE=0 writes v120. Prediction and kill line: s579.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v121_build.py
"""
import glob, hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V121_BASE'] if CLEAN else 'submissions/quick-boat_v120.parquet'
OUT = os.environ.get('V121_OUT', 'submissions/quick-boat_v121.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v121_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA120 = '16e294f1b6a0f52c72e53db9e2115eabb040b6cac5394c5ddf13d25b57d6d507'
SHA121 = '7acc7069a8c3974cec476cb68e97c278d7fd2e473a12fef53bdc46cc3fbd8fa7'   # recorded from the first run
N_RUN, MIN_INSIDE = 5, 10
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA120, 'v120 on disk is not the shipped file'
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
fs = sorted(glob.glob(f'{D}/training_2025-*.parquet')) + [f'{D}/ranking.parquet']
T = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in fs]).drop_duplicates('MVT_ID_mvt').reset_index(drop=True)
for c in ('MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'): T[c] = pd.to_datetime(T[c], utc=True)
T['car'] = T.FLIGHT_mvt.astype(str).str[:3]; T['yr'] = T.MVT_TIME_UTC_mvt.dt.year
T['ms'] = (T.MVT_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds()
T['cp'] = (T.BLOCK_TIME_UTC_mvt - T.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
A = T[T.PHASE_mvt.eq('ARR') & T.ADES_mvt.eq('LIRF') & (T.ms < -600)].sort_values('MVT_TIME_UTC_mvt')
Dp = T[T.PHASE_mvt.eq('DEP') & T.ADEP_mvt.eq('LIRF')]
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
SP = pd.DataFrame(spans, columns=['car', 'start', 'end']); SP['yr'] = pd.to_datetime(SP.start).dt.year
def inside(df, sp):
    m = np.zeros(len(df), bool); t = df.MVT_TIME_UTC_mvt.dt.tz_localize(None).values; cr = df.car.values
    for _, s in sp.iterrows(): m |= (cr == s.car) & (t >= s.start) & (t <= s.end)
    return m
D25 = Dp[Dp.yr == 2025]; in25 = inside(D25, SP[SP.yr == 2025])
per = pd.DataFrame({'car': D25.car.values[in25], 'cp': D25.cp.values[in25]}).groupby('car').cp.agg(['size', 'mean'])
per = per[per['size'] >= MIN_INSIDE]; fol = per['mean'] >= 0.5
PI = float(fol.mean()); I25 = pd.DataFrame({'car': D25.car.values[in25], 'cp': D25.cp.values[in25]})
qF = float(I25.cp[I25.car.isin(per.index[fol])].mean()); qN = float(I25.cp[I25.car.isin(per.index[~fol])].mean())
HW = PI * qF + (1 - PI) * qN; lo, hi = np.quantile(D25.ms.values[in25], [.01, .99])
print(f'  2025 span carriers (>= {MIN_INSIDE} departures inside): {len(per)}, followers {int(fol.sum())} (PI {PI:.3f}); inside copy rate followers {qF:.3f}, non-followers {qN:.3f}')
print(f'  class hedge for carriers without 2025 spans: {HW:.3f}; ms support {lo:.0f}..{hi:.0f}')
seen25 = set(SP.car[SP.yr == 2025])
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
R = Dp[Dp.yr == 2026].set_index('MVT_ID_mvt').reindex(ids)
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
fp = np.load(os.environ['DC26_WF'] if CLEAN else os.environ.get('DC26_SESSION', f'{W}/session') + '/p2P/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
Mr = pd.read_parquet(os.environ['DC26_MEMBERS_RK'] if CLEAN else f'{W}/seeds/chain/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
mon = R.MVT_TIME_UTC_mvt.dt.month.values; ms = R.ms.values
Bk = np.clip(np.rint(Mr[FULL].to_numpy(float) @ wF + (pd.Series(mon).map({m: float(os.environ[f'SHIFT_{int(m)}']) for m in np.unique(mon[np.isfinite(mon)])}).fillna(float(os.environ['SHIFT_7'])).to_numpy(float) if CLEAN else np.where(mon == 1, -8.778, -3.861))), 60, 200000); Ak = np.clip(ms, 60, 200000)
with np.errstate(divide='ignore', invalid='ignore'): p = np.clip((inc - Bk) / (Ak - Bk), 0, 1)
R2 = R.assign(car=R.car.fillna('')).copy(); R2['MVT_TIME_UTC_mvt'] = R2.MVT_TIME_UTC_mvt
in26 = np.zeros(len(ids), bool); ok = R2.MVT_TIME_UTC_mvt.notna().values
in26[ok] = inside(R2[ok], SP[SP.yr == 2026])
tgt = np.full(len(ids), np.nan); cars = R2.car.values
for cr in np.unique(cars[in26]):
    k = in26 & (cars == cr)
    if cr in per.index and fol.get(cr, False): tgt[k] = per.loc[cr, 'mean']
    elif cr not in seen25: tgt[k] = HW
fire = in26 & np.isfinite(tgt) & (ms >= lo) & (ms <= hi) & np.isfinite(p) & (p < tgt)
out = inc.copy()
if FIRE: out[fire] = np.clip(np.rint(tgt[fire] * Ak[fire] + (1 - tgt[fire]) * Bk[fire]), 60, 200000)
d = out - inc; Q = float((d ** 2).sum())
print('  2026 rows fired by carrier: ' + '  '.join(f'{c} {int((fire & (cars == c)).sum())} @ w {np.nanmax(tgt[fire & (cars == c)]):.3f}' for c in np.unique(cars[fire])))
print(f'  rows changed {int((d != 0).sum())}; Q_b {Q:.4e}' + (f'; largest row share {float((d ** 2).max() / Q):.1%}' if Q else ''))
np.save(os.path.join(tempfile.gettempdir(), 'v121_fire.npy'), np.column_stack([fire, np.nan_to_num(tgt)]))
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA120), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v120 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA121:
    assert h == SHA121, f'v121 does not reproduce: {h}'; print('  sha256 matches the recorded v121')
