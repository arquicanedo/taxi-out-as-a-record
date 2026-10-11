"""The clean base model against the submitted one on the 2026 set, for the paper (v2, section 5.1). Both predict the
same 344,841 rows; the clean base is scripts/clean_base_build.py's output, the submitted base is submissions/
quick-boat_v100.parquet. Reports agreement within 60 s, rows moved more than 600 s off each model's own bulk (the
nine-member blend plus that model's month offsets), and where the squared difference between the two sits: on rows where the clean base's mixture weights the date branch
(and whether schedule and take-off cross the Rome local date, the branch's necessary condition), and on rows it weights
only toward the schedule copy. Also the scores both pairs returned on the 2026 set (s560), for the stage's squared error.
  CLEAN_DIR=$DC26_WORK/clean uv run --quiet --with pandas --with pyarrow --with numpy python scripts/paper_clean_vs_submitted.py
"""
import json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
C = os.environ.get('CLEAN_DIR', f'{W}/clean'); P2 = os.environ.get('DC26_SESSION', f'{W}/session') + '/p2P'
ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
rd = lambda p: pq.read_table(p).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(ids).values.astype(float)
sub, cln = rd('submissions/quick-boat_v100.parquet'), rd(f'{C}/submissions/quick-boat_v100.parquet')
fp = np.load(f'{P2}/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
MR = pd.read_parquet(f'{W}/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
mon = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True).dt.month.values; apt = R.ADEP_mvt.values.astype(str)
off = json.load(open(f'{C}/offsets.json'))
B = np.stack([MR[k].values.astype(float) for k in FULL], 1) @ wF
bulk_c = I.cl(np.rint(B + np.where(mon == 1, off['SHIFT_1'], off['SHIFT_7'])))
bulk_s = I.cl(np.rint(B + np.where(mon == 1, -8.778, -3.861)))
d = cln - sub; Q = float((d ** 2).sum()); big = np.abs(d) > 600
print(f'rows {len(ids):,}; agree within 60 s {float((np.abs(d) <= 60).mean()):.2%}; identical {float((d == 0).mean()):.2%}')
print(f'moved more than 600 s off their own bulk: submitted {int((np.abs(sub - bulk_s) > 600).sum())}, clean {int((np.abs(cln - bulk_c) > 600).sum())}')
print(f'squared difference {Q:.4e}; rows differing by more than 600 s {int(big.sum())}, carrying {float((d[big] ** 2).sum()) / Q:.1%}; '
      f'of those at LIRF {int((big & (apt == "LIRF")).sum())}, carrying {float((d[big & (apt == "LIRF")] ** 2).sum()) / Q:.1%}')
G = pd.read_parquet(f'{W}/b4_gate_rk.parquet').set_index('MVT_ID_mvt').reindex(ids); P = G[[f'P{i}' for i in range(5)]].to_numpy(float)
Qm = P ** 0.90; Qm = Qm / Qm.sum(1, keepdims=True); Qm[:, 1:][Qm[:, 1:] < 0.10] = 0.0
loc = lambda x: pd.to_datetime(x, utc=True).dt.tz_convert('Europe/Rome').dt.date.values
cross = loc(R.SCHED_TIME_UTC_mvt) != loc(R.MVT_TIME_UTC_mvt)
m3 = Qm[:, 3] > 0; m1 = (Qm[:, 1] > 0) & ~m3; sh = lambda m: float((d[m] ** 2).sum()) / Q
print(f'date branch weighted: {int(m3.sum())} rows ({dict(pd.Series(apt[m3]).value_counts())}), {sh(m3):.1%} of the squared difference; '
      f'not crossing the Rome local date {int((m3 & ~cross).sum())}, {sh(m3 & ~cross):.1%}')
print(f'copy branch only: {int(m1.sum())} rows, {int((m1 & (apt == "LIRF")).sum())} at LIRF, {sh(m1):.1%}')
print(f'within the {int(big.sum())} rows differing by more than 600 s: date branch weighted {int((big & m3).sum())} ({sh(big & m3):.1%}), '
      f'not crossing {int((big & m3 & ~cross).sum())} ({sh(big & m3 & ~cross):.1%}); copy branch only {int((big & m1).sum())}, '
      f'{int((big & m1 & (apt == "LIRF")).sum())} at LIRF ({sh(big & m1):.1%}); neither {int((big & ~m3 & ~m1).sum())} ({sh(big & ~m3 & ~m1):.1%})')
N = len(ids); sc = {'submitted': (254.5228, 226.8129), 'clean': (292.0485, 268.1373)}   # returned by the 2026 set (s560)
for k, (b, f) in sc.items():
    print(f'{k}: base {b} final {f}  stage {b - f:.4f} s, squared error removed {N * (b * b - f * f):.4e}')
dS = lambda i: N * (sc['clean'][i] ** 2 - sc['submitted'][i] ** 2)
print(f'clean - submitted: base {sc["clean"][0] - sc["submitted"][0]:+.4f} s (dSSE {dS(0):+.4e}, Q {Q:.4e}, cross term 2T {dS(0) - Q:+.4e}), '
      f'final {sc["clean"][1] - sc["submitted"][1]:+.4f} s (dSSE {dS(1):+.4e})')
