"""p2p step 5 (the ship rule passed): save the RANKING-side member predictions of the FULL package to
$DC26_WORK/p2_members_rk.parquet keyed by MVT_ID_mvt (submitting.parquet order), and print the FULL weights at
full float64 precision for scripts/v100_build.py.

Columns (all float64, unclipped, exactly the arrays scripts/p2p_price.py and scripts/p2p_checks.py priced):
  L55, P55   v55_trees.parquet lgbm_opdi / perapt_opdi (shipped)       K32, K128  d2_window_rk(_K128).parquet (shipped)
  ID, RQ     p2p_arm.py MODE=rank, seeds 0,1 (mean)                    CL         p2p_clean.py MODE=rank, seeds 0,1 (mean)
  L55o, P55o p2p_v55.py MODE=rank TE=oof, seeds 0,1 (mean)
CHECK: rebuilding v100 from the saved file reproduces the vector p2p_checks.py saved (full_package.npz) on every row.
"""
import os, numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
ARCH = os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P'   # the archived package: full_package.npz (weights and the priced vector)
P2 = os.environ.get('P2P_OUT', ARCH)   # where the member arrays are read from
MOUT = os.environ.get('P2M_OUT', os.path.join(os.environ['DC26_WORK'], 'p2_members_rk.parquet'))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mean_of = lambda fs: np.mean([np.load(f) for f in fs], 0)
rk_ids = pd.read_parquet(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.values
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
r2s = pd.Series(np.arange(len(rk_ids)), index=rk_ids).reindex(sub_ids).values.astype(int); assert (rk_ids[r2s] == sub_ids).all()
rkc = lambda f, c: pd.read_parquet(f'{W}/{f}').set_index('MVT_ID_mvt').reindex(sub_ids)[c].values.astype(float)
MR = {'L55': rkc('v55_trees.parquet', 'lgbm_opdi'), 'P55': rkc('v55_trees.parquet', 'perapt_opdi'),
      'K32': rkc('d2_window_rk.parquet', 'd2_window'), 'K128': rkc('d2_window_rk_K128.parquet', 'd2_window'),
      'ID': mean_of([f'{P2}/arms/rank_idcat_s{s}.npy' for s in (0, 1)])[r2s], 'RQ': mean_of([f'{P2}/arms/rank_rwyq_s{s}.npy' for s in (0, 1)])[r2s],
      'CL': mean_of([f'{P2}/clean/rank_clean_s{s}.npy' for s in (0, 1)]),
      'L55o': mean_of([f'{P2}/trees/rank_oof_pooled_s{s}.npy' for s in (0, 1)])[r2s], 'P55o': mean_of([f'{P2}/trees/rank_oof_perapt_s{s}.npy' for s in (0, 1)])[r2s]}
z = np.load(f'{ARCH}/full_package.npz', allow_pickle=True); FULL = [str(k) for k in z['members']]; w = z['w']
assert FULL == list(MR), FULL
for k, v in MR.items(): assert np.isfinite(v).all(), k
df = pd.DataFrame({'MVT_ID_mvt': sub_ids, **MR})
df.to_parquet(MOUT, index=False)
print(f'wrote {MOUT}  rows {len(df):,}  columns {list(df.columns)}')
print(f'MEMBERS = {FULL!r}')
print(f'W_NEW = [{", ".join(repr(float(x)) for x in w)}]   (sum {w.sum():.17g})')
# check: rebuild from the saved file exactly as v100_build.py will
M = pd.read_parquet(MOUT).set_index('MVT_ID_mvt').reindex(sub_ids)
Mr = np.ascontiguousarray(M[FULL].to_numpy(float)); M0r = np.ascontiguousarray(M[['L55', 'P55', 'K32', 'K128']].to_numpy(float))
delta = Mr @ w - M0r @ np.array([.2230, .2030, .0880, .4860])
v99 = pq.read_table(f'{REPO}/submissions/quick-boat_v99.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)
if len(v99) != len(sub_ids):   # the final phase's four-month set (2026-10-08): v99 and v100 exist only for Jan+Jul
    print(f'CHECK skipped: v99 holds {len(v99):,} rows, this scored set {len(sub_ids):,}'); raise SystemExit(0)
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
ms = np.clip(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt), 60, 200000); app = np.abs(v99 - ms) > 60
p = v99.copy(); p[app] = np.clip(np.rint(v99[app] + delta[app]), 60, 200000)
nd = int((p != z['v100']).sum())
print(f'CHECK: v100 rebuilt from the saved members differs from the priced vector on {nd} rows;  Q_b {((p - v99) ** 2).sum():.4e}')
if os.environ.get('P2P_ALLOW_DRIFT') == '1':
    # a from-scratch regeneration re-draws some LightGBM members: LightGBM picks col-wise or row-wise histograms by
    # timing them at run time, and early stopping amplifies the difference (CL: iteration 3,631 vs 3,716). Feature
    # subsampling is NOT the cause (v99's colsample components rebuild identically). Findings s513b. Report, do not fail
    print(f'  DRIFT ALLOWED: {nd} rows differ, RMS {np.sqrt(np.mean((p - z["v100"]) ** 2)):.3f} s')
else:
    assert nd == 0
