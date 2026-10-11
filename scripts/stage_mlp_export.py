"""s539: export the ADS-B stage's inputs (v104's rows and movement features) for a separate-process PyTorch stage
(LightGBM and PyTorch cannot share a process here). -> $DC26_WORK/adsblol/cand/stage_inputs.npz
  uv run --with pandas --with pyarrow --with numpy --with lightgbm python scripts/stage_mlp_export.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.environ.get('ADSB_FEAT_V104', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v104.parquet'))); os.environ['ADSB_FOLDS'] = 'janjul'
__file__ = os.path.join(HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
tmw_ = X.tm.values == 1; COLS = [c for c in X.columns if c != 'rwy']
num = [c for c in COLS if c != 'apt']
A = X[num].to_numpy(float); Ar = Xr[num].to_numpy(float)
np.savez(os.path.join(I.CAND, 'stage_inputs.npz'), A=A, Ar=Ar, apt=X['apt'].astype(int).values, aptr=Xr['apt'].astype(int).values,
         t=rres, fit=has & ~tmw_ & (y < 7200), app=has & (v100w < 3600), appr=hasr & (v100r < 3600), J=J, L=L, cols=np.array(num))
print('exported', A.shape, Ar.shape)
