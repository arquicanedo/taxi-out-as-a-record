"""s540: v108 = v104 with its ADS-B stage an EQUAL-QUARTERS blend of four model families on the same rows and v104's
movement features: LightGBM (v104), XGBoost (v107's frozen FLAML settings), CatBoost (fixed settings: 3,000 iterations,
learning rate 0.03, depth 8, seed 0) and a PyTorch MLP (scripts/stage_mlp.py, CPU deterministic; its predictions are
archived in artifacts/v108_stage_mlp.npz because LightGBM and PyTorch cannot share a process). No fitted weights.
Priced as an increment over the shipped v107 on the validated Jan|Jul instrument, with the paired day-cluster SE.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost \
      python scripts/adsb_price_v108.py   -> cand/v108_adsb.npz
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.environ.get('ADSB_FEAT_V104', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v104.parquet'))); os.environ['ADSB_FOLDS'] = 'janjul'
__file__ = os.path.join(HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
import xgboost as xgb
from catboost import CatBoostRegressor
PP = json.load(open(os.path.join(REPO, 'artifacts', 'v107_xgb_params.json'))); COLS = PP['cols']
assert COLS == [c for c in X.columns if c != 'rwy']
tmw_ = X.tm.values == 1; app_w, app_r, fit_w = has & (v100w < 3600), hasr & (v100r < 3600), has & ~tmw_ & (y < 7200)
ref = run('v104 LightGBM stage', cols=COLS, app_w=app_w, app_r=app_r, fit_w=fit_w)
Xa, Xra = X[COLS].copy(), Xr[COLS].copy(); Xa['apt'] = Xa['apt'].astype(int); Xra['apt'] = Xra['apt'].astype(int)
def xfit(p, m):
    q = dict(p); q.update(n_jobs=8, random_state=0, verbosity=0); q.pop('missing', None); return xgb.XGBRegressor(**q).fit(Xa[m], rres[m])
dX, dXr = np.zeros(nw), np.zeros(len(sub_ids))
dX[app_w & J] = xfit(PP['fit_jul'], fit_w & L).predict(Xa[app_w & J]); dX[app_w & L] = xfit(PP['fit_jan'], fit_w & J).predict(Xa[app_w & L])
dXr[app_r] = xfit(PP['fit_all'], fit_w).predict(Xra[app_r])
CI = [COLS.index('apt')]
def cfit(m): return CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=8, loss_function='RMSE', random_seed=0, thread_count=8, verbose=0, cat_features=CI).fit(Xa[m], rres[m])
dC, dCr = np.zeros(nw), np.zeros(len(sub_ids))
dC[app_w & J] = cfit(fit_w & L).predict(Xa[app_w & J]); dC[app_w & L] = cfit(fit_w & J).predict(Xa[app_w & L]); dCr[app_r] = cfit(fit_w).predict(Xra[app_r])
Zm = np.load(os.environ.get('STAGE_MLP_FILE', os.path.join(REPO, 'artifacts', 'v108_stage_mlp.npz'))); dM, dMr = Zm['dM'], Zm['dMr']
assert np.array_equal(dM != 0, app_w & (dM != 0)) and (dM[~app_w] == 0).all() and (dMr[~app_r] == 0).all(), 'MLP artefact rows do not match the stage rows'
def vecs(dw, dr):
    nW = v100w.copy(); nW[app_w] = cl(np.rint(v100w[app_w] + dw[app_w])); nR = v100r.copy(); nR[app_r] = cl(np.rint(v100r[app_r] + dr[app_r])); return nW, nR
v107w, v107r = vecs(0.5 * ref['dW'] + 0.5 * dX, 0.5 * ref['dR'] + 0.5 * dXr)
assert np.array_equal(v107r, I.sub('v107')), 'LGB + XGB half must reproduce shipped v107'
assert np.array_equal(v107w, np.load(f'{I.CAND}/v107_adsb.npz')['newW'].astype(float)), 'and its window vector'
nW8, nR8 = vecs((ref['dW'] + dX + dC + dM) / 4, (ref['dR'] + dXr + dCr + dMr) / 4)
lin = -2.0 * (v107w - y) * (nW8 - v107w); own = (v107w - y) ** 2 - (nW8 - y) ** 2
Qb = float(((nR8 - v107r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
rng = np.random.default_rng(5); cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own); k = len(cs)
ii = [rng.integers(0, k, k) for _ in range(2000)]
SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
print(f'\nstage correlations on applied rows: LGB-XGB {np.corrcoef(ref["dW"][app_w], dX[app_w])[0,1]:.3f}  XGB-CB {np.corrcoef(dX[app_w], dC[app_w])[0,1]:.3f}  '
      f'XGB-MLP {np.corrcoef(dX[app_w], dM[app_w])[0,1]:.3f}')
print(f'INCREMENT v108 - v107: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e}, {own.sum()*sc/SEo:+.2f} SE)  rows R {int((nR8 != v107r).sum()):,}  max |change| {np.abs(nR8 - v107r).max():.0f} s\n'
      f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  = {G/ONE:+.3f} bs  -> {np.sqrt(231.8661**2 - G/344841):.4f}')
print('  per airport own: ' + '  '.join(f'{a} {own[apt == a].sum()*sc:+.2e}' for a in sorted(AC)))
np.savez(f'{I.CAND}/v108_adsb.npz', ids=sub_ids, v107=v107r, new=nR8, win_ids=ctx['ids'], incW=v107w, newW=nW8)
