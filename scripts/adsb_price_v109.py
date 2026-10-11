"""s543: v109 = v104 with its ADS-B stage an EQUAL-THIRDS blend of the three TREE families on v104's rows and movement
features: LightGBM (v104), XGBoost (v107's frozen FLAML settings), CatBoost (s539's fixed settings). No MLP: v108's
MLP-bearing blend transferred at 0.57x (s541) while the tree-only v107 transferred at 1.18x (s538). Priced on the Jan|Jul
instrument against v107 AND against the shipped best v108; the window cannot referee the MLP's removal (it rated the MLP
higher than the board did), so the v108 comparison is reported, not trusted.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost \
      python scripts/adsb_price_v109.py   -> cand/v109_adsb.npz
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
def vecs(dw, dr):
    nW = v100w.copy(); nW[app_w] = cl(np.rint(v100w[app_w] + dw[app_w])); nR = v100r.copy(); nR[app_r] = cl(np.rint(v100r[app_r] + dr[app_r])); return nW, nR
v107w, v107r = vecs(0.5 * ref['dW'] + 0.5 * dX, 0.5 * ref['dR'] + 0.5 * dXr)
assert np.array_equal(v107r, I.sub('v107')), 'LGB + XGB half must reproduce shipped v107'
assert np.array_equal(v107w, np.load(f'{W}/adsblol/cand/v107_adsb.npz')['newW'].astype(float)), 'and its window vector'
nW9, nR9 = vecs((ref['dW'] + dX + dC) / 3, (ref['dR'] + dXr + dCr) / 3)
z8 = np.load(f'{W}/adsblol/cand/v108_adsb.npz'); v108w, v108r = z8['newW'].astype(float), I.sub('v108')
assert np.array_equal(z8['new'].astype(float), v108r)
def price(name, iw, ir, s0):
    lin = -2.0 * (iw - y) * (nW9 - iw); own = (iw - y) ** 2 - (nW9 - y) ** 2
    Qb = float(((nR9 - ir) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
    rng = np.random.default_rng(5); cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own); k = len(cs)
    ii = [rng.integers(0, k, k) for _ in range(2000)]
    SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
    print(f'INCREMENT v109 - {name}: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e}, {own.sum()*sc/SEo:+.2f} SE)  rows R {int((nR9 != ir).sum()):,}  max |change| {np.abs(nR9 - ir).max():.0f} s\n'
          f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  -> {np.sqrt(s0**2 - G/344841):.4f}')
    return G
price('v107', v107w, v107r, 231.8661); price('v108', v108w, v108r, 231.7292)
np.savez(f'{W}/adsblol/cand/v109_adsb.npz', ids=sub_ids, v108=v108r, v107=v107r, new=nR9, win_ids=ctx['ids'], newW=nW9)
