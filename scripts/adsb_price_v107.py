"""s537 step B (deterministic): v107 = v104 with its ADS-B stage replaced by a 50/50 blend of the LightGBM stage and an
XGBoost stage trained from the settings FLAML found (artifacts/v107_xgb_params.json, scripts/v107_flaml_search.py).
Same rows, same v104 movement features. Priced as an increment over the shipped v104 on the validated Jan|Jul instrument,
with the PAIRED day-cluster SE of the window difference, and a year-drift check (s535): PSI between 2025 window and 2026
scored rows for the features the XGBoost stage leans on.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_price_v107.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.environ.get('ADSB_FEAT_V104', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v104.parquet'))); os.environ['ADSB_FOLDS'] = 'janjul'
__file__ = os.path.join(HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
import xgboost as xgb
PP = json.load(open(os.path.join(REPO, 'artifacts', 'v107_xgb_params.json'))); COLS = PP['cols']
assert COLS == [c for c in X.columns if c != 'rwy'], 'feature list changed since the search'
tmw_ = X.tm.values == 1
app_w, app_r, fit_w = has & (v100w < 3600), hasr & (v100r < 3600), has & ~tmw_ & (y < 7200)
ref = run('v104 LightGBM stage', cols=COLS, app_w=app_w, app_r=app_r, fit_w=fit_w)
v104r = I.sub('v104'); assert np.array_equal(ref['nR'], v104r), 'LightGBM half must reproduce shipped v104'
assert np.array_equal(ref['nW'], np.load(f'{I.CAND}/v104_adsb.npz')['newW'].astype(float)), 'and its window vector'
Xa, Xra = X[COLS].copy(), Xr[COLS].copy(); Xa['apt'] = Xa['apt'].astype(int); Xra['apt'] = Xra['apt'].astype(int)
def xfit(p, m):
    q = dict(p); q.update(n_jobs=8, random_state=0, verbosity=0); q.pop('missing', None)
    return xgb.XGBRegressor(**q).fit(Xa[m], rres[m])
dX = np.zeros(nw); dXr = np.zeros(len(sub_ids))
dX[app_w & J] = xfit(PP['fit_jul'], fit_w & L).predict(Xa[app_w & J]); dX[app_w & L] = xfit(PP['fit_jan'], fit_w & J).predict(Xa[app_w & L])
mall = xfit(PP['fit_all'], fit_w); dXr[app_r] = mall.predict(Xra[app_r])
iw = ref['nW']
def build(wx):
    dW = (1 - wx) * ref['dW'] + wx * dX; dR = (1 - wx) * ref['dR'] + wx * dXr
    nW = v100w.copy(); nW[app_w] = cl(np.rint(v100w[app_w] + dW[app_w])); nR = v100r.copy(); nR[app_r] = cl(np.rint(v100r[app_r] + dR[app_r]))
    return nW, nR
def price(name, nW, nR):
    lin = -2.0 * (iw - y) * (nW - iw); own = (iw - y) ** 2 - (nW - y) ** 2
    Qb = float(((nR - v104r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
    rng = np.random.default_rng(5); cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own); k = len(cs)
    ii = [rng.integers(0, k, k) for _ in range(2000)]
    SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
    print(f'\n{name}: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e}, {own.sum()*sc/SEo:+.2f} SE)  rows R {int((nR != v104r).sum()):,}  max |change| {np.abs(nR - v104r).max():.0f} s\n'
          f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  = {G/ONE:+.3f} bs  -> {np.sqrt(233.1847**2 - G/344841):.4f}')
    print('  per airport own: ' + '  '.join(f'{a} {own[apt == a].sum()*sc:+.2e}' for a in sorted(AC)))
    return G, SE
price('XGBoost stage alone', *build(1.0))
nW7, nR7 = build(0.5); G7, SE7 = price('v107 = 50/50 LightGBM + XGBoost stage', nW7, nR7)
# ---- year-drift check (s535): the features the XGBoost stage leans on, 2025 window rows against 2026 scored rows -----
imp = pd.Series(mall.get_booster().get_score(importance_type='gain')).reindex(COLS).fillna(0); imp = imp / imp.sum()
def psi(a, b, n=10):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 100 or len(b) < 100: return np.nan
    q = np.unique(np.quantile(a, np.linspace(0, 1, n + 1))); q[0], q[-1] = -np.inf, np.inf
    pa = np.histogram(a, q)[0] / len(a) + 1e-4; pb = np.histogram(b, q)[0] / len(b) + 1e-4
    return float(((pa - pb) * np.log(pa / pb)).sum())
print('\nDRIFT CHECK (PSI, 2025 window rows vs 2026 scored rows; > 0.25 is a large shift):')
for c in imp.sort_values(ascending=False).index[:10]:
    print(f'  {c:<10} XGB gain share {imp[c]:.3f}   PSI {psi(Xa.loc[app_w, c].to_numpy(float), Xra.loc[app_r, c].to_numpy(float)):.3f}')
np.savez(f'{I.CAND}/v107_adsb.npz', ids=sub_ids, v104=v104r, new=nR7, win_ids=ctx['ids'], incW=iw, newW=nW7)
