"""p1a: window-side versions of the tree members that ACTUALLY ship (v55_trees.parquet), so the window
base matches v96's blend in structure and quality.

v64_build.py blends v55_trees.parquet's lgbm_opdi / perapt_opdi, not v15_trees_jul19's lgbm / perapt, and
every submission from v64 to v99 carries that blend (v96 on its 314,550 unfired rows equals
rint(.2230 lgbm_opdi + .2030 perapt_opdi + .0880 D2 + .4860 K128 + SHIFT_month) exactly on 80.0% of rows
and within 1 s on 98.7%; with v15_trees_jul19 in its place, 6.5% and 19.2%). v55_trees.py builds:
  pooled       v38 recipe: lr .03, 6000 trees, 511 leaves, feature_fraction .7, min_child 20, F_BULK + OPDI
  per-airport  600/127/100 on F_BULK + OPDI, pooled fallback under 500 rows
  target       y - naive (apt_hr_naive fallback, then nanmedian), non-extreme training rows, seeds 0,1,2
Here the same construction is trained on the TEN NON-WINDOW months and predicts the 344,336 window rows
(v15_tr_jul19 file order). n_jobs capped. Every seed saved separately; resumable.
"""
import os, json, time, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
W = os.environ['DC26_WORK']; t0 = time.time()
OUT = os.environ.get('P1A_OUT', (os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackA')) + '/wc'
os.makedirs(OUT, exist_ok=True); NJ = int(os.environ.get('NJ', '5'))
OPDI = ['af_since_arr', 'af_sched_turn', 'af_prev_taxiin', 'af_has']
tr = pd.read_parquet(f'{W}/v15_tr_jul19.parquet')
c = json.load(open(f'{W}/v15_cols_jul19.json')); FB, CATS = list(c['F_BULK']), c['CATS']
A = pd.read_parquet(f'{W}/opdi_airframe.parquet'); n0 = len(tr)
tr = tr.merge(A, on='MVT_ID_mvt', how='left'); assert len(tr) == n0, 'opdi merge changed row count'
for x in CATS:
    u = pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str))))
    tr[x] = tr[x].astype(str).astype(u)
mon = pd.to_datetime(tr.MVT_TIME_UTC_mvt, utc=True).dt.month.values
y = tr.y.values.astype(float); ext = y >= 7200
win = np.isin(mon, [1, 7]); TRN = ~win & ~ext
nv = tr.naive.values.astype(float); ah = tr.apt_hr_naive.values.astype(float)
b = np.where(~np.isnan(nv), nv, ah); g = np.nanmedian(b[TRN]); b = np.nan_to_num(b, nan=g); rt = y - b
F = FB + OPDI; X = tr[F]; Xw = X[win]; bw = b[win]; yw = y[win]
apt = tr.apt.astype(str).values; aw = apt[win]
print(f'train {int(TRN.sum()):,}  window {int(win.sum()):,}  features {len(F)}  OPDI coverage window {tr.af_has.fillna(0)[win].mean():.1%}', flush=True)
V38 = dict(learning_rate=.03, n_estimators=6000, num_leaves=511, feature_fraction=.7, min_child_samples=20, reg_lambda=1., n_jobs=NJ, verbose=-1)
PA = dict(learning_rate=.06, n_estimators=600, num_leaves=127, min_child_samples=100, reg_lambda=1., n_jobs=NJ, verbose=-1)
CS = float(os.environ.get('CS', '1.0'))  # per-airport colsample; 1.0 is the shipped v55 setting
if CS < 1: PA['colsample_bytree'] = CS
f = lambda t, s: f'{OUT}/v55_{t}_s{s}.npy' if (t == 'pooled' or CS == 1.0) else f'{OUT}/v55_{t}_cs{CS:.1f}_s{s}.npy'
ONLY = os.environ.get('ONLY', '')          # 'perapt' runs the per-airport loop alone (no airport has < 500 rows)
for s in (0, 1, 2):
    if ONLY != 'perapt' and not os.path.exists(f('pooled', s)):
        p = bw + lgb.LGBMRegressor(random_state=s, **V38).fit(X[TRN], rt[TRN], categorical_feature=CATS).predict(Xw)
        np.save(f('pooled', s), p); print(f'  v55 pooled seed {s} [{time.time()-t0:.0f}s]', flush=True)
for s in (0, 1, 2):
    if not os.path.exists(f('perapt', s)):
        p = np.full(len(yw), np.nan)
        for a in np.unique(aw):
            it = TRN & (apt == a); iv = aw == a
            if it.sum() < 500: p[iv] = np.load(f('pooled', s))[iv]; continue
            p[iv] = bw[iv] + lgb.LGBMRegressor(random_state=s, **PA).fit(X[it], rt[it], categorical_feature=CATS).predict(Xw[iv])
        assert not np.isnan(p).any(); np.save(f('perapt', s), p); print(f'  v55 perapt seed {s} [{time.time()-t0:.0f}s]', flush=True)
if ONLY == 'perapt': sys.exit(0)
ew = ext[win]; mw = mon[win]
r = lambda p, k: float(np.sqrt(np.mean((np.clip(p, 60, 200000)[k] - yw[k]) ** 2)))
for t in ('pooled', 'perapt'):
    p = np.mean([np.load(f(t, s)) for s in (0, 1, 2)], 0)
    print(f'  v55-equivalent {t}: ordinary {r(p,~ew):.3f}  Jan {r(p,~ew&(mw==1)):.4f}  Jul {r(p,~ew&(mw==7)):.4f}')
