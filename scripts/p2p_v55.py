"""p2p steps 1 and 2: the two SHIPPED tree members (v55_trees.parquet's lgbm_opdi / perapt_opdi construction),
on the window and on the ranking side, with the target encodings chosen explicitly (s492 P).

Copy of scripts/p1a_window_v55.py (the window version of scripts/v55_trees.py) with the te_* columns swapped:
  pooled       v38 recipe: lr .03, 6000 trees, 511 leaves, feature_fraction .7, min_child 20, F_BULK + 4 OPDI
  per-airport  600 trees / 127 leaves / min_child 100 on the same 75 features, pooled fallback under 500 rows
  target       y - naive (apt_hr_naive fallback, then the training nanmedian), non-extreme (y < 7200) training rows

MODE=win   train on the ten non-window months (2-6, 8-12), predict the 344,336 window rows (v15_tr_jul19 order).
           TE=lf   ten-month encodings on every row (p2p_te.py te10): the LEAK-FREE faithful replica.
           TE=oof  training rows get 5-fold calendar-day OUT-OF-FOLD ten-month encodings (te10_oof); window rows te10.
MODE=rank  train on all twelve 2025 months, predict the 344,841 v15_rk_jul19 rows (production, as v55_trees.py).
           TE=stored  the stored twelve-month encodings (= production; CONTROL: reproduces v55_trees.parquet)
           TE=oof     training rows get 5-fold calendar-day out-of-fold twelve-month encodings (te12_oof);
                      ranking rows keep the stored twelve-month encodings (te12_rk == stored, asserted in p2p_te.py).
Env: MODE, TE, MODELS (pooled,perapt), SEEDS (0,1,2), NJ. Every seed saved separately; resumable.
"""
import os, json, time, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
W = os.environ['DC26_WORK']; t0 = time.time()
P2 = os.environ.get('P2P_OUT', os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')
MODE = os.environ.get('MODE', 'win'); TE = os.environ.get('TE', 'lf' if MODE == 'win' else 'stored')
MODELS = os.environ.get('MODELS', 'pooled,perapt').split(','); SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2').split(',')]
NJ = int(os.environ.get('NJ', '6')); OUT = f'{P2}/trees'; os.makedirs(OUT, exist_ok=True)
assert (MODE, TE) in {('win', 'lf'), ('win', 'oof'), ('rank', 'stored'), ('rank', 'oof')}, (MODE, TE)
OPDI = ['af_since_arr', 'af_sched_turn', 'af_prev_taxiin', 'af_has']
tr = pd.read_parquet(f'{W}/v15_tr_jul19.parquet')
c = json.load(open(f'{W}/v15_cols_jul19.json')); FB, CATS = list(c['F_BULK']), c['CATS']
z = np.load(f'{P2}/te.npz', allow_pickle=True); TEC = list(z['te_cols'])
assert (z['ids_tr'] == tr.MVT_ID_mvt.values).all(), 'te.npz order (tr)'
A = pd.read_parquet(f'{W}/opdi_airframe.parquet'); n0 = len(tr)
mon = pd.to_datetime(tr.MVT_TIME_UTC_mvt, utc=True).dt.month.values
y = tr.y.values.astype(float); ext = y >= 7200; win = np.isin(mon, [1, 7])
if MODE == 'win':
    arr = {'lf': z['te10'], 'oof': z['te10_oof']}[TE]
    for j, n in enumerate(TEC): tr[n] = arr[:, j]
    TRN = ~win & ~ext; ev = tr[win].copy()
else:
    rk = pd.read_parquet(f'{W}/v15_rk_jul19.parquet'); assert (z['ids_rk'] == rk.MVT_ID_mvt.values).all(), 'te.npz order (rk)'
    if TE == 'oof':
        for j, n in enumerate(TEC): tr[n] = z['te12_oof'][:, j]
    else:
        for j, n in enumerate(TEC): assert np.allclose(tr[n].values, z['te12'][:, j], equal_nan=True)
    for j, n in enumerate(TEC): assert np.allclose(rk[n].values, z['te12_rk'][:, j], equal_nan=True)
    TRN = ~ext; ev = rk
tr = tr.merge(A, on='MVT_ID_mvt', how='left'); assert len(tr) == n0, 'opdi merge changed row count'
ne = len(ev); ev = ev.merge(A, on='MVT_ID_mvt', how='left'); assert len(ev) == ne
for x in CATS:   # window: categories from v15_tr (p1a_window_v55.py); rank: tr | rk union (v55_trees.py)
    vals = set(tr[x].dropna().astype(str)) | (set(ev[x].dropna().astype(str)) if MODE == 'rank' else set())
    u = pd.api.types.CategoricalDtype(sorted(vals)); tr[x] = tr[x].astype(str).astype(u); ev[x] = ev[x].astype(str).astype(u)
nv = tr.naive.values.astype(float); ah = tr.apt_hr_naive.values.astype(float)
b = np.where(~np.isnan(nv), nv, ah); g = np.nanmedian(b[TRN]); b = np.nan_to_num(b, nan=g); rt = y - b
ne_ = ev.naive.values.astype(float); be = np.where(~np.isnan(ne_), ne_, ev.apt_hr_naive.values.astype(float)); be = np.nan_to_num(be, nan=g)
F = FB + OPDI; X = tr[F]; Xe = ev[F]
apt = tr.apt.astype(str).values; ae = ev.apt.astype(str).values
print(f'MODE {MODE} TE {TE}: train {int(TRN.sum()):,}  predict {len(ev):,}  features {len(F)}  OPDI coverage (predicted rows) '
      f'{ev.af_has.fillna(0).mean():.1%}  NJ {NJ}  [{time.time()-t0:.0f}s]', flush=True)
V38 = dict(learning_rate=.03, n_estimators=6000, num_leaves=511, feature_fraction=.7, min_child_samples=20, reg_lambda=1., n_jobs=NJ, verbose=-1)
PA = dict(learning_rate=.06, n_estimators=600, num_leaves=127, min_child_samples=100, reg_lambda=1., n_jobs=NJ, verbose=-1)
f = lambda t, s: f'{OUT}/{MODE}_{TE}_{t}_s{s}.npy'
if 'pooled' in MODELS:
    for s in SEEDS:
        if os.path.exists(f('pooled', s)): continue
        t1 = time.time()
        p = be + lgb.LGBMRegressor(random_state=s, **V38).fit(X[TRN], rt[TRN], categorical_feature=CATS).predict(Xe)
        np.save(f('pooled', s), p); print(f'  pooled seed {s} [{time.time()-t1:.0f}s, total {time.time()-t0:.0f}s]', flush=True)
if 'perapt' in MODELS:
    for s in SEEDS:
        if os.path.exists(f('perapt', s)): continue
        t1 = time.time(); p = np.full(len(ev), np.nan)
        for a in np.unique(ae):
            it = TRN & (apt == a); iv = ae == a
            if it.sum() < 500: p[iv] = np.load(f('pooled', s))[iv]; continue
            p[iv] = be[iv] + lgb.LGBMRegressor(random_state=s, **PA).fit(X[it], rt[it], categorical_feature=CATS).predict(Xe[iv])
        assert not np.isnan(p).any(); np.save(f('perapt', s), p)
        print(f'  perapt seed {s} [{time.time()-t1:.0f}s, total {time.time()-t0:.0f}s]', flush=True)
r = lambda p, yy, k: float(np.sqrt(np.mean((np.clip(p, 60, 200000)[k] - yy[k]) ** 2)))
if MODE == 'win':
    yw = y[win]; ow = yw < 7200; mw = mon[win]
    for t in MODELS:
        fs = [f(t, s) for s in SEEDS if os.path.exists(f(t, s))]
        if not fs: continue
        ps = [np.load(x) for x in fs]; p = np.mean(ps, 0)
        print(f'  {MODE}/{TE} {t}: {len(ps)}-seed mean ordinary {r(p,yw,ow):.3f}  Jan {r(p,yw,ow&(mw==1)):.4f}  Jul {r(p,yw,ow&(mw==7)):.4f}'
              f'   seeds ' + ' '.join(f'{r(q,yw,ow):.3f}' for q in ps), flush=True)
else:
    ref = pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(ev.MVT_ID_mvt.values)
    for t, col in (('pooled', 'lgbm_opdi'), ('perapt', 'perapt_opdi')):
        fs = [f(t, s) for s in SEEDS if os.path.exists(f(t, s))]
        if t not in MODELS or not fs: continue
        p = np.mean([np.load(x) for x in fs], 0); d = p - ref[col].values
        print(f'  rank/{TE} {t} ({len(fs)} seeds) vs shipped {col}: mean {d.mean():+.4f}  RMS {np.sqrt(np.mean(d**2)):.4f}  |max| {np.abs(d).max():.3f}'
              f'  corr {np.corrcoef(p, ref[col].values)[0,1]:.6f}', flush=True)
