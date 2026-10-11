"""p2p step 2: candidate new bulk members 'idcat' and 'rwyq' (s489, s492 P), built on BOTH sides with the same code.

Model = the leak-free member protocol's pooled arm model (scripts/p1v/e_v_fit.py, MODEL=pooled, CS=0.4):
LightGBM lr .06, 2000 trees, 255 leaves, min_child 20, reg_lambda 1, colsample_bytree 0.4, on F_BULK (71) plus
the arm's features; target y - naive (apt_hr_naive fallback, then the training nanmedian); non-extreme rows.
  idcat  s421's identity columns as native categoricals: op x apt, stand x apt, carrier (FLIGHT_mvt[:3]) x apt
         (scripts/p1e_arms.py idcat_feats, re-implemented here so the ranking rows can be joined too)
  rwyq   same-(airport, runway) neighbours' observable naive (scripts/p1e_arms.py rwyq_feats, called verbatim),
         computed inside the frame being predicted (2025 file for the window, the ranking file for 2026)
MODE=win   ten non-window months, ten-month te_* (p2p_te.py te10) on every row; predicts the 344,336 window rows.
MODE=rank  all twelve 2025 months, the stored twelve-month te_*; predicts the 344,841 v15_rk_jul19 rows.
Env: MODE, ARM, SEEDS (0,1), NJ. Output <P2P_OUT>/arms/{MODE}_{ARM}_s{seed}.npy.
"""
import os, sys, glob, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import p1e_arms
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']; t0 = time.time()
P2 = os.environ.get('P2P_OUT', os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')
MODE = os.environ.get('MODE', 'win'); ARM = os.environ.get('ARM', 'idcat'); NJ = int(os.environ.get('NJ', '4'))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1').split(',')]; OUT = f'{P2}/arms'; os.makedirs(OUT, exist_ok=True)
assert MODE in ('win', 'rank') and ARM in ('idcat', 'rwyq')
tr = pd.read_parquet(f'{W}/v15_tr_jul19.parquet')
c = json.load(open(f'{W}/v15_cols_jul19.json')); FB, CATS = list(c['F_BULK']), list(c['CATS'])
z = np.load(f'{P2}/te.npz', allow_pickle=True); TEC = list(z['te_cols']); assert (z['ids_tr'] == tr.MVT_ID_mvt.values).all()
mon = pd.to_datetime(tr.MVT_TIME_UTC_mvt, utc=True).dt.month.values
y = tr.y.values.astype(float); ext = y >= 7200; win = np.isin(mon, [1, 7])
if MODE == 'win':
    for j, n in enumerate(TEC): tr[n] = z['te10'][:, j]
    TRN = ~win & ~ext; ev = tr[win].copy()
else:
    rk = pd.read_parquet(f'{W}/v15_rk_jul19.parquet'); assert (z['ids_rk'] == rk.MVT_ID_mvt.values).all()
    for j, n in enumerate(TEC):
        assert np.allclose(tr[n].values, z['te12'][:, j], equal_nan=True) and np.allclose(rk[n].values, z['te12_rk'][:, j], equal_nan=True)
    TRN = ~ext; ev = rk

def idcat(frame, files):
    C = ['MVT_ID_mvt', 'PHASE_mvt', 'FLIGHT_mvt', 'AIRCRAFT_OPERATOR_flt', 'STAND_mvt', 'ADEP_mvt']
    meta = pd.concat([pq.read_table(f, columns=C).to_pandas() for f in files], ignore_index=True)
    meta = meta[meta.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt')
    M = meta.set_index('MVT_ID_mvt').reindex(frame.MVT_ID_mvt.values)
    assert M.ADEP_mvt.notna().mean() > 0.999, f'join coverage {M.ADEP_mvt.notna().mean():.4f}'
    S = lambda c: M[c].astype(str).values
    return pd.DataFrame({'op_apt': S('ADEP_mvt').astype(object) + '|' + S('AIRCRAFT_OPERATOR_flt'),
                         'stand_apt': S('ADEP_mvt').astype(object) + '|' + S('STAND_mvt'),
                         'car_apt': S('ADEP_mvt').astype(object) + '|' + pd.Series(S('FLIGHT_mvt')).str[:3].values}, index=frame.index)
TRF = sorted(glob.glob(f'{D}/training_*.parquet'))
if ARM == 'idcat':
    a_tr = idcat(tr, TRF)
    a_ev = a_tr[win].copy() if MODE == 'win' else idcat(ev, [f'{D}/ranking.parquet'])
    CATS = CATS + list(a_tr.columns)
else:
    a_tr = p1e_arms.rwyq_feats(tr)
    a_ev = a_tr[win].copy() if MODE == 'win' else p1e_arms.rwyq_feats(ev)
for cc in a_tr.columns: tr[cc] = a_tr[cc].values; ev[cc] = a_ev[cc].values
FA = FB + list(a_tr.columns)
for x in CATS:
    vals = set(tr[x].dropna().astype(str)) | (set(ev[x].dropna().astype(str)) if MODE == 'rank' else set())
    u = pd.api.types.CategoricalDtype(sorted(vals)); tr[x] = tr[x].astype(str).astype(u); ev[x] = ev[x].astype(str).astype(u)
for cc in a_tr.columns:
    if ev[cc].dtype.kind == 'f':
        print(f'    {cc:<14} coverage train {np.isfinite(tr[cc].values[TRN]).mean():6.1%}  predicted rows {np.isfinite(ev[cc].values).mean():6.1%}'
              f'  median {np.nanmedian(ev[cc].values):9.2f}', flush=True)
    else:
        seen = ev[cc].astype(str).isin(set(tr[cc][TRN].astype(str))).mean()
        print(f'    {cc:<14} categories {len(tr[cc].cat.categories):,}  predicted rows whose level occurs in training {seen:.2%}', flush=True)
nv = tr.naive.values.astype(float); b = np.where(~np.isnan(nv), nv, tr.apt_hr_naive.values.astype(float))
g = np.nanmedian(b[TRN]); b = np.nan_to_num(b, nan=g); rt = y - b
ne = ev.naive.values.astype(float); be = np.nan_to_num(np.where(~np.isnan(ne), ne, ev.apt_hr_naive.values.astype(float)), nan=g)
P = dict(objective='regression', learning_rate=.06, n_estimators=2000, num_leaves=255, min_child_samples=20, reg_lambda=1.,
         colsample_bytree=0.4, n_jobs=NJ, verbose=-1)
print(f'MODE {MODE} ARM {ARM}: train {int(TRN.sum()):,}  predict {len(ev):,}  features {len(FA)}  cats {len(CATS)}  [{time.time()-t0:.0f}s]', flush=True)
for s in SEEDS:
    fn = f'{OUT}/{MODE}_{ARM}_s{s}.npy'
    if os.path.exists(fn): continue
    t1 = time.time()
    p = be + lgb.LGBMRegressor(random_state=s, **P).fit(tr[FA][TRN], rt[TRN], categorical_feature=CATS).predict(ev[FA])
    np.save(fn, p); print(f'  seed {s} [{time.time()-t1:.0f}s]', flush=True)
    if MODE == 'win':
        yw = y[win]; o = yw < 7200; mw = mon[win]; r = lambda k: float(np.sqrt(np.mean((np.clip(p, 60, 200000)[k] - yw[k]) ** 2)))
        print(f'    window ordinary {r(o):.4f}  Jan {r(o&(mw==1)):.4f}  Jul {r(o&(mw==7)):.4f}', flush=True)
        ref = f'{P2}/../trackE/verify/win_{ARM}_pooled_cs0.4_s{s}_teleakfree.npy'
        if os.path.exists(ref):
            q = np.load(ref); print(f'    vs trackE leak-free {ARM} seed {s}: RMS diff {np.sqrt(np.mean((q-np.clip(p,60,200000))**2)):.4f}  '
                                    f'|max| {np.abs(q-np.clip(p,60,200000)).max():.3f}', flush=True)
    else:
        print(f'    ranking prediction median {np.median(p):.1f}  q01 {np.quantile(p,.01):.1f}  q99 {np.quantile(p,.99):.1f}', flush=True)
