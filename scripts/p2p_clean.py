"""p2p step 2: the clean-room bulk as a candidate member (track C's 'fin' model, f3 feature set), on BOTH sides.

Model and encodings are copied verbatim from the Phase 1 track C research scripts p1c_bulk.py and p1c_common.py,
which are not part of the release: this file is self-contained (residual LightGBM on y - te_k_ssr,
lr 0.05, 255 leaves, feature_fraction .5, bagging .8, lambda_l2 10, up to 30000 rounds with early stopping on a
10% calendar-day holdout of the TRAINING months; training rows y < 7200; nohc feature list; 131 features, 10 cats).
Its target encodings (te_k_*, cr_k_*) are fitted on TRAINING rows only: out-of-fold by calendar day (5 folds) on
training rows, full training fit on the predicted rows. The only generalisation is which rows are "predicted":

  MODE=win   training = the ten non-window 2025 months, predicted = 2025 January + July (src 0 and 6), exactly
             p1c_common.load(nohc=True, f3=True). CONTROL: reproduces track C's p1c_bulk_fin_s{seed}.npy.
  MODE=rank  training = all twelve 2025 months, predicted = the 344,841 ranking departures (src 12), whose
             features come from scripts/p2p_clean_feat{1,2,3}.py SEL=rank (the same feature code run on the ranking
             file). The integer category codes c_* are re-derived from the raw strings over the UNION of the two
             frames, because p1c_features.py codes them within a frame and a code must mean the same level on both.

Output (window: v15_tr_jul19 window order; ranking: submitting order):
  <P2P_OUT>/clean/{win,rank}_clean_s{seed}.npy
Env: MODE, SEEDS (0,1), NJ (5), FEATSRC (p2p = the p2p rebuild of the 2025 features, trackC = track C's files).
"""
import os, sys, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq
t0 = time.time()
TC = os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackC/'
P2 = os.environ.get('P2P_OUT', os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P') + '/clean/'
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
MODE = os.environ.get('MODE', 'win'); SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1').split(',')]
NJ = int(os.environ.get('NJ', '5')); FEATSRC = os.environ.get('FEATSRC', 'p2p'); LR = 0.05
DROP = {'MVT_ID_mvt', 'src', 'y', 'apt', 'k_flight', 'k_ssr', 'k_cs', 'k_opst', 'k_dest', 't_mvt', 't_sch', 'day'}

def frame(sel):
    if sel == 'train' and FEATSRC == 'trackC':
        fs = [TC + 'p1c_feat.parquet', TC + 'p1c_feat2.parquet', TC + 'p1c_feat3.parquet']
    else:
        fs = [P2 + f'feat{i}_{sel}.parquet' for i in (1, 2, 3)]
    F = pd.read_parquet(fs[0])
    for f in fs[1:]:
        G = pd.read_parquet(f); assert (G.MVT_ID_mvt.values == F.MVT_ID_mvt.values).all()
        F = pd.concat([F, G.drop(columns=['MVT_ID_mvt'])], axis=1)
    return F

F = frame('train')
if MODE == 'rank':
    R = frame('rank'); assert list(R.columns) == list(F.columns), 'ranking feature columns differ'
    # re-derive the c_* codes over the union of both frames (p1c_features.py cat_src, verbatim expressions)
    raw = pd.read_parquet(TC + 'p1c_raw.parquet')
    def cat_src(dep):
        return {'c_apt': dep.apt, 'c_stand': dep.apt + '_' + dep.STAND_mvt.fillna('NA'), 'c_rwy': dep.apt + '_' + dep.RUNWAY_mvt.fillna('NA'),
                'c_type': dep.AIRCRAFT_TYPE_mvt, 'c_wtc': dep.WK_TBL_CAT_flt, 'c_mseg': dep.MARKET_SEGMENT_flt, 'c_ftype': dep.FLIGHT_TYPE_flt,
                'c_op': dep.AIRCRAFT_OPERATOR_flt, 'c_cs3': dep.CALLSIGN_flt.str[:3], 'c_fl2': dep.FLIGHT_mvt.str.extract(r'^([A-Z0-9]{2})', expand=False),
                'c_ades': dep.ADES_mvt, 'c_stpre': dep.apt + '_' + dep.STAND_mvt.fillna('NA').str.extract(r'^([A-Za-z]*\d?)', expand=False).fillna('NA'),
                'c_rule': dep.FLIGHT_RULE_mvt, 'c_adesfiled_diff': (dep.ADES_FILED_flt != dep.ADES_mvt).astype('string'),
                'c_typematch': (dep.AIRCRAFT_TYPE_flt != dep.AIRCRAFT_TYPE_mvt).astype('string')}
    dt = raw[(raw.src < 12) & raw.dep].reset_index(drop=True); dr = raw[(raw.src == 12) & raw.dep].reset_index(drop=True)
    assert (dt.MVT_ID_mvt.values == F.MVT_ID_mvt.values).all() and (dr.MVT_ID_mvt.values == R.MVT_ID_mvt.values).all()
    st, sr = cat_src(dt), cat_src(dr)
    for k in st:
        a = st[k].astype('string').fillna('NA'); b = sr[k].astype('string').fillna('NA')
        c0 = pd.Categorical(a).codes; assert (c0 == F[k].values).all(), f'{k}: 2025 codes not reproduced from the raw strings'
        cats = pd.Categorical(pd.concat([a, b], ignore_index=True)).categories
        F[k] = pd.Categorical(a, categories=cats).codes.astype('int32'); R[k] = pd.Categorical(b, categories=cats).codes.astype('int32')
        print(f'  recode {k:<17} levels 2025 {len(pd.unique(a)):>6,}  union {len(cats):>6,}  ranking rows at a level unseen in 2025 {(~b.isin(set(a))).mean():.4%}', flush=True)
    del raw, dt, dr
    F = pd.concat([F, R], ignore_index=True)
    win = (F.src == 12).values
else:
    win = F.src.isin([0, 6]).values
# ---------------- p1c_common.load(nohc=True, f3=True), verbatim except that `win` is the predicted set -----------
y = F.y.values
ordtr = (~win) & (y < 7200)
dh = pd.util.hash_array(F.day.values.astype('int64'))
fold = (dh % 5).astype(int)
hold = (~win) & ((pd.util.hash_array(F.day.values.astype('int64') + 7919) % 10) == 0)
aptmean = pd.Series(y[ordtr]).groupby(F.apt.values[ordtr]).mean()
prior = F.apt.map(aptmean).values
def te(key, alpha=20.0):
    k = pd.Series(pd.factorize(F[key].values)[0])
    out = np.full(len(F), np.nan); cnt = np.zeros(len(F))
    def agg(mask): return pd.DataFrame({'k': k[mask].values, 'y': y[mask]}).groupby('k').y.agg(['sum', 'count'])
    for f in range(5):
        m_fit = ordtr & (fold != f); m_app = (~win) & (fold == f)
        g = agg(m_fit); s = k[m_app].map(g['sum']).fillna(0).values; n = k[m_app].map(g['count']).fillna(0).values
        out[m_app] = (s + alpha * prior[m_app]) / (n + alpha); cnt[m_app] = n
    g = agg(ordtr); m_app = win
    s = k[m_app].map(g['sum']).fillna(0).values; n = k[m_app].map(g['count']).fillna(0).values
    out[m_app] = (s + alpha * prior[m_app]) / (n + alpha); cnt[m_app] = n
    return out, cnt
for key in ['k_ssr', 'k_flight', 'k_cs', 'k_opst', 'k_dest']:
    F['te_' + key], F['n_' + key] = te(key)
cs = (np.abs(y - F.d_sch.values) <= 10).astype(float)
trany = ~win
for key in ['k_flight', 'k_opst', 'k_ssr']:
    k = pd.Series(pd.factorize(F[key].values)[0]); out = np.full(len(F), np.nan)
    pr = pd.Series(cs[trany]).groupby(F.apt.values[trany]).mean(); prr = F.apt.map(pr).values
    def agg(mask): return pd.DataFrame({'k': k[mask].values, 'c': cs[mask]}).groupby('k').c.agg(['sum', 'count'])
    for f in range(5):
        m_fit = trany & (fold != f); m_app = trany & (fold == f)
        g = agg(m_fit); s = k[m_app].map(g['sum']).fillna(0).values; n = k[m_app].map(g['count']).fillna(0).values
        out[m_app] = (s + 20 * prr[m_app]) / (n + 20)
    g = agg(trany); m_app = win
    s = k[m_app].map(g['sum']).fillna(0).values; n = k[m_app].map(g['count']).fillna(0).values
    out[m_app] = (s + 20 * prr[m_app]) / (n + 20)
    F['cr_' + key] = out
feats = [c for c in F.columns if c not in DROP]
feats = [c for c in feats if c not in ('c_stand', 'c_ades', 'c_cs3', 'c_fl2', 'c_op')]
cats = [c for c in feats if c.startswith('c_')]
base = F['te_k_ssr'].values
print(f'MODE {MODE}: rows {len(F):,}  predicted {int(win.sum()):,}  training months rows {int((~win).sum()):,}  '
      f'ordinary training {int(ordtr.sum()):,}  holdout days share {hold[~win].mean():.3f}   {len(feats)} features {len(cats)} cats  [{time.time()-t0:.0f}s]', flush=True)
assert len(feats) == 131 and len(cats) == 10
# ---------------- p1c_bulk.py, verbatim ----------------------------------------------------------------------------
X = F[feats]; Xw = X[win]
tr = (~win) & (~hold) & (y < 7200); ho = (~win) & hold & (y < 7200)
tgt = y - base
ids_pred = F.MVT_ID_mvt.values[win]
if MODE == 'win':
    v15 = pd.read_parquet(f'{W}/v15_tr_jul19.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'y'])
    vw = v15[np.isin(pd.to_datetime(v15.MVT_TIME_UTC_mvt, utc=True).dt.month.values, [1, 7])]
    pos = pd.Series(np.arange(len(ids_pred)), index=ids_pred).reindex(vw.MVT_ID_mvt.values)
    assert pos.notna().all(), 'a v15 window row has no clean-room row'; pos = pos.values.astype(int)
    assert np.allclose(y[win][pos], vw.y.values)
else:
    sub = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
    pos = pd.Series(np.arange(len(ids_pred)), index=ids_pred).reindex(sub)
    assert pos.notna().all(), 'a scored row has no clean-room row'; pos = pos.values.astype(int)
for SEED in SEEDS:
    fn = P2 + f'{MODE}_clean_s{SEED}.npy'
    if os.path.exists(fn): continue
    t1 = time.time()
    P = dict(objective='regression', learning_rate=LR, num_leaves=255, min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=10.0, max_bin=255, cat_smooth=20, cat_l2=10, max_cat_to_onehot=8, n_jobs=NJ, seed=SEED, verbose=-1, min_data_per_group=50)
    dtr = lgb.Dataset(X[tr], tgt[tr], categorical_feature=cats, free_raw_data=False)
    dho = lgb.Dataset(X[ho], tgt[ho], categorical_feature=cats, reference=dtr)
    b = lgb.train(P, dtr, num_boost_round=30000, valid_sets=[dho], valid_names=['hold'], callbacks=[lgb.log_evaluation(1000), lgb.early_stopping(200, verbose=False)])
    pb = b.predict(Xw, num_iteration=b.best_iteration) + base[win]
    hol = np.sqrt(np.mean((b.predict(X[ho], num_iteration=b.best_iteration) - tgt[ho]) ** 2))
    out = pb[pos]; np.save(fn, out)
    msg = f'{MODE} seed {SEED} best_iter {b.best_iteration} holdout {hol:.4f}'
    if MODE == 'win':
        yw = vw.y.values; o = yw < 7200
        msg += f'  window ordinary (v15 rows) {np.sqrt(np.mean((out[o]-yw[o])**2)):.4f}'
        ref = TC + f'p1c_bulk_fin_s{SEED}.npy'
        if os.path.exists(ref):
            q = np.load(ref)[pos]; msg += f'   vs track C fin s{SEED}: RMS diff {np.sqrt(np.mean((q-out)**2)):.4f} |max| {np.abs(q-out).max():.3f}'
    else:
        msg += f'  ranking prediction median {np.median(out):.1f}  q01 {np.quantile(out,.01):.1f}  q99 {np.quantile(out,.99):.1f}'
    print(msg + f'  [{time.time()-t1:.0f}s]', flush=True)
