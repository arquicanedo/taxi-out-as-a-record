"""s554: a FULL-YEAR ADS-B stage on top of the full-year bulk. v110 still carries 38% of v108, whose ADS-B stage was trained on
ONE month per Jan|Jul fold. Here the stage is trained on month-fold out-of-fold residuals of the full-year bulk
(scripts/oof_folds_v112.sh -> p2p_v55_adsb.py MODE=oof):
  base    B2 = wL L55A + wP P55A, the two full-year members at their covered-row simplex weights renormalised (s551)
  stage   y - B2 on covered ordinary rows (y < 7200, not time-matched), v107's feature set (movement + detection + parking),
          equal halves of LightGBM (v104 settings, 3 seeds) and XGBoost (v107's frozen FLAML settings); trees only (the window
          over-rates a network ~10x, s544)
  window  stage trained on the TEN non-window months (NEST=win bases: no Jan/Jul label anywhere), applied to the window
  scored  stage trained on all TWELVE months (NEST=all bases), applied to 2026
  C2      on v110's covered unfired rows: v100 with the bulk swapped for B2 + stage (stage where B2 < 3600)
Priced against v110 (the incumbent, 228.4607) for C2 and v108 + a (C2 - v108) with a Jan|Jul cross-fitted; rule fixed
before the run: highest G_hat with G/SE > 1. SAVE=1 writes cand/v112_adsb.npz.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_stage_full.py
"""
import os, glob, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_v110_diag.py')).read().split('# ---- the pick, per airport')[0])
import lightgbm as lgb, xgboost as xgb, json
v110w, v110r = fam[pick]; assert pick == 'C alpha'
assert np.array_equal(v110r, I.sub('v110')), 'v110 rebuilt'
print(f'v110 rebuilt from the diag setup; window RMSE {rm(v110w, np.ones(nw, bool)):.4f}')
wL, wP = w_c[names.index('L55A')], w_c[names.index('P55A')]; wL, wP = wL / (wL + wP), wP / (wL + wP)
print(f'base B2 = {wL:.4f} L55A + {wP:.4f} P55A')
TA = f'{P2}/trees_adsb'
def oof(nest):
    ids, p = [], []
    for fi in sorted(glob.glob(f'{TA}/oof_{nest}_m*_ids.npy')):
        tag = fi[len(f'{TA}/oof_'):-len('_ids.npy')]
        ids.append(np.load(fi)); p.append(wL * np.load(f'{TA}/oof_{tag}_pooled_s0.npy') + wP * np.load(f'{TA}/oof_{tag}_perapt_s0.npy'))
    s = pd.Series(np.concatenate(p), index=np.concatenate(ids)); assert s.index.is_unique; return s
BW, BA = oof('win'), oof('all'); print(f'OOF bases: NEST=win {len(BW):,} rows, NEST=all {len(BA):,} rows')
F = pd.read_parquet(os.environ.get('ADSB_FEAT_FULL', f'{REPO}/artifacts/adsb_push_feat_full.parquet')).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
T = pd.concat([pq.read_table(f, columns=['MVT_ID_mvt', 'ADEP_mvt', 'PHASE_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                         'LOBT_flt', 'AOBT_3_flt']).to_pandas() for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))])
T = T[T.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
AC = {a: i for i, a in enumerate(sorted(set(aptw)))}
def X_full(ids, base, mvt, sched, lobt, aobt, ap):
    Fx = F.reindex(ids); tmv = sec(mvt); ms = cl(tmv - sec(sched)); lb = tmv - sec(lobt); nv = tmv - sec(aobt)
    X = pd.DataFrame({'d_ab': Fx.ab_taxi.values - base, 'ab_taxi': Fx.ab_taxi.values, 'base': base, 'd_naive': nv - base, 'd_ms': ms - base,
                      'd_lb': lb - base, 'how': Fx.ab_how.values, 'dwell': Fx.ab_dwell.values, 'd0': Fx.ab_d0.values, 'dpush': Fx.ab_dpush.values,
                      'n': Fx.ab_n.values, 'gap': Fx.ab_gap.values, 'stop': Fx.ab_stop.values, 'gs0': Fx.ab_gs0.values, 'dto': Fx.ab_dto.values,
                      'apt': np.array([AC[a] for a in ap]), 'tm': Fx.ab_tm.values, 'path': Fx.ab_path.values, 'tmove': Fx.ab_tmove.values,
                      'hold': Fx.ab_hold.values, 'pk_dist': Fx.ab_pk_dist.values, 'pk_age': Fx.ab_pk_age.values, 'pk_gs': Fx.ab_pk_gs.values,
                      'taxi_pk': Fx.ab_taxi_pk.values})
    X['d_ab_pk'] = X.taxi_pk.values - base
    return X, Fx.ab_push.notna().values
PP = json.load(open(f'{REPO}/artifacts/v107_xgb_params.json')); COLS = PP['cols']
LP = dict(objective='l2', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
          lambda_l2=10.0, verbose=-1, num_threads=12, deterministic=True, force_col_wise=True)
def stage(B, Xa, Xb):
    """train on the rows of T whose OOF base is B, predict the frame Xb (same columns); returns the half/half stage delta"""
    tt = T.reindex(B.index); Xt, cov = X_full(B.index.values, B.values, tt.MVT_TIME_UTC_mvt, tt.SCHED_TIME_UTC_mvt, tt.LOBT_flt, tt.AOBT_3_flt, tt.ADEP_mvt.values)
    yt = tt.TAXITIME_SEC_mvt.values.astype(float); fit = cov & (yt < 7200) & (Xt.tm.values != 1) & np.isfinite(yt)
    r_ = yt - B.values; print(f'  stage fit rows {fit.sum():,} of {len(B):,} (coverage {cov.mean():.1%})', flush=True)
    Xl = Xt.loc[fit, COLS].copy(); Xl['apt'] = pd.Categorical(Xl.apt, categories=range(len(AC))); Xp = Xb[COLS].copy(); Xp['apt'] = pd.Categorical(Xp.apt, categories=range(len(AC)))
    dl = np.mean([lgb.train({**LP, 'seed': s}, lgb.Dataset(Xl, r_[fit]), 400).predict(Xp) for s in (0, 1, 2)], 0)
    q = dict(PP['fit_all']); q.update(n_jobs=12, random_state=0, verbosity=0); q.pop('missing', None)
    dx = xgb.XGBRegressor(**q).fit(Xt.loc[fit, COLS].astype(float), r_[fit]).predict(Xb[COLS].astype(float))
    print(f'  stage corr LGB-XGB on the applied frame {np.corrcoef(dl, dx)[0, 1]:.3f}', flush=True)
    return 0.5 * dl + 0.5 * dx
Wt = pd.concat([pq.read_table(f'{D}/training_2025-01-01_2025-02-01.parquet', columns=T.reset_index().columns.tolist()).to_pandas(),
                pq.read_table(f'{D}/training_2025-07-01_2025-08-01.parquet', columns=T.reset_index().columns.tolist()).to_pandas()])
Wt = Wt[Wt.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ctx['ids']); assert np.allclose(Wt.TAXITIME_SEC_mvt.values, y)
B2w = wL * MW['L55A'] + wP * MW['P55A']; B2r = wL * MR['L55A'] + wP * MR['P55A']
Xw, covw = X_full(ctx['ids'], B2w, Wt.MVT_TIME_UTC_mvt, Wt.SCHED_TIME_UTC_mvt, Wt.LOBT_flt, Wt.AOBT_3_flt, aptw)
Rr = pq.read_table(f'{D}/ranking.parquet', columns=T.reset_index().columns.tolist()).to_pandas(); Rr = Rr[Rr.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
Xr2, covr = X_full(sub_ids, B2r, Rr.MVT_TIME_UTC_mvt, Rr.SCHED_TIME_UTC_mvt, Rr.LOBT_flt, Rr.AOBT_3_flt, Rapt)
print('window stage (ten non-window months):'); SW = stage(BW, None, Xw)
print('scored stage (twelve months):'); SR = stage(BA, None, Xr2)
gw, gr = awc & covw & (B2w < 3600), arc & covr & (B2r < 3600)
newbW = B2w + np.where(gw, SW, 0.0); newbR = B2r + np.where(gr, SR, 0.0)
C2w = v8w.copy(); C2w[awc] = cl(np.rint(v100w[awc] + (newbW - M0w @ wF)[awc])); C2r = v8r.copy(); C2r[arc] = cl(np.rint(v100r[arc] + (newbR - M0r @ wF)[arc]))
for nm, m in (('covered ordinary', O & hasw), ('covered ordinary unfired', O & awc), ('all rows', np.ones(nw, bool))):
    print(f'  {nm:26s} v108 {rm(v8w, m):8.2f}  v110 {rm(v110w, m):8.2f}  C (s551) {rm(nWc, m):8.2f}  C2 {rm(C2w, m):8.2f}   pure B2 {rm(cl(B2w + SHW), m):8.2f}  B2+stage {rm(cl(newbW + SHW), m):8.2f}')
fam2 = {'C2': (C2w, C2r)}
dW_, dR_ = C2w - v8w, C2r - v8r; e = v8w - y; acf = np.zeros(nw); a_in = float(-(e * dW_).sum() / (dW_ ** 2).sum())
for fit, app in ((L, J), (J, L)): acf[app] = float(-(e * dW_)[fit].sum() / (dW_[fit] ** 2).sum())
print(f'  alpha in-sample {a_in:.3f}; Jan-fit {acf[L][0]:.3f} -> Jul, Jul-fit {acf[J][0]:.3f} -> Jan')
fam2['C2 alpha'] = (cl(np.rint(v8w + acf * dW_)), cl(np.rint(v8r + a_in * dR_)))
res2 = {}
for tag, (aW, aR) in fam2.items():
    lin = -2.0 * (v110w - y) * (aW - v110w); own = (v110w - y) ** 2 - (aW - y) ** 2; Qb = float(((aR - v110r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
    cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own)
    SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc; res2[tag] = (G, SE)
    print(f'  vs v110  {tag:9s} own {own.sum()*sc:+.4e} ({own.sum()*sc/SEo:+.2f} SE)  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  G/SE {G/SE:+.2f}'
          f'  -> {np.sqrt(228.4607**2 - G/344841):.4f}  rows R {int((aR != v110r).sum()):,}')
    print('      per airport window own: ' + '  '.join(f'{a} {own[aptw == a].sum()*sc:+.1e}' for a in sorted(pd.unique(aptw))))
ok = {k: v for k, v in res2.items() if v[0] / v[1] > 1}; p2 = max(ok, key=lambda k: ok[k][0]) if ok else None
print(f'RULE (highest G_hat with G/SE > 1) -> {p2}')
if os.environ.get('PIN_PICKS') == '1':   # final phase: keep the variant the clean chain shipped
    p2 = os.environ.get('PIN_V112', 'C2 alpha'); assert p2 in fam2, p2; print(f'PINNED -> {p2}')
if os.environ.get('SAVE') == '1' and p2:
    np.savez(f'{I.CAND}/v112_adsb.npz', ids=sub_ids, v110=v110r, new=fam2[p2][1], win_ids=ctx['ids'], newW=fam2[p2][0], variant=p2, wLP=np.array([wL, wP]))
    print(f'saved cand/v112_adsb.npz as "{p2}"')
