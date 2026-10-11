"""Experiments on the v102 ADS-B stage, Jan|Jul cross-fitted on the window, each priced two ways:
  window   SSE removed on the window (all rows), day-cluster SE
  scored   per airport: window linear gain PER COVERED ROW x 2026 covered rows, minus the exact Q_b of the variant's
           2026 change (s517: the Q-anchored G_hat is a lower bound for a stage whose coverage differs by year)
All variants are compared against v100; the v102 line is the reference.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm python scripts/adsb_stage_exp.py [variant ...]
REPRODUCING s518's table (feature table via ADSB_FEAT; the harness adds tm / detection columns when the table has them):
  base fired allfit cap nb nbx     ADSB_FEAT=artifacts/adsb_push_feat_v102.parquet
  cs_all cs_all_lo lowcap lowcap2 nr200
                                   ADSB_FEAT=artifacts/adsb_push_feat_v103.parquet ADSB_NO_DET=1
  det det_norwy                    ADSB_FEAT=artifacts/adsb_push_feat_v103.parquet
  (s518's base/allfit on the time-matched table: ADSB_FEAT=...v103.parquet ADSB_NO_DET=1, variants base allfit)
"""
import sys, os
_argv = sys.argv[1:]; sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adsb_price.py')).read().split("print(f'\\nCANDIDATES")[0])
if 'ab_tm' in F.columns: X['tm'] = Fw.ab_tm.values
RW = pq.read_table(f'{D}/training_2025-01-01_2025-02-01.parquet', columns=['MVT_ID_mvt', 'RUNWAY_mvt']).to_pandas()
RW = pd.concat([RW, pq.read_table(f'{D}/training_2025-07-01_2025-08-01.parquet', columns=['MVT_ID_mvt', 'RUNWAY_mvt']).to_pandas(),
                pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'RUNWAY_mvt']).to_pandas()]).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').RUNWAY_mvt
RWC = {k: i for i, k in enumerate(sorted(set(RW.dropna().astype(str))))}
EXTRA = [c for c in ('ab_path', 'ab_tmove', 'ab_hold', 'ab_pk_dist', 'ab_pk_age', 'ab_pk_gs', 'ab_taxi_pk', 'ab_px', 'ab_py', 'ab_qx', 'ab_qy') if c in F.columns and os.environ.get('ADSB_NO_DET') != '1']   # ADSB_NO_DET=1: s518's pre-detection runs
if EXTRA:
    for c in EXTRA: X[c[3:]] = Fw[c].values
    if 'ab_taxi_pk' in EXTRA: X['d_ab_pk'] = X['taxi_pk'].values - X['base'].values
    X['rwy'] = pd.Categorical([RWC.get(str(v), -1) if isinstance(v, str) else -1 for v in RW.reindex(ctx['ids']).values], categories=range(-1, len(RWC)))
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                                  'LOBT_flt', 'AOBT_3_flt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
tm = sec(R.MVT_TIME_UTC_mvt); msr = cl(tm - sec(R.SCHED_TIME_UTC_mvt)); lbr = tm - sec(R.LOBT_flt); nvr = tm - sec(R.AOBT_3_flt)
aptr = R.ADEP_mvt.values.astype(str)
Fr = F.reindex(sub_ids); hasr = Fr.ab_push.notna().values
Xr = X_of(Fr, v100r, nvr, aptr, msr, lbr)
if 'ab_tm' in F.columns: Xr['tm'] = Fr.ab_tm.values
if EXTRA:
    for c in EXTRA: Xr[c[3:]] = Fr[c].values
    if 'ab_taxi_pk' in EXTRA: Xr['d_ab_pk'] = Xr['taxi_pk'].values - Xr['base'].values
    Xr['rwy'] = pd.Categorical([RWC.get(str(v), -1) if isinstance(v, str) else -1 for v in RW.reindex(sub_ids).values], categories=range(-1, len(RWC)))
firew = ~appw; firer = np.abs(v100r - msr) <= 60
def FOLDS(fit_w, app_w):
    """ADSB_FOLDS=janjul (default: fit one month, score the other) or day6 (6 folds by calendar day, both months in training,
    which is how the scored side is trained; s524)."""
    if os.environ.get('ADSB_FOLDS', 'janjul') == 'day6':
        dys = np.array(sorted(set(clus))); prm_ = np.random.default_rng(0).permutation(len(dys))
        fd = pd.Series(np.arange(len(dys)) % 6, index=dys[prm_]).reindex(clus).values
        return [(fit_w & (fd != k), app_w & (fd == k)) for k in range(6)]
    return [(fit_w & L, app_w & J), (fit_w & J, app_w & L)]
ONE = 1.7717e8; SEEDS = tuple(range(int(os.environ.get('ADSB_NSEEDS', '3'))))   # v102/v103: 3
def run(name, cols=None, prm=None, nr=None, app_w=None, app_r=None, fit_w=None):
    cols = cols or list(X.columns); prm = {**PRM, **(prm or {})}; nr = nr or NR
    app_w = has & appw if app_w is None else app_w; app_r = hasr & ~firer if app_r is None else app_r
    fit_w = app_w & (y < 7200) if fit_w is None else fit_w
    dW = np.zeros(nw)
    for f_, a_ in FOLDS(fit_w, app_w):
        dW[a_] = np.mean([lgb.train({**prm, 'seed': s}, lgb.Dataset(X.loc[f_, cols], rres[f_]), nr).predict(X.loc[a_, cols]) for s in SEEDS], 0)
    dR = np.zeros(len(sub_ids))
    dR[app_r] = np.mean([lgb.train({**prm, 'seed': s}, lgb.Dataset(X.loc[fit_w, cols], rres[fit_w]), nr).predict(Xr.loc[app_r, cols]) for s in SEEDS], 0)
    nW = v100w.copy(); nW[app_w] = cl(np.rint(v100w[app_w] + dW[app_w])); nR = v100r.copy(); nR[app_r] = cl(np.rint(v100r[app_r] + dR[app_r]))
    c = (v100w - y) ** 2 - (nW - y) ** 2; lin = -2.0 * (v100w - y) * (nW - v100w)
    cs = np.bincount(clus, weights=c); rng = np.random.default_rng(5); k = len(cs)
    se = float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(2000)]).std())
    tot = 0.0; per = []
    for a in AC:
        mw, mr = app_w & (apt == a), app_r & (aptr == a)
        if not mw.any(): continue
        g = float(lin[mw].sum()) * sc / mw.sum() * mr.sum() - float(((nR - v100r) ** 2)[aptr == a].sum()); tot += g; per.append(f'{a} {g/ONE:+.2f}')
    print(f'{name:<22} window {c.sum():+.4e} (SE {se:.2e}, {c.sum()*sc/ONE:+.3f} bs)  ord {c[O].sum():+.3e} ext {c[ext].sum():+.3e}   '
          f'SCORED {tot:+.4e} = {tot/ONE:+.3f} bs   rows W {int((nW!=v100w).sum()):,} R {int((nR!=v100r).sum()):,}', flush=True)
    print('        ' + '  '.join(per), flush=True)
    return dict(dW=dW, dR=dR, nR=nR, nW=nW, G=c.sum(), S=tot)
V = {}
want = _argv or ['base', 'fired', 'allfit', 'cap']
if 'base' in want:  V['base'] = run('v102 (reference)')
if 'fired' in want: V['fired'] = run('+ copy-fired rows', app_w=has.copy(), app_r=hasr.copy(), fit_w=has & appw & (y < 7200))
if 'allfit' in want: V['allfit'] = run('+ fired, fit on them', app_w=has.copy(), app_r=hasr.copy(), fit_w=has & (y < 7200))
if 'cap' in want:   V['cap'] = run('capacity 31/100/800', prm=dict(num_leaves=31, min_data_in_leaf=100), nr=800)
if 'cap2' in want:  V['cap2'] = run('capacity 63/50/1500 lr.02', prm=dict(num_leaves=63, min_data_in_leaf=50, learning_rate=0.02), nr=1500)
# ---------------- neighbour features: other departures' ADS-B taxi relative to v100, same airport, +-H s ---------------
def nb_feats(t, ap, dab, H):
    """mean (clipped +-1500) and count of other rows' d_ab within +-H seconds at the same airport; label-free"""
    mean = np.full(len(t), np.nan); cnt = np.zeros(len(t))
    v = np.clip(np.nan_to_num(dab, nan=0.0), -1500, 1500); ok = ~np.isnan(dab)
    for a in np.unique(ap):
        ix = np.where(ap == a)[0]; o = ix[np.argsort(t[ix])]; ts = t[o]
        cs_v = np.r_[0, np.cumsum(v[o] * ok[o])]; cs_n = np.r_[0, np.cumsum(ok[o])]
        lo = np.searchsorted(ts, ts - H, 'left'); hi = np.searchsorted(ts, ts + H, 'right')
        s = cs_v[hi] - cs_v[lo] - v[o] * ok[o]; n = cs_n[hi] - cs_n[lo] - ok[o]
        cnt[o] = n; mean[o] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return mean, cnt
tw = sec(tr.MVT_TIME_UTC_mvt)
for H in ((900, 1800) if ('nb' in want or 'nbx' in want) else ()):   # only when asked: they measured negative (s518)
    X[f'nb{H}'], X[f'nbn{H}'] = nb_feats(tw, apt, X.d_ab.values, H)
    Xr[f'nb{H}'], Xr[f'nbn{H}'] = nb_feats(tm, aptr, Xr.d_ab.values, H)
if 'nb' in want or 'nbx' in want:
  nbw = (X.nbn1800.values > 0); nbr = (Xr.nbn1800.values > 0)
  print(f'neighbour coverage: window rows with ADS-B or neighbours {np.mean(has | nbw):.1%} (own ADS-B {has.mean():.1%}); '
      f'2026 {np.mean(hasr | nbr):.1%} (own {hasr.mean():.1%})', flush=True)
if 'nb' in want:
    V['nb'] = run('+ fired + neighbours', app_w=has.copy(), app_r=hasr.copy(), fit_w=has & (y < 7200))
if 'nbx' in want:
    V['nbx'] = run('+ neighbours, all rows', app_w=has | nbw, app_r=hasr | nbr, fit_w=(has | nbw) & (y < 7200))
if 'tm' in X.columns:
    tmw_ = X.tm.values == 1
    if 'cs_all' in want:
        V['cs_all'] = run('fit callsign, apply all', app_w=has.copy(), app_r=hasr.copy(), fit_w=has & ~tmw_ & (y < 7200))
    if 'cs_all_lo' in want:
        V['cs_all_lo'] = run('... v100 < 3600 only', app_w=has & (v100w < 3600), app_r=hasr & (v100r < 3600), fit_w=has & ~tmw_ & (y < 7200))
    base3 = dict(app_w=has & (v100w < 3600), app_r=hasr & (v100r < 3600), fit_w=has & ~tmw_ & (y < 7200))
    if 'lowcap' in want: V['lowcap'] = run('... leaves 7 / min 400', prm=dict(num_leaves=7, min_data_in_leaf=400), **base3)
    if 'lowcap2' in want: V['lowcap2'] = run('... leaves 15 / min 800', prm=dict(min_data_in_leaf=800), **base3)
    if 'det' in want: V['det'] = run('+ path/tmove/hold/rwy', **base3)
    if 'det_norwy' in want: V['det_norwy'] = run('+ path/tmove/hold', cols=[c for c in X.columns if c != 'rwy'], **base3)
    if 'nr200' in want: V['nr200'] = run('... 200 rounds', nr=200, **base3)
    if 'relax' in want: V['relax'] = run('v103 stage (table as given)', cols=[c for c in X.columns if c != 'rwy'], **base3)
    base4 = dict(cols=[c for c in X.columns if c != 'rwy'], **base3)
    if 'v104' in want: V['v104'] = run('v104 stage (reference)', **base4)
    if 'huber' in want: V['huber'] = run('v104, huber 300', prm=dict(objective='huber', alpha=300.0), **base4)
    if 'huber150' in want: V['huber150'] = run('v104, huber 150', prm=dict(objective='huber', alpha=150.0), **base4)
    if 'clean' in want:   # train without rows whose record contradicts ADS-B by > 30 min (stale records); apply everywhere
        b_ = dict(base4); b_['fit_w'] = base4['fit_w'] & ~(np.abs(y - X.ab_taxi.values) > 1800)
        V['clean'] = run('v104, fit w/o |y-ab_taxi|>1800', **b_)
    if 'oper' in want or 'oper_wtc' in want:
        cols_ = ['MVT_ID_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt', 'WK_TBL_CAT_flt']
        OP = pd.concat([pq.read_table(f, columns=cols_).to_pandas() for f in (f'{D}/training_2025-01-01_2025-02-01.parquet',
             f'{D}/training_2025-07-01_2025-08-01.parquet', f'{D}/ranking.parquet')]).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
        opn = OP.AIRCRAFT_OPERATOR_flt.fillna(OP.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})', expand=False)).fillna('?')
        top = opn.reindex(ctx['ids'])[has].value_counts().index[:60]; OC = {k: i for i, k in enumerate(top)}
        enc = lambda s: pd.Categorical([OC.get(v, -1) for v in s], categories=range(-1, len(OC)))
        X['oper'] = enc(opn.reindex(ctx['ids']).values); Xr['oper'] = enc(opn.reindex(sub_ids).values)
        WC = {'L': 0, 'M': 1, 'H': 2, 'J': 3}
        X['wtc'] = OP.WK_TBL_CAT_flt.reindex(ctx['ids']).map(WC).values; Xr['wtc'] = OP.WK_TBL_CAT_flt.reindex(sub_ids).map(WC).values
        c0 = [c for c in X.columns if c not in ('rwy', 'wtc')]
        if 'oper' in want: V['oper'] = run('v104 + operator', **{**base4, 'cols': c0})
        if 'oper_wtc' in want: V['oper_wtc'] = run('v104 + operator + wake class', **{**base4, 'cols': c0 + ['wtc']})
    if 'xy' in want: V['xy'] = run('v104 + parked/first-sighting position', **base4)
    if 'noxy' in want: V['noxy'] = run('v104 (xy table, xy columns dropped)', **{**base4, 'cols': [c for c in base4['cols'] if c not in ('px', 'py', 'qx', 'qy')]})
    if any(w.startswith('lc') for w in want):   # s525: learning curve, training days subsampled inside every fold
        dys_ = np.array(sorted(set(clus)))
        for frac in (0.25, 0.5, 1.0):
            if f'lc{int(frac*100)}' not in want: continue
            keep = np.isin(clus, np.random.default_rng(7).permutation(dys_)[:int(round(frac * len(dys_)))])
            V[f'lc{int(frac*100)}'] = run(f'v104 stage, {int(frac*100)}% of training days', **{**base4, 'fit_w': base4['fit_w'] & keep})
