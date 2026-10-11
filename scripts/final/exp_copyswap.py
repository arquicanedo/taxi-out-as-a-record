"""The interim experiment (analysis/FINAL-PLAN.md, "The interim experiment"): one Jan/Jul vector S that differs from the
scored v127 (247.7412) only in where the copy weight comes from. Nothing here submits anything.

THE SWAP RULE (fixed before any 2026 quantity was computed)
  gate      the clean base's learned five-class gate, Q = mixq(P0..P4) (P^0.90, renormalised, branches 1-4 below 0.10
            zeroed), branch values V = [b, vsd, vlb, clip(86400 + b), b] (scripts/clean_base_build.py)
  copy wt   gate_c = Q1 (+ Q2 where LOBT is null, since vlb = vsd there)
  chain P   P(alpha) = (1 - alpha) P23 + alpha P25, the chain's v29 recipe with its leaderboard alpha (0.35) replaced
            by a labelled one. P23 = isotonic fitted IN SAMPLE (classifier scores on its own training rows), P25 =
            isotonic fitted on 3-fold OUT-OF-FOLD scores, both against the chain's own label ext = (y >= 7200)
            (v15_trees.py, v25_build.py, calib3.py)
  swap      on rows that are NOT hits (hit = MVT - LOBT >= 7200 with LOBT present): Q1 := P(alpha), Q2 := 0 where LOBT
            is null, every other branch (0, 3, 4, and 2 where LOBT is present) scaled by (1 - c) / (1 - gate_c); no 0.10
            floor on c. Where 1 - gate_c <= 1e-6 (the gate's other branches are only the 1e-12 clip floor) the
            remainder 1 - c goes to the bulk branch 0. Hit rows unchanged. LIRF day-scale rows (ADEP LIRF, LOBT null,
            MVT - SCHED >= 20,000 s) held at the gate's values. The base construction after the mixture is unchanged
            (B1 - B0 added where |inc - vsd| > 60).

MODES
  window   steps 1-3: P23_w / P25_w on the 2025 Jan|Jul window (calib_scores.npz), the window base rebuilt and checked
           against clean/base_window.npz, alpha on a 0.05 grid (January-fitted on July, July-fitted on January, pooled),
           the window replay at the pooled alpha with a day-clustered bootstrap, D^2-weighted precision.
  bases    step 4a: the scored-side gate base rebuilt and checked against the v116 run's v100 (sha 00bf453e...), P23 and
           P25 recovered on the scored rows (P23 from v15_trees_jul19, checked by rebuilding v23 byte for byte; P25 by
           inverting the archived v25 against the same bulk), the swapped base written as quick-boat_v100.parquet into
           a copy of the v116 run (clean_S); an unswapped copy (clean_U) is the control.
  tail D   step 4b: regen_clean.sh SKIP_BASE=1 PIN_PICKS=1 WF=archived CLEAN_DIR=D, v117 = v114, then v120..v126 in CLEAN
           mode exactly as scripts/v127_build.sh runs them (without its copy into submissions/).
  final    steps 5-7: S = clean_S's v126 with the LIRF day-scale rows set to v127's values and the 5% single-row mass
           cap; the deployed-level instrument's prediction; S.parquet and prereg.json. Asserts S's sha256 once recorded.
  all      window, bases, tail clean_U (must reproduce v127's sha256), tail clean_S, final.

  uv run --quiet --with pandas --with numpy --with pyarrow --with lightgbm --with scikit-learn \\
      python scripts/final/exp_copyswap.py all
Outputs go to $EXP_DIR (required). Reads $DC26_WORK and the 2026-09-04 extract; writes neither.
"""
import hashlib, json, os, shutil, subprocess, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

for k in ('CLEAN', 'DC26_SUBDIR', 'DC26_CAND', 'DC26_WF'): os.environ.pop(k, None)   # the instrument reads no scored chain
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f'{REPO}/scripts'); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
assert I.W == W and I.DATA == D
P2 = f'{W}/session-20260922/p2P'; SRC = f'{W}/clean'
EXP = os.environ['EXP_DIR']
for bad in (W, f'{REPO}/submissions', *[os.environ[k] for k in ('FINAL_WORK',) if os.environ.get(k)]):
    assert not os.path.realpath(EXP).startswith(os.path.realpath(bad)), f'EXP_DIR {EXP} inside {bad}'
os.makedirs(EXP, exist_ok=True)
FLOOR, CAP, DAY, THR, N_B = 60., 200000., 86400., 7200., 344841
DS_MS = 20000.                                    # LIRF day-scale population (scripts/final/lirf_dayscale.py)
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
CAP_SHARE = 0.05                                  # single-row mass cap, fixed in FINAL-PLAN.md before S existed
SHA = {'v116': 'a460c540c9d36e7930b2695f8b1c30533b70bcba5c21de14a8baa1884257a9c7',
       'v115': '00bf453efc80d6d5805332ef81c3becde0d28da0e8ad7f35a2ad622009ebf8e4',
       'v127': '74fcf296d1ee90b08cfa6ce7b7356a8b82143ccf51dd2b87e1754925938f4d12',
       'v23': '3f3c616e1cd88c07772d5560e7ed0d5356c4cb69bd0e076d967f94139370bafa',
       'v25': '7c12d8c02c8b724615d3b0d1a75d27af88dc1c66c7439e70acebe79c7a1eb032',
       'gate_rk': '2374a269a107afd88b72d50a5d09aba30adc8775ec0a018ac09193b6c66dcfd6',
       'S_base': '079cb28f4dfa16370bc65a04e371df20ec98dfe5c5c847576bd26fa11dd9d57e',    # clean_S v100, recorded 2026-10-08
       'S_v126': '597eb2b368afdcac94676e359ce2ae8a975e9adb42e1c5d45f61a823e07cb6cb'}    # clean_S v126 (before step 5)
SHA_S = os.environ.get('SHA_S', '11fe7fc77ad067aeed80e77a3542846620c73eb49ab0e7f8f77f91d29711405c')   # recorded from the first run (2026-10-08)
V127_SCORE, V127_PREREG = 247.7412, 246.41
cl = lambda p: np.clip(p, FLOOR, CAP)
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
E0 = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E0).dt.total_seconds().values
mean_of = lambda ps: np.mean([np.load(p) for p in ps], 0)
FOUR = ('L55', 'P55', 'K32', 'K128'); w0 = np.array([.2230, .2030, .0880, .4860])
WIN = {'L55': [f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)], 'P55': [f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)],
       'K32': [f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)], 'K128': [f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)],
       'ID': [f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)], 'RQ': [f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)],
       'CL': [f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)], 'L55o': [f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)],
       'P55o': [f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)]}


def mixq(P):
    Q = P ** 0.90; Q = Q / Q.sum(1, keepdims=True); Q[:, 1:][Q[:, 1:] < 0.10] = 0.0; Q[:, 0] = 1.0 - Q[:, 1:].sum(1)
    Q = np.clip(Q, 1e-12, None); return Q / Q.sum(1, keepdims=True)


def mixv(Q, bulk, vsd, vlb):
    b = cl(bulk); V = np.stack([b, vsd, vlb, cl(DAY + b), b], 1); return cl(np.rint((Q * V).sum(1)))


def base(Q, vsd, vlb, B0, B1):
    inc = mixv(Q, B0, vsd, vlb); app = np.abs(inc - vsd) > 60
    out = inc.copy(); out[app] = cl(np.rint(inc[app] + (B1 - B0)[app])); return out


def swapQ(Q, c, hit, lnull, hold):
    """The swap rule of the docstring. Q (n, 5) the gate's mixture weights, c (n,) the chain's P(alpha)."""
    gate_c = Q[:, 1] + np.where(lnull, Q[:, 2], 0.0); r = 1.0 - gate_c
    deg = r <= 1e-6
    s = np.where(deg, 0.0, (1.0 - c) / np.where(deg, 1.0, r))
    N = Q * s[:, None]
    N[:, 1] = c
    N[:, 2] = np.where(lnull, 0.0, Q[:, 2] * s)
    N[deg, 0] = 1.0 - c[deg]; N[deg, 2:] = 0.0
    m = ~hit & ~hold
    out = Q.copy(); out[m] = N[m]
    assert np.allclose(out.sum(1), 1.0, atol=1e-9) and (out >= -1e-15).all()
    return out


def iso(x, lab, xq):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds='clip').fit(x, lab.astype(float)).predict(xq)


# ------------------------------------------------------------------------------------------------ window (steps 1-3)
def window_ctx():
    """The deployed-level window (p1a_instrument.window), the clean gate base rebuilt on it, P23_w / P25_w, hit, hold."""
    ctx = I.window(); y, ext = ctx['y'], ctx['ext']; O = ~ext; mon = ctx['month'].astype(int); ids = ctx['ids']
    fp = np.load(f'{P2}/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
    wz = np.load(f'{SRC}/wF.npz'); assert [str(k) for k in wz['members']] == FULL and np.array_equal(wz['w'], wF), 'v116 run used WF=archived'
    Mw = {k: mean_of(WIN[k]) for k in FULL}
    B0 = np.stack([Mw[k] for k in FOUR], 1) @ w0; B1 = np.stack([Mw[k] for k in FULL], 1) @ wF
    off = {m: float((y - B1)[O & (mon == m)].mean()) for m in (1, 7)}
    shw = np.where(mon == 1, off[1], off[7])
    gateW = base(ctx['Qg'], ctx['vsd'], ctx['vlb'], B0 + shw, B1 + shw)
    bw = np.load(f'{SRC}/base_window.npz')
    assert np.array_equal(bw['win_ids'], ids), 'window row order'
    assert np.array_equal(bw['newW'], gateW), 'window gate base does not reproduce clean/base_window.npz newW'
    # labels, hit and the day-scale rows from the 2025 files by movement id
    fs = [f'{D}/training_2025-01-01_2025-02-01.parquet', f'{D}/training_2025-07-01_2025-08-01.parquet']
    T = pd.concat([pq.read_table(f, columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                             'LOBT_flt']).to_pandas() for f in fs])
    T = T[T.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
    assert T.ADEP_mvt.notna().all()
    ms = sec(T.MVT_TIME_UTC_mvt) - sec(T.SCHED_TIME_UTC_mvt); lb = sec(T.MVT_TIME_UTC_mvt) - sec(T.LOBT_flt)
    lnull = np.isnan(lb); hit = ~lnull & (lb >= THR)
    hold = T.ADEP_mvt.eq('LIRF').values & lnull & (ms >= DS_MS)
    assert np.array_equal(ctx['vsd'], cl(ms)), 'window vsd = clip(MVT - SCHED)'
    assert np.array_equal(ctx['vlb'][lnull], ctx['vsd'][lnull]) and np.array_equal(ctx['vlb'][~lnull], cl(lb[~lnull])), 'window vlb'
    # the chain's classifier on the window: Pv (trained on 2025 minus Jan/Jul), Pin / Poof on its training rows
    z = np.load(f'{W}/gate_stage_a2.npz'); cs = np.load(f'{W}/calib_scores.npz')
    vid = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
    pos = pd.Series(np.arange(len(vid)), index=vid).reindex(ids).values.astype(int)
    assert np.array_equal(z['yv'][pos], y)
    lbv = z['lb_v'][pos]; assert np.array_equal(np.isnan(lbv), lnull) and np.array_equal(~np.isnan(lbv) & (lbv >= THR), hit), 'hit'
    ty = pq.read_table(f'{W}/night_tr_all.parquet', columns=['y']).to_pandas().y.values.astype(float)
    assert np.array_equal(ty, z['yt']) and len(cs['Pin']) == len(ty)
    lab = z['yt'] >= THR                                       # the chain's label: ext = y >= 7200
    Pv = cs['Pv'][pos]
    P23 = iso(cs['Pin'], lab, Pv); P25 = iso(cs['Poof'], lab, Pv)
    return dict(ctx=ctx, y=y, ext=ext, mon=mon, apt=ctx['apt'].astype(str), clus=ctx['cluster'], gateW=gateW, B0=B0 + shw,
                B1=B1 + shw, hit=hit, lnull=lnull, hold=hold, P23=P23, P25=P25, off=off, ms=ms, n_train=len(ty),
                n_train_ext=int(lab.sum()))


def boot(c, clus, B=2000, seed=0):
    cs = np.bincount(clus, weights=c); k = len(cs); rng = np.random.default_rng(seed)
    bs = np.array([cs[rng.integers(0, k, k)].sum() for _ in range(B)])
    return float(bs.std()), [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], k


def run_window():
    w = window_ctx(); y, mon = w['y'], w['mon']; ctx = w['ctx']
    swapW = lambda a: base(swapQ(ctx['Qg'], (1 - a) * w['P23'] + a * w['P25'], w['hit'], w['lnull'], w['hold']),
                           ctx['vsd'], ctx['vlb'], w['B0'], w['B1'])
    e0 = (w['gateW'] - y) ** 2; J, L = mon == 1, mon == 7
    print(f'window: {len(y):,} rows, {int(w["ext"].sum())} extremes; gate base RMSE {np.sqrt(e0.mean()):.4f} (reproduces clean/base_window.npz newW)')
    print(f'  copy label: ext = y >= 7200 (the chain\'s); training rows {w["n_train"]:,}, positives {w["n_train_ext"]:,}')
    print(f'  hit rows {int(w["hit"].sum())}, LIRF day-scale rows held {int(w["hold"].sum())}')
    tab = []
    for a in GRID:
        e1 = (swapW(a) - y) ** 2; d = e1 - e0
        tab.append(dict(alpha=float(a), dSSE=float(d.sum()), dSSE_jan=float(d[J].sum()), dSSE_jul=float(d[L].sum())))
    T = pd.DataFrame(tab); print(T.to_string(index=False, float_format=lambda v: f'{v:+.4e}'))
    aJ = float(T.alpha[T.dSSE_jan.idxmin()]); aL = float(T.alpha[T.dSSE_jul.idxmin()]); aP = float(T.alpha[T.dSSE.idxmin()])
    xJ = float(T.dSSE_jul[T.alpha == aJ].iloc[0]); xL = float(T.dSSE_jan[T.alpha == aL].iloc[0])
    print(f'cross-fit: alpha fitted on January {aJ:.2f} -> July dSSE {xJ:+.4e};  fitted on July {aL:.2f} -> January dSSE {xL:+.4e};  '
          f'pooled alpha {aP:.2f}')
    # ---- step 3: the replay at the pooled alpha
    S = swapW(aP); e1 = (S - y) ** 2; c = e1 - e0; sc = N_B / len(y)
    se, ci, k = boot(c, w['clus'])
    O, X = ~w['ext'], w['ext']
    byapt = {a: dict(n_moved=int(((S != w['gateW']) & (w['apt'] == a)).sum()), dSSE=float(c[w['apt'] == a].sum()))
             for a in sorted(set(w['apt']))}
    Dd = S - w['gateW']; big = np.abs(Dd) > 600; D2 = Dd ** 2
    right = np.abs(S - y) < np.abs(w['gateW'] - y)
    prec = float(D2[big & right].sum() / D2[big].sum()) if big.any() else float('nan')
    rep = dict(alpha_grid=tab, alpha_jan=aJ, alpha_jul=aL, alpha_pooled=aP, xfit_jul_dSSE_at_alpha_jan=xJ,
               xfit_jan_dSSE_at_alpha_jul=xL,
               replay=dict(alpha=aP, dSSE=float(c.sum()), SE=se, ci95_pct=ci, days=k, dSSE_ord=float(c[O].sum()),
                           dSSE_ext=float(c[X].sum()), rows_moved=int((Dd != 0).sum()), rows_moved_60=int((np.abs(Dd) > 60).sum()),
                           rows_moved_600=int(big.sum()), by_airport=byapt, rmse_gate=float(np.sqrt(e0.mean())),
                           rmse_swap=float(np.sqrt(e1.mean())), board_scale=sc,
                           d2_precision_600=prec, d2_precision_rows=int(big.sum()), d2_precision_rows_right=int((big & right).sum()),
                           dSSE_on_600_rows=float(c[big].sum())),
               off=w['off'], n_hit=int(w['hit'].sum()), n_hold=int(w['hold'].sum()))
    r = rep['replay']
    print(f'\nREPLAY at alpha {aP:.2f}: window dSSE (swap - gate) {r["dSSE"]:+.4e}, day-cluster SE {se:.3e} ({k} days), '
          f'95% [{ci[0]:+.3e}, {ci[1]:+.3e}];  ordinary {r["dSSE_ord"]:+.4e}  extreme {r["dSSE_ext"]:+.4e}')
    print(f'  RMSE gate {r["rmse_gate"]:.4f} -> swap {r["rmse_swap"]:.4f};  rows moved {r["rows_moved"]:,} (>60 s {r["rows_moved_60"]:,}, '
          f'>600 s {r["rows_moved_600"]:,});  scaled to the 2026 row count x{sc:.5f}: {r["dSSE"] * sc:+.4e}')
    print(f'  D^2-weighted precision on rows |swap - gate| > 600 s: {prec:.4f} ({r["d2_precision_rows_right"]} of {r["d2_precision_rows"]} rows closer to y); '
          f'their dSSE {r["dSSE_on_600_rows"]:+.4e}')
    for a, v in byapt.items(): print(f'    {a}  moved {v["n_moved"]:6,}  dSSE {v["dSSE"]:+.4e}')
    json.dump(rep, open(f'{EXP}/window.json', 'w'), indent=1)
    np.savez(f'{EXP}/window_vectors.npz', ids=w['ctx']['ids'], gateW=w['gateW'], swapW=S, alpha=aP)
    return rep


# ------------------------------------------------------------------------------------------------ scored side (step 4a)
def scored_inputs():
    ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
    R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt',
                                                       'SCHED_TIME_UTC_mvt', 'LOBT_flt']).to_pandas()
    R = R[R.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
    ms = sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt); lb = sec(R.MVT_TIME_UTC_mvt) - sec(R.LOBT_flt)
    vsd = cl(ms); vlb = np.where(np.isnan(lb), vsd, cl(np.nan_to_num(lb, nan=0.0))); lnull = np.isnan(lb)
    hit = ~lnull & (lb >= THR); hold = R.ADEP_mvt.eq('LIRF').values & lnull & (ms >= DS_MS)
    assert sha(f'{W}/b4_gate_rk.parquet') == SHA['gate_rk']
    G = pd.read_parquet(f'{W}/b4_gate_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
    P = G[[f'P{i}' for i in range(5)]].to_numpy(float); assert np.isfinite(P).all()
    fp = np.load(f'{SRC}/wF.npz'); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
    MR = pd.read_parquet(f'{W}/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
    mon = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True).dt.month.values
    off = json.load(open(f'{EXP}/window.json'))['off']; off = {int(k): v for k, v in off.items()}
    sh = pd.Series(mon).map(off).to_numpy(float); assert np.isfinite(sh).all()
    B0 = MR[list(FOUR)].to_numpy(float) @ w0 + sh; B1 = MR[FULL].to_numpy(float) @ wF + sh
    Q = mixq(P); gate = base(Q, vsd, vlb, B0, B1)
    assert sha(f'{SRC}/submissions/quick-boat_v100.parquet') == SHA['v115']
    v100 = pq.read_table(f'{SRC}/submissions/quick-boat_v100.parquet').to_pandas()
    assert np.array_equal(v100.MVT_ID_mvt.values, ids) and np.array_equal(v100.TAXITIME_SEC_mvt.values.astype(float), gate), \
        'scored gate base does not reproduce the v116 run v100'
    # the chain's P23 and P25 on the scored rows
    t = pd.read_parquet(f'{W}/v15_trees_jul19.parquet'); n = pd.read_parquet(f'{W}/v15_net_jul19.parquet')
    rk = pq.read_table(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt', 'apt', 'naive']).to_pandas()
    m = t.merge(n, on='MVT_ID_mvt').merge(rk, on='MVT_ID_mvt', how='left').set_index('MVT_ID_mvt').reindex(ids)
    sd, lbc = m.sched_delay.values, m.mvt_minus_lobt.values
    hitc = ~np.isnan(lbc) & (lbc >= THR); assert np.array_equal(hitc, hit), 'chain hit = ranking hit'
    copy = np.where(hitc, cl(np.where(np.isnan(lbc), sd, lbc)), cl(sd))
    bo = 0.2 * m.lgbm.values + 0.4 * m.mlp.values + 0.4 * m.perapt.values
    Pc23 = m.Pc.values.astype(float)
    Pq = np.maximum(Pc23, hitc * 0.9); pred = cl(Pq * copy + (1 - Pq) * bo); nv = m.naive.values
    edit = (m.apt.values == 'EHAM') & (nv >= 4800) & (pred > 6000) & ~np.isnan(nv); pred = np.where(edit, nv, pred)
    for v in ('v23', 'v25'): assert sha(f'{REPO}/submissions/quick-boat_{v}.parquet') == SHA[v]
    v23 = pq.read_table(f'{REPO}/submissions/quick-boat_v23.parquet').to_pandas()
    v25 = pq.read_table(f'{REPO}/submissions/quick-boat_v25.parquet').to_pandas()
    assert np.array_equal(v23.MVT_ID_mvt.values, ids) and np.array_equal(v25.MVT_ID_mvt.values, ids)
    assert np.array_equal(np.rint(pred), v23.TAXITIME_SEC_mvt.values.astype(float)), 'P23 does not rebuild v23'
    assert not (edit & ~hitc).any(), 'the EHAM edit touches only hit rows'
    # P25 by inverting the archived int32 v25 against the same bulk. Where v25 equals v23 the file cannot tell P25 from
    # P23 (the difference is inside the rounding), so P25 := P23 there; elsewhere the inversion, error <= 0.5 / |copy - bulk|
    a25 = v25.TAXITIME_SEC_mvt.values.astype(float); den = copy - bo
    ok = (a25 != v23.TAXITIME_SEC_mvt.values.astype(float)) & (np.abs(den) > 1.0)
    P25 = np.clip(np.where(ok, (a25 - bo) / np.where(ok, den, 1.0), Pc23), 0.0, 1.0)
    return dict(ids=ids, vsd=vsd, vlb=vlb, lnull=lnull, hit=hit, hold=hold, Q=Q, B0=B0, B1=B1, gate=gate, P23=Pc23, P25=P25,
                inv_ok=ok, den=den, apt=R.ADEP_mvt.values.astype(str), mon=mon, ms=ms)


def write_sub(path, ids, vals, like):
    tpl = pq.read_table(like)
    t = pa.table({'MVT_ID_mvt': pa.array(ids, pa.float64()), 'TAXITIME_SEC_mvt': pa.array(vals.astype('int32'), pa.int32())},
                 schema=tpl.schema)
    assert t.schema.equals(tpl.schema, check_metadata=True); pq.write_table(t, path); return sha(path)


def run_bases():
    rep = json.load(open(f'{EXP}/window.json')); a = rep['alpha_pooled']
    s = scored_inputs(); nh = ~s['hit']
    c = (1 - a) * s['P23'] + a * s['P25']
    Qs = swapQ(s['Q'], c, s['hit'], s['lnull'], s['hold'])
    new = base(Qs, s['vsd'], s['vlb'], s['B0'], s['B1'])
    d = new - s['gate']
    print(f'scored gate base reproduces the v116 run v100 (sha {SHA["v115"][:8]}); P23 rebuilds v23 byte for byte')
    print(f'  P25 inverted from v25 on the {int((s["inv_ok"] & nh).sum()):,} non-hit rows where v25 != v23; P25 := P23 on the other '
          f'{int((~s["inv_ok"] & nh).sum()):,} of {int(nh.sum()):,}')
    print(f'  alpha {a:.2f}: base rows moved {int((d != 0).sum()):,} (>60 s {int((np.abs(d) > 60).sum()):,}, >600 s '
          f'{int((np.abs(d) > 600).sum()):,}); base-level Q {float((d ** 2).sum()):.4e}; hit rows {int(s["hit"].sum())}, '
          f'day-scale rows held {int(s["hold"].sum())}')
    for nm in ('clean_U', 'clean_S'):
        dst = f'{EXP}/{nm}'
        if os.path.exists(dst): shutil.rmtree(dst)
        os.makedirs(f'{dst}/logs'); shutil.copytree(f'{SRC}/cand', f'{dst}/cand'); shutil.copytree(f'{SRC}/submissions', f'{dst}/submissions')
        for f in ('offsets.json', 'wF.npz', 'base_window.npz'): shutil.copy2(f'{SRC}/{f}', f'{dst}/{f}')
    hS = write_sub(f'{EXP}/clean_S/submissions/quick-boat_v100.parquet', s['ids'], new, f'{SRC}/submissions/quick-boat_v100.parquet')
    assert sha(f'{EXP}/clean_U/submissions/quick-boat_v100.parquet') == SHA['v115']
    assert hS == SHA['S_base'], f'swapped base does not reproduce: {hS}'
    np.savez(f'{EXP}/scored_base.npz', ids=s['ids'], gate=s['gate'], swap=new, c=c, P23=s['P23'], P25=s['P25'], hit=s['hit'],
             hold=s['hold'], alpha=a)
    print(f'  wrote clean_S/submissions/quick-boat_v100.parquet sha256 {hS}; clean_U is the unswapped copy')
    return hS


# ------------------------------------------------------------------------------------------------ tail (step 4b)
def run_tail(C):
    C = os.path.abspath(C); assert os.path.realpath(C).startswith(os.path.realpath(EXP))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DC26_', 'SHIFT_', 'V12', 'V11', 'FABLE', 'SHA_', 'ADSB_'))}
    env.update(DC26_DATA=D, SKIP_BASE='1', PIN_PICKS='1', WF='archived', CLEAN_DIR=C, TMPDIR=f'{C}/tmp')
    os.makedirs(f'{C}/tmp', exist_ok=True)
    say = lambda m: print(f'{pd.Timestamp.now("UTC"):%H:%M:%SZ} {m}', flush=True)
    say(f'[tail] regen_clean.sh SKIP_BASE=1 PIN_PICKS=1 WF=archived CLEAN_DIR={C}')
    with open(f'{C}/logs/regen_clean.log', 'w') as lg:
        subprocess.run(['bash', 'scripts/regen_clean.sh'], cwd=REPO, env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
    S = f'{C}/submissions'
    off = json.load(open(f'{C}/offsets.json'))
    env.update(SHIFT_1=str(off['SHIFT_1']), SHIFT_7=str(off['SHIFT_7']), DC26_SUBDIR=S, DC26_CAND=f'{C}/cand', DC26_WF=f'{C}/wF.npz',
               CLEAN='1', STAGE_MLP_FILE=f'{C}/cand/stage_mlp.npz', DC26_MEMBERS_RK=f'{REPO}/artifacts/p2_members_rk.parquet')
    for k in ('SKIP_BASE', 'WF', 'CLEAN_DIR'): env.pop(k)
    say('[v117] not built (the clean chain did not pass its rule): v117 = v114')
    shutil.copy2(f'{S}/quick-boat_v114.parquet', f'{S}/quick-boat_v117.parquet')
    prev = 117
    for v in (120, 121, 122, 123, 124, 125, 126):
        e = dict(env, **{f'V{v}_BASE': f'{S}/quick-boat_v{prev}.parquet', f'V{v}_OUT': f'{S}/quick-boat_v{v}.parquet',
                         'V121_REF': f'{S}/quick-boat_v121.parquet'})
        r = subprocess.run(['uv', 'run', '--quiet', '--with', 'pandas', '--with', 'pyarrow', '--with', 'numpy', 'python',
                            f'scripts/v{v}_build.py'], cwd=REPO, env=e, capture_output=True, text=True)
        open(f'{C}/logs/v{v}.log', 'w').write(r.stdout + r.stderr)
        if r.returncode: print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f'v{v} failed')
        say(f'[v{v}] ' + ' | '.join(l.strip() for l in r.stdout.splitlines() if 'sha256' in l)[:160])
        prev = v
    h = sha(f'{S}/quick-boat_v126.parquet'); say(f'{C}: v126 sha256 {h}'); return h


# ------------------------------------------------------------------------------------------------ final (steps 5-7)
def run_final():
    rep = json.load(open(f'{EXP}/window.json')); a = rep['alpha_pooled']
    ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
    p127 = f'{REPO}/submissions/quick-boat_v127.parquet'; assert sha(p127) == SHA['v127']
    hU = sha(f'{EXP}/clean_U/submissions/quick-boat_v126.parquet')
    assert hU == SHA['v127'], f'the unswapped tail does not reproduce v127: {hU}'
    t127 = pq.read_table(p127).to_pandas(); assert np.array_equal(t127.MVT_ID_mvt.values, ids)
    v127 = t127.TAXITIME_SEC_mvt.values.astype(float)
    assert sha(f'{EXP}/clean_S/submissions/quick-boat_v126.parquet') == SHA['S_v126'], 'the swapped tail does not reproduce'
    tS = pq.read_table(f'{EXP}/clean_S/submissions/quick-boat_v126.parquet').to_pandas(); assert np.array_equal(tS.MVT_ID_mvt.values, ids)
    raw = tS.TAXITIME_SEC_mvt.values.astype(float)
    s = scored_inputs(); apt = s['apt']; sb = np.load(f'{EXP}/scored_base.npz')
    out = raw.copy()
    ds = s['hold']; n_ds_diff = int((raw[ds] != v127[ds]).sum()); out[ds] = v127[ds]
    Q_precap = float(((out - v127) ** 2).sum()); held = []
    while True:
        d = out - v127; Q = float((d ** 2).sum())
        if Q == 0: break
        i = int(np.argmax(d ** 2)); sh = float(d[i] ** 2 / Q)
        if sh <= CAP_SHARE: break
        held.append(dict(MVT_ID=float(ids[i]), apt=apt[i], S=float(out[i]), v127=float(v127[i]), d=float(d[i]), share=sh,
                         Q_before=Q, base_gate=float(sb['gate'][i]), base_swap=float(sb['swap'][i]), c=float(sb['c'][i])))
        out[i] = v127[i]
    d = out - v127; D2 = d ** 2; Q = float(D2.sum())
    byapt = {x: dict(rows_moved=int(((d != 0) & (apt == x)).sum()), rows_600=int(((np.abs(d) > 600) & (apt == x)).sum()),
                     Q=float(D2[apt == x].sum())) for x in sorted(set(apt)) if (D2[apt == x].sum() > 0)}
    big = np.abs(d) > 600
    chg = dict(rows_moved=int((d != 0).sum()), rows_moved_60=int((np.abs(d) > 60).sum()), rows_moved_600=int(big.sum()),
               Q=Q, Q_600=float(D2[big].sum()), Q_le600=float(D2[~big].sum()), largest_share=float(D2.max() / Q) if Q else 0.0,
               by_airport=byapt, held_cap=held, dayscale_rows=int(ds.sum()), dayscale_rows_where_S_differed=n_ds_diff,
               base_level_rows_moved=int((sb['swap'] != sb['gate']).sum()),
               base_level_Q=float(((sb['swap'] - sb['gate']) ** 2).sum()), Q_before_cap=Q_precap)
    print(f'S vs v127: rows moved {chg["rows_moved"]:,} (>60 s {chg["rows_moved_60"]:,}, >600 s {chg["rows_moved_600"]:,}); '
          f'Q {Q:.4e} (>600 band {chg["Q_600"]:.4e}); largest remaining share {chg["largest_share"]:.4f}')
    print(f'  day-scale rows set to v127: {int(ds.sum())} ({n_ds_diff} differed); held by the 5% cap: {len(held)}')
    for h in held: print(f'    held {h["MVT_ID"]:.0f} {h["apt"]} S {h["S"]:.0f} v127 {h["v127"]:.0f} share {h["share"]:.3f}')
    for x, v in byapt.items(): print(f'    {x}  moved {v["rows_moved"]:6,}  >600 {v["rows_600"]:4}  Q {v["Q"]:.4e}')
    # ---- the deployed-level instrument: window base-level replay, Q-anchored on the final vectors
    ctx = I.window(); wv = np.load(f'{EXP}/window_vectors.npz'); assert np.array_equal(wv['ids'], ctx['ids'])
    est = I.bulk(ctx, wv['gateW'], wv['swapW'], v127, out)
    G = est['G_hat']; SE = est['SE']; pred = -G; lo, hi = pred - 1.96 * SE, pred + 1.96 * SE
    score = lambda x: float(np.sqrt(V127_SCORE ** 2 + x / N_B))
    inst = dict(m2T_w=est['m2T_w'], m2T_w_scaled=est['m2T_w'] * N_B / len(ctx['y']), Q_w=est['Q_w'], Q_b=est['Q_b'],
                G_hat=G, SE=SE, dSSE_pred=pred, dSSE_ci95=[lo, hi], dSSE_ci_3SE=[pred - 3 * SE, pred + 3 * SE],
                score_pred=score(pred), score_ci95=[score(lo), score(hi)], score_3SE=[score(pred - 3 * SE), score(pred + 3 * SE)],
                window_base_level_gain_scaled=est['G_window'], m2T_w_ord_scaled=est['m2T_w_ord'] * N_B / len(ctx['y']),
                m2T_w_ext_scaled=est['m2T_w_ext'] * N_B / len(ctx['y']), record='v127 pre-registered 246.41, paid 247.7412',
                Q_w_scaled=est['Q_w'] * N_B / len(ctx['y']), Q_w_scaled_over_Q_b=est['Q_w'] * N_B / len(ctx['y']) / est['Q_b'])
    print(f'\nINSTRUMENT (p1a_instrument.bulk, Q-anchored): -2T_w x N_b/N_w {inst["m2T_w_scaled"]:+.4e} (SE {SE:.3e}), Q_b {est["Q_b"]:.4e}')
    print(f'  predicted dSSE = SSE(S) - SSE(v127) = {pred:+.4e}, 95% [{lo:+.4e}, {hi:+.4e}]; 3 SE [{pred-3*SE:+.4e}, {pred+3*SE:+.4e}]')
    print(f'  predicted score {inst["score_pred"]:.4f}, 95% [{inst["score_ci95"][0]:.4f}, {inst["score_ci95"][1]:.4f}], '
          f'3 SE [{inst["score_3SE"][0]:.4f}, {inst["score_3SE"][1]:.4f}]   (record: v127 pre-registered {V127_PREREG}, paid {V127_SCORE})')
    print(f'  window base-level gain (G_window, scaled) {est["G_window"]:+.4e}; Q_w scaled {inst["Q_w_scaled"]:.4e} = '
          f'{inst["Q_w_scaled_over_Q_b"]:.2f} x Q_b; Q of S before the cap {Q_precap:.4e}')
    # beside it, the window's own base-level change scaled to the 2026 row count (CLAUDE.md: when Q_w >> Q_b trust the window's
    # own gain over the Q-anchored G_hat); interval = the day-cluster bootstrap percentiles of step 3, scaled
    rp = rep['replay']; sc = rp['board_scale']; wd = rp['dSSE'] * sc; wci = [rp['ci95_pct'][0] * sc, rp['ci95_pct'][1] * sc]
    inst['window_own'] = dict(dSSE=wd, ci95=wci, SE=rp['SE'] * sc, score=score(wd), score_ci95=[score(wci[0]), score(wci[1])])
    print(f'  the window\'s own base-level change, scaled: dSSE {wd:+.4e}, 95% [{wci[0]:+.4e}, {wci[1]:+.4e}] -> score {score(wd):.4f} '
          f'[{score(wci[0]):.4f}, {score(wci[1]):.4f}]')
    # sensitivity, not the pre-registered number: the same 5% cap applied to the window's own change set
    sw = wv['swapW'].copy(); gw = wv['gateW']; nheld_w = 0
    while True:
        dw = (sw - gw) ** 2; Qw = dw.sum()
        if Qw == 0 or dw.max() / Qw <= CAP_SHARE: break
        i = int(np.argmax(dw)); sw[i] = gw[i]; nheld_w += 1
    e2 = I.bulk(ctx, gw, sw, v127, out); p2 = -e2['G_hat']
    inst['sensitivity_window_capped'] = dict(window_rows_held=nheld_w, m2T_w_scaled=e2['m2T_w'] * N_B / len(ctx['y']), SE=e2['SE'],
                                             dSSE_pred=p2, dSSE_ci95=[p2 - 1.96 * e2['SE'], p2 + 1.96 * e2['SE']], score_pred=score(p2))
    print(f'  sensitivity, the 5% cap applied to the window change set too ({nheld_w} window rows held): predicted dSSE {p2:+.4e} '
          f'(SE {e2["SE"]:.3e}), score {score(p2):.4f}')
    p = f'{EXP}/S.parquet'; hS = write_sub(p, ids, out, p127)
    chk = pq.read_table(p); assert chk.num_rows == N_B and chk.schema.field('MVT_ID_mvt').type == pa.float64()
    print(f'\nwrote {p}  sha256 {hS}')
    if SHA_S: assert hS == SHA_S, f'S does not reproduce: {hS} != {SHA_S}'; print('  sha256 asserted')
    pre = dict(
        experiment='analysis/FINAL-PLAN.md, The interim experiment: does the rebuilt copy model beat the learned gate on 2026 rows?',
        S=dict(path=p, sha256=hS, rows=N_B, base='v127 (sha ' + SHA['v127'] + ', scored 247.7412)',
               unswapped_tail_reproduces_v127=hU == SHA['v127'], data=D),
        swap_rule=__doc__.split('THE SWAP RULE')[1].split('MODES')[0].strip(),
        copy_label='ext = TAXITIME >= 7200 s, the label the chain classifier and both isotonic maps are fitted on',
        alpha=dict(pooled=a, fitted_january=rep['alpha_jan'], fitted_july=rep['alpha_jul'],
                   july_dSSE_at_january_alpha=rep['xfit_jul_dSSE_at_alpha_jan'], january_dSSE_at_july_alpha=rep['xfit_jan_dSSE_at_alpha_jul'],
                   grid=rep['alpha_grid'], replaces='0.35, chosen on the leaderboard (REPRODUCE.md section 9)'),
        window_replay=rep['replay'], cap=dict(rule=f'hold at v127 any row carrying more than {CAP_SHARE:.0%} of Q, largest first, '
                                                  'recomputing Q after each hold', share=CAP_SHARE),
        dayscale='LIRF departures with LOBT null and MVT - SCHED >= 20,000 s: held at the gate in the swap, set to v127 in S',
        change_set=chg, prediction=inst,
        decision_rule=['If S scores below v127 (247.7412), Feb/Jun uses the rebuilt copy model; otherwise it keeps the gate.',
                       'If the score falls outside the predicted interval widened to 3 SE, stop: the build or the instrument is '
                       'broken. Neither change ships until that is explained.',
                       'The ratio of actual to predicted gain is recorded and applied to the final expected Feb/Jun range. No '
                       'further submission is chosen in light of this one.'],
        notes=['The window replay is at base level (gate base vs swapped base); the 2026 change is at the final level, after the '
               'CLEAN tail re-ran on the swapped base. The tail window side always uses the gate base, so the stage retrains on '
               'unchanged window inputs.', 'P25 on the scored rows is inverted from the archived int32 v25 against the v15 bulk where v25 != v23 '
               '(error <= 0.5 / |copy - bulk| per row); P25 := P23 where v25 == v23, which the archive cannot distinguish.'],
        code='scripts/final/exp_copyswap.py all')
    json.dump(pre, open(f'{EXP}/prereg.json', 'w'), indent=1, default=float)
    print(f'wrote {EXP}/prereg.json')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('window', 'all'): run_window()
    if mode in ('bases', 'all'): run_bases()
    if mode == 'tail': print(run_tail(sys.argv[2]))
    if mode == 'all':
        hU = run_tail(f'{EXP}/clean_U'); assert hU == SHA['v127'], f'unswapped tail does not reproduce v127: {hU}'
        run_tail(f'{EXP}/clean_S')
    if mode in ('final', 'all'): run_final()
