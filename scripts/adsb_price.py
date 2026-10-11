"""Price an ADS-B pushback correction on top of v100, on the window, at the deployed level (s488 instrument).
1. CONTROL: rebuild v100 on the window from the shipped members and weights (full_package.npz) exactly as
   p2p_price.py does, and reproduce its linear term against v99.
2. CANDIDATES, fitted on one month and scored on the other (January|July cross-fit, the s417 selection rule),
   applied only to rows v100's own rule moves (|inc0 - ms| > 60) that have an ADS-B pushback:
     LIN  per-airport least squares  y - v100 = a + b (ab_taxi - v100)
     GBM  LightGBM on y - v100 with the ADS-B features, trained on ordinary rows (y < 7200)
   The window SSE change is reported over ALL window rows of the parsed days, ordinary and extreme, with a
   day-cluster bootstrap SE. Q_b needs the 2026 side and is printed only when push_feat covers 2026.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm python scripts/adsb_price.py
"""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{REPO}/scripts'); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
P2 = os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P'; N_B = I.N_B; cl = I.cl
ctx = I.window(); y, ext = ctx['y'], ctx['ext']; O = ~ext; nw = len(y); sc = N_B / nw
mon, apt, clus = ctx['month'].astype(int), ctx['apt'].astype(str), ctx['cluster']
SHW = np.where(mon == 1, float(os.environ.get('SHIFT_1', '-8.778')), float(os.environ.get('SHIFT_7', '-3.861')))
mean_of = lambda ps: np.mean([np.load(p) for p in ps], 0)
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
# ---------------- 1. v100 on the window --------------------------------------------------------------------------
WIN = {'L55': [f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)], 'P55': [f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)],
       'K32': [f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)], 'K128': [f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)],
       'ID': [f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)], 'RQ': [f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)],
       'CL': [f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)], 'L55o': [f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)],
       'P55o': [f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)]}
fp = np.load(os.environ.get('DC26_WF', f'{P2}/full_package.npz')); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
M = np.stack([mean_of(WIN[k]) for k in FULL], 1)
w0 = np.array([.2230, .2030, .0880, .4860]); B0 = np.stack([mean_of(WIN[k]) for k in ('L55', 'P55', 'K32', 'K128')], 1) @ w0 + SHW
inc0 = I.mix(ctx, B0); appw = np.abs(inc0 - ctx['vsd']) > 60
v100w = inc0.copy(); v100w[appw] = cl(np.rint(inc0[appw] + (M @ wF + SHW - B0)[appw]))
v100r = I.sub('v100'); v99r = v100r if I.CLEAN else I.sub('v99')
assert I.CLEAN or np.array_equal(v100r, fp['v100'].astype(float)), 'shipped v100 != full_package v100'
m2T = -2.0 * float(((inc0 - y) * (v100w - inc0)).sum()) * sc; Qb = float(((v100r - v99r) ** 2).sum())
print(f'CONTROL v100 on the window: members {FULL} w {np.round(wF, 4)}\n  -2T_w {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {m2T - Qb:+.4e}   '
      f'(recorded pre-registered G_hat +3.4646e8)   window RMSE inc0 {np.sqrt(np.mean((inc0-y)**2)):.4f} -> v100 {np.sqrt(np.mean((v100w-y)**2)):.4f}')
# ---------------- 2. ADS-B features on the window ----------------------------------------------------------------
F = pd.read_parquet(os.environ.get('ADSB_FEAT', f'{W}/adsblol/push_feat.parquet')).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
Fw = F.reindex(ctx['ids'])
parsed = Fw.day.notna().values; has = Fw.ab_push.notna().values & parsed
days_parsed = sorted(set(Fw.day.dropna()))
print(f'\nADS-B: parsed window days {len(days_parsed)} of 62; window rows on them {parsed.sum():,} ({parsed.mean():.1%}); with a pushback {has.sum():,} '
      f'({has.sum()/max(parsed.sum(),1):.1%} of parsed rows); of those, v100-rule rows {int((has & appw).sum()):,}')
tr = pq.read_table(f'{D}/training_2025-01-01_2025-02-01.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt', 'PHASE_mvt']).to_pandas()
tr = pd.concat([tr, pq.read_table(f'{D}/training_2025-07-01_2025-08-01.parquet', columns=tr.columns.tolist()).to_pandas()])
tr = tr[tr.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(ctx['ids'])
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
naive = sec(tr.MVT_TIME_UTC_mvt) - sec(tr.AOBT_3_flt)
AC = {a: i for i, a in enumerate(sorted(set(apt)))}
def X_of(Fx, base, nv, ap, ms, lb):
    return pd.DataFrame({'d_ab': Fx.ab_taxi.values - base, 'ab_taxi': Fx.ab_taxi.values, 'base': base, 'd_naive': nv - base,
                         'd_ms': ms - base, 'd_lb': lb - base, 'how': Fx.ab_how.values, 'dwell': Fx.ab_dwell.values,
                         'd0': Fx.ab_d0.values, 'dpush': Fx.ab_dpush.values, 'n': Fx.ab_n.values, 'gap': Fx.ab_gap.values,
                         'stop': Fx.ab_stop.values, 'gs0': Fx.ab_gs0.values, 'dto': Fx.ab_dto.values,
                         'apt': pd.Categorical([AC[a] for a in ap], categories=range(len(AC)))})
X = X_of(Fw, v100w, naive, apt, ctx['vsd'], ctx['vlb']); rres = y - v100w
J, L = mon == 1, mon == 7; use = has & appw
PRM = dict(objective='l2', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0, num_threads=8,
           deterministic=True, force_col_wise=True)   # s513b: pin the runtime histogram choice so the build reproduces
NR = 400
def cand_lin(fit, app):
    out = np.zeros(nw)
    for a in AC:
        f = fit & (apt == a); g = app & (apt == a)
        if f.sum() < 100 or not g.any(): continue
        A = np.c_[np.ones(f.sum()), X.d_ab.values[f]]; coef = np.linalg.lstsq(A, rres[f], rcond=None)[0]
        out[g] = coef[0] + coef[1] * X.d_ab.values[g]
    return out
def cand_gbm(fit, app, seed=0):
    m = lgb.train({**PRM, 'seed': seed}, lgb.Dataset(X[fit], rres[fit]), NR)
    out = np.zeros(nw); out[app] = m.predict(X[app]); return out
def score(delta, name):
    new = v100w.copy(); new[use] = cl(np.rint(v100w[use] + delta[use]))
    c = (v100w - y) ** 2 - (new - y) ** 2                 # SSE removed, per row
    cs = np.bincount(clus, weights=c, minlength=clus.max() + 1); pd_ = np.unique(clus[parsed])
    rng = np.random.default_rng(5); bs = np.array([cs[pd_][rng.integers(0, len(pd_), len(pd_))].sum() for _ in range(2000)])
    G, SE = float(c.sum()), max(float(bs.std()), 1e-9)
    per = {a: float(c[apt == a].sum()) for a in AC}
    print(f'  {name:<5} window SSE removed {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}   ordinary {c[O].sum():+.3e}  extreme {c[ext].sum():+.3e}  '
          f'rows moved {int((new != v100w).sum()):,}  RMSE on moved rows {np.sqrt(np.mean((new-v100w)[new != v100w]**2)):.1f} s')
    print('        per airport ' + '  '.join(f'{a} {v:+.2e}' for a, v in per.items() if v != 0))
    return new, G, SE
print(f'\nCANDIDATES, January|July cross-fit (fit on one month\'s ordinary rows, score the other). Board scale: x {sc:.4f}; 1 board-s = 1.7717e8')
fitO = use & (y < 7200)
for nm, fn in (('LIN', cand_lin), ('GBM', cand_gbm)):
    d = np.zeros(nw)
    for fit, app in ((fitO & L, use & J), (fitO & J, use & L)):
        if fit.sum() < 500 or not app.any(): continue
        d += fn(fit, app)
    new, G, SE = score(d, nm)
    print(f'        board-scale (x sc) {G*sc:+.4e} = {G*sc/1.7717e8:+.3f} board-s on the parsed days only')
# ---------------- 3. the scored side: train on the whole window, apply to 2026, exact Q_b -------------------------
Fr = F.reindex(sub_ids); hasr = Fr.ab_push.notna().values
if hasr.sum() == 0: sys.exit('no 2026 ADS-B features yet')
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt',
                                                  'LOBT_flt', 'AOBT_3_flt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
tm = sec(R.MVT_TIME_UTC_mvt); msr = cl(tm - sec(R.SCHED_TIME_UTC_mvt)); lbr = tm - sec(R.LOBT_flt); nvr = tm - sec(R.AOBT_3_flt)
aptr = R.ADEP_mvt.values.astype(str); monr = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True).dt.month.values
Xr = X_of(Fr, v100r, nvr, aptr, msr, lbr); user = hasr & (np.abs(v100r - msr) > 60)
SEEDS = (0, 1, 2); fitA = use & (y < 7200)
dW = np.zeros(nw)                                              # the cross-fitted window delta, 3-seed mean
for fit, app in ((fitA & L, use & J), (fitA & J, use & L)):
    dW[app] = np.mean([lgb.train({**PRM, 'seed': s}, lgb.Dataset(X[fit], rres[fit]), NR).predict(X[app]) for s in SEEDS], 0)
dR = np.zeros(len(sub_ids))
dR[user] = np.mean([lgb.train({**PRM, 'seed': s}, lgb.Dataset(X[fitA], rres[fitA]), NR).predict(Xr[user]) for s in SEEDS], 0)
newW = v100w.copy(); newW[use] = cl(np.rint(v100w[use] + dW[use]))
newR = v100r.copy(); newR[user] = cl(np.rint(v100r[user] + dR[user]))
est = I.bulk(ctx, v100w, newW, v100r, newR, B=2000)
print(f'\nSCORED SIDE (v102 candidate = v100 + ADS-B stage, 3 seeds; window term Jan|Jul cross-fitted, Q_b exact)\n'
      f'  2026 departures with a pushback {hasr.sum():,} ({hasr.mean():.1%}); moved {int((newR != v100r).sum()):,}; '
      f'RMS change on moved rows {np.sqrt(np.mean((newR-v100r)[newR != v100r]**2)):.1f} s\n'
      f'  -2T_w*sc {est["m2T_w"]*sc:+.4e}  Q_w*sc {est["Q_w"]*sc:.4e}  Q_b {est["Q_b"]:.4e}  G_hat {est["G_hat"]:+.4e}  SE {est["SE"]:.3e}  '
      f'G/SE {est["G_hat"]/est["SE"]:+.2f}  = {est["G_hat"]/1.7717e8:+.3f} board-s  (window-own G {est["G_window"]:+.4e})')
print('\n  per airport: covered rule rows (window | 2026), mean and RMS correction (window | 2026), linear term, Q_b, '
      'and the linear term rescaled by 2026/window covered rows (an ESTIMATOR, assumes per-row gain transfers)')
tot = 0.0
for a in AC:
    mw, mr = use & (apt == a), user & (aptr == a)
    lin = -2.0 * float(((v100w - y) * (newW - v100w))[apt == a].sum()) * sc; qb = float(((newR - v100r) ** 2)[aptr == a].sum())
    k = mr.sum() / max(mw.sum(), 1) / sc; adj = lin * k * sc - qb; tot += adj if mw.sum() else 0.0
    f_ = lambda v, m: (f'{v[m].mean():+7.1f} {np.sqrt(np.mean(v[m]**2)):6.1f}' if m.any() else '      -      -')
    print(f'    {a}  rows {mw.sum():>6,} | {mr.sum():>6,}   corr {f_(newW - v100w, mw)} | {f_(newR - v100r, mr)}   '
          f'lin {lin:+.3e}  Q_b {qb:.3e}  G {lin-qb:+.3e}   coverage-scaled G {adj:+.3e}')
print(f'  coverage-scaled total {tot:+.4e} = {tot/1.7717e8:+.3f} board-s  (estimator; the plain G_hat above is the instrument)')
os.makedirs(f'{I.CAND}', exist_ok=True)
np.savez(f'{I.CAND}/v102_adsb.npz', ids=sub_ids, v100=v100r, new=newR, dR=dR, user=user, win_ids=ctx['ids'], newW=newW)   # newW: v102 on the window (the v103 incumbent)
