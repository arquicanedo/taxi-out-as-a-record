"""Verifier: REFIT and the v73 retrodiction re-priced with a window K128 built on the SHIPPED net feature spec."""
import os, sys, itertools, numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.environ['DC26_WORK']; DATA = os.environ['DC26_DATA']
TA = (os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackA'); REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_B = 344841; cl = lambda p: np.clip(p, 60., 200000.)
z = np.load(f'{TA}/verify/v_indep_ctx.npz', allow_pickle=True)
y, ext, mon, day, apt, vsd, vlb, Qg = z['y'], z['ext'], z['mon'], z['day'], z['apt'], z['vsd'], z['vlb'], z['Qg']
L55, P55, K32, K128g, Lo, Ln, Po, Pn = (z[k] for k in ('L55', 'P55', 'K32', 'K128', 'Lo', 'Ln', 'Po', 'Pn'))
ms, rid = z['ms'], z['rid']; sc = N_B / len(y); O = ~ext
SEEDS = [s for s in (0, 1, 2) if os.path.exists(f'{TA}/verify/out2/k128rs/k128rs_s{s}.npy')]
K128r = np.mean([np.load(f'{TA}/verify/out2/k128rs/k128rs_s{s}.npy') for s in SEEDS], 0)
r = lambda p, k: float(np.sqrt(np.mean((cl(p)[k] - y[k]) ** 2)))
print(f'window K128 on the gate matrix (track): ordinary solo {r(K128g,O):.3f};  on the shipped 49-feature spec ({len(SEEDS)} seeds): {r(K128r,O):.3f};  '
      f'RMS difference {np.sqrt(np.mean((K128g-K128r)**2)):.1f}s  corr {np.corrcoef(K128g,K128r)[0,1]:.4f}')
def mix(b):
    b = cl(b); V = np.stack([b, vsd, vlb, cl(86400. + b), b], 1); return cl(np.rint((Qg * V).sum(1)))
def rule(inc, delta):
    app = np.abs(inc - vsd) > 60; p = inc.copy(); p[app] = cl(np.rint(inc[app] + delta[app])); return p
clus = pd.factorize(pd.Series(mon).astype(str) + '-' + pd.Series(day).astype(str))[0]
def se(contrib, B=2000, seed=5):
    cs = np.bincount(clus, weights=contrib); k = len(cs); rng = np.random.default_rng(seed)
    return np.array([cs[rng.integers(0, k, k)].sum() for _ in range(B)]).std() * sc
def simplex_exact(A, yy):
    best = None; G = A.T @ A; b = A.T @ yy; n = A.shape[1]
    for rr in range(1, n + 1):
        for S in itertools.combinations(range(n), rr):
            S = list(S); k = len(S); Kk = np.zeros((k + 1, k + 1)); Kk[:k, :k] = G[np.ix_(S, S)]; Kk[:k, k] = 1; Kk[k, :k] = 1
            q = np.zeros(k + 1); q[:k] = b[S]; q[k] = 1; ws = np.linalg.solve(Kk, q)[:k]
            if (ws < -1e-10).any(): continue
            wf = np.zeros(n); wf[S] = ws; f = float(((A @ wf - yy) ** 2).sum())
            if best is None or f < best[0]: best = (f, wf)
    return best[1]
sub = lambda v: pq.read_table(f'{REPO}/submissions/quick-boat_{v}.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)
v99 = sub('v99'); rk = lambda fn, col: pd.read_parquet(f'{W}/{fn}').set_index('MVT_ID_mvt').reindex(rid)[col].values.astype(float)
MR = np.stack([rk('v55_trees.parquet', 'lgbm_opdi'), rk('v55_trees.parquet', 'perapt_opdi'), rk('d2_window_rk.parquet', 'd2_window'), rk('d2_window_rk_K128.parquet', 'd2_window')], 1)
appr = np.abs(v99 - ms) > 60; w0 = np.array([.2230, .2030, .0880, .4860])
for lab, K1 in (('K128 gate-matrix (track faithful base)', K128g), ('K128 shipped 49-feature spec', K128r)):
    M = np.stack([L55, P55, K32, K1], 1); inc99 = rule(mix(M @ w0), 0.2016 * ((Ln - Lo) + (Pn - Po)))
    wA = simplex_exact(M[O], y[O])
    dR = np.zeros(len(v99)); dR[appr] = cl(np.rint(v99[appr] + (MR @ (wA - w0))[appr])) - v99[appr]; Qb = (dR ** 2).sum()
    print(f'\n{lab}: shipped-weight bulk ord {r(M@w0,O):.3f}; refit weights {np.round(wA,4)} -> {r(M@wA,O):.3f}')
    for nm, dw in (('in-sample', M @ (wA - w0)),
                   ('Jan|Jul cross-fit', np.where(mon == 1, M @ (simplex_exact(M[O & (mon == 7)], y[O & (mon == 7)]) - w0), M @ (simplex_exact(M[O & (mon == 1)], y[O & (mon == 1)]) - w0)))):
        d = rule(inc99, dw) - inc99; c_ = -2 * (inc99 - y) * d; m2T = c_.sum() * sc; s = se(c_)
        print(f'  REFIT {nm:18} -2T_w {m2T:.4e}  Q_b {Qb:.4e}  G_hat {m2T-Qb:+.4e}  SE {s:.3e}  G/SE {(m2T-Qb)/s:+.2f}  ({(m2T-Qb)/1.7717e8:+.3f} board-s)')
    # v73 retrodiction on this base: v64 blend -> v73 blend (D from the window file)
    vid = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
    pos = pd.Series(np.arange(len(vid)), index=vid).reindex(z['ids']).values.astype(int)
    D = np.load(f'{W}/arch_d_window_va.npy')[pos]
    b64 = .2473 * L55 + .2217 * P55 + .0849 * D + .4462 * K32; b73 = .2230 * L55 + .2030 * P55 + .0880 * K32 + .4860 * K1
    wo, wn = mix(b64), mix(b73); d = wn - wo; c_ = -2 * (wo - y) * d; m2T = c_.sum() * sc
    Qb73 = ((sub('v73') - sub('v64')) ** 2).sum(); Gb = N_B * (263.0551 ** 2 - 262.8071 ** 2)
    print(f'  v73 retrodiction: -2T_w {m2T:.4e}  G_hat {m2T-Qb73:+.4e}  SE {se(c_):.3e}  board {Gb:.4e}  (G_hat-B)/SE {(m2T-Qb73-Gb)/se(c_):+.2f}')
