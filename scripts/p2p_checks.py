"""p2p step 4 diagnostics for the FULL package (s492 P): where the priced change lives. Measurement only.

Re-fits the FULL package exactly as scripts/p2p_price.py does (same members, same simplex, same SHIFT, same base) and
breaks down (a) the ranking-side change Q_b by month, by date band and by airport, and its row concentration;
(b) the January|July cross-fit window linear term by month and airport; (c) the window's own net gain for the
guard (-2T_w - Q_w, cross-fit, scaled to the scored set); (d) the FULL package with each new member left out.
"""
import os, sys, itertools, numpy as np, pandas as pd, pyarrow.parquet as pq
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{REPO}/scripts'); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
SP = (os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + ''); P2 = f'{SP}/p2P'
N_B = I.N_B; cl = I.cl; ONE = 1.7717e8
ctx = I.window(); y, ext = ctx['y'], ctx['ext']; O = ~ext; nw = len(y); sc = N_B / nw
mon, apt, clus = ctx['month'].astype(int), ctx['apt'].astype(str), ctx['cluster']
SHW = np.where(mon == 1, -8.778, -3.861); J, L = mon == 1, mon == 7
mean_of = lambda fs: np.mean([np.load(f) for f in fs], 0)
rk_ids = pd.read_parquet(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.values
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
r2s = pd.Series(np.arange(len(rk_ids)), index=rk_ids).reindex(sub_ids).values.astype(int)
rkc = lambda f, c: pd.read_parquet(f'{W}/{f}').set_index('MVT_ID_mvt').reindex(sub_ids)[c].values.astype(float)
MW = {'L55': mean_of([f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)]), 'P55': mean_of([f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)]),
      'K32': mean_of([f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)]), 'K128': mean_of([f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)]),
      'ID': mean_of([f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)]), 'RQ': mean_of([f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)]),
      'CL': mean_of([f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)]),
      'L55o': mean_of([f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)]), 'P55o': mean_of([f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)])}
MR = {'L55': rkc('v55_trees.parquet', 'lgbm_opdi'), 'P55': rkc('v55_trees.parquet', 'perapt_opdi'),
      'K32': rkc('d2_window_rk.parquet', 'd2_window'), 'K128': rkc('d2_window_rk_K128.parquet', 'd2_window'),
      'ID': mean_of([f'{P2}/arms/rank_idcat_s{s}.npy' for s in (0, 1)])[r2s], 'RQ': mean_of([f'{P2}/arms/rank_rwyq_s{s}.npy' for s in (0, 1)])[r2s],
      'CL': mean_of([f'{P2}/clean/rank_clean_s{s}.npy' for s in (0, 1)]),
      'L55o': mean_of([f'{P2}/trees/rank_oof_pooled_s{s}.npy' for s in (0, 1)])[r2s], 'P55o': mean_of([f'{P2}/trees/rank_oof_perapt_s{s}.npy' for s in (0, 1)])[r2s]}
SHIP = ['L55', 'P55', 'K32', 'K128']; w0 = np.array([.2230, .2030, .0880, .4860]); FULL = list(MW)
M0 = np.stack([MW[k] for k in SHIP], 1); M0r = np.stack([MR[k] for k in SHIP], 1)
B0 = M0 @ w0 + SHW; inc0 = I.mix(ctx, B0); appw = np.abs(inc0 - ctx['vsd']) > 60
v99 = I.sub('v99')
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
ms = cl(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt)); appr = np.abs(v99 - ms) > 60
tr_ = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True); rmon, rdom = tr_.dt.month.values, tr_.dt.day.values; rapt = R.ADEP_mvt.astype(str).values
def simplex(A, t):
    A = A / 1000.0; t = t / 1000.0; G = A.T @ A; b = A.T @ t; n = A.shape[1]; best = None
    for rr in range(1, n + 1):
        for S in itertools.combinations(range(n), rr):
            S = list(S); k = len(S); K = np.zeros((k + 1, k + 1)); K[:k, :k] = G[np.ix_(S, S)]; K[:k, k] = 1; K[k, :k] = 1
            q = np.zeros(k + 1); q[:k] = b[S]; q[k] = 1
            try: ws = np.linalg.solve(K, q)[:k]
            except np.linalg.LinAlgError: continue
            if (ws < -1e-10).any(): continue
            wf = np.zeros(n); wf[S] = np.clip(ws, 0, None); wf /= wf.sum(); f = float(wf @ G @ wf - 2 * wf @ b)
            if best is None or f < best[0]: best = (f, wf)
    return best[1]
t = y - SHW
def boot_se(c, B=2000, seed=5):
    cs = np.bincount(clus, weights=c); k = len(cs); rng = np.random.default_rng(seed)
    return float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(B)]).std()) * sc
def price(ks):
    M = np.stack([MW[k] for k in ks], 1); Mr = np.stack([MR[k] for k in ks], 1)
    w = simplex(M[O], t[O]); wj = simplex(M[O & L], t[O & L]); wl = simplex(M[O & J], t[O & J])
    bn = np.where(J, M @ wj, M @ wl)
    new = inc0.copy(); new[appw] = cl(np.rint(inc0[appw] + (bn + SHW - B0)[appw])); d = new - inc0; c = -2 * (inc0 - y) * d
    dr = Mr @ w - M0r @ w0; v100 = v99.copy(); v100[appr] = cl(np.rint(v99[appr] + dr[appr])); qb = (v100 - v99) ** 2
    return w, c, d, qb, v100
w, c, d, qb, v100 = price(FULL)
Qb = qb.sum(); m2T = c.sum() * sc; se = boot_se(c); Qw = (d ** 2).sum() * sc
print(f'FULL weights {dict(zip(FULL, np.round(w, 4)))}')
print(f'FULL Jan|Jul: -2T_w {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {m2T-Qb:+.4e}  SE {se:.4e}  G/SE {(m2T-Qb)/se:+.3f}   window net gain (-2T_w - Q_w) {m2T-Qw:+.4e}  Q_w {Qw:.4e}')
print('\n(a) where Q_b lives on the scored set')
bands = [('Jan 1', (rmon == 1) & (rdom == 1)), ('Jan 2-7', (rmon == 1) & (rdom >= 2) & (rdom <= 7)), ('Jan 8-31', (rmon == 1) & (rdom >= 8)),
         ('Jul 1-31', rmon == 7)]
for nm, m in bands:
    print(f'   {nm:<9} rows {int(m.sum()):>7,}  Q_b {qb[m].sum():.4e} ({qb[m].sum()/Qb:6.1%})  RMS change {np.sqrt(np.mean(qb[m])):6.2f} s')
for a in sorted(set(rapt)):
    m = rapt == a; print(f'   {a}  rows {int(m.sum()):>7,}  Q_b share {qb[m].sum()/Qb:6.1%}  RMS change {np.sqrt(np.mean(qb[m])):6.2f} s')
srt = np.sort(qb)[::-1]
print(f'   concentration: top 100 rows {srt[:100].sum()/Qb:.1%}, top 1% ({len(qb)//100:,} rows) {srt[:len(qb)//100].sum()/Qb:.1%}, largest single row {srt[0]:.3e} ({srt[0]/Qb:.2%})')
print('\n(b) window linear term (Jan|Jul cross-fit) by month and airport, scaled to the scored set; Q_w alongside')
for nm, m in (('Jan', J), ('Jul', L)):
    print(f'   {nm}: -2T_w {c[m].sum()*sc:+.4e}  Q_w {(d[m]**2).sum()*sc:.4e}   (ranking Q_b {qb[rmon==(1 if nm=="Jan" else 7)].sum():.4e})')
for a in sorted(set(apt)):
    m = apt == a; mr = rapt == a
    print(f'   {a}  -2T_w {c[m].sum()*sc:+.4e}  Q_w {(d[m]**2).sum()*sc:.3e}  Q_b {qb[mr].sum():.3e}   net {c[m].sum()*sc - qb[mr].sum():+.3e}')
print('\n(d) FULL with each new member left out (Jan|Jul G_hat, SE), a stability read, NOT a selection')
for k in ['ID', 'RQ', 'CL', 'L55o', 'P55o']:
    ks = [x for x in FULL if x != k]; w_, c_, d_, qb_, _ = price(ks); g = c_.sum() * sc - qb_.sum(); s_ = boot_se(c_)
    print(f'   FULL minus {k:<5} G_hat {g:+.4e}  SE {s_:.3e}  G/SE {g/s_:+.2f}  Q_b {qb_.sum():.3e}')
np.savez(f'{P2}/full_package.npz', members=np.array(FULL), w=w, v100=v100)
print(f'\nsaved {P2}/full_package.npz (FULL weights and the v100 vector these members imply)')
