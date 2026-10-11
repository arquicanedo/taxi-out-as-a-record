"""p2p steps 1 (report), 3 and 4: weights and pricing of the Phase 2 bulk package (s492 P), through s488's
estimator on the LEAK-FREE faithful window base. Measurement only; writes nothing but its log and an npz.

BASE (window, 344,336 rows, v15_tr_jul19 window order == b4ab_ctx order, checked in p2p_controls.py)
  members  L55 / P55   p2p_v55.py MODE=win TE=lf, seeds 0,1,2 (v55_trees recipe, ten months, ten-month te_*)
           K32 / K128  p2p_knet.py (shipped 49-feature spec, ten-month te_*), seeds 0,1,2
  old bulk B0 = .2230 L55 + .2030 P55 + .0880 K32 + .4860 K128 + SHIFT_month (Jan -8.778, Jul -3.861)
  inc0 = the deployed mixture of B0 (p1a_instrument.mix: b4 gate weights, copy / LOBT / day components)
CANDIDATE (members M, simplex weights w: w >= 0, sum w = 1; SHIFT kept as shipped, so w is fitted to y - SHIFT on
window ORDINARY rows, y < 7200, which selects fitting rows only)
  window  B1 = M w + SHIFT; the change B1 - B0 is replayed through v99's shipping rule on ALL window rows:
          rows with |inc0 - vsd| > 60 get clip(rint(inc0 + B1 - B0)), the rest keep inc0
  ranking v100 = v99 on rows with |v99 - ms| <= 60; elsewhere clip(rint(v99 + M_r w_full - w0 M0_r))
  G_hat = -2 T_w N_b/N_w - Q_b,  -2T_w = -2 sum((inc0 - y) * (new - inc0)) over all window rows,
          Q_b = sum((v100 - v99)^2) EXACT, SE = day-cluster bootstrap of the linear term (62 days, B=2000)
  Weight schemes for the LINEAR term: in-sample; out-of-fold by calendar day (6 folds, weights refitted inside
  each fold on the other days' ordinary rows); January|July cross-fit (fit one month, score the other: THE
  PRE-REGISTERED DECISION VARIANT for the FULL package); leave-one-airport-out. Q_b always uses the full-window
  weights, which are what would ship.
RETRODICTIONS on the same base: v99 (0.2016 x colsample 1.0 -> 0.4 on the v15-construction trees, leak-free) and
v73 (v64 blend -> v73 blend) against the measured board.
"""
import os, sys, itertools, numpy as np, pandas as pd, pyarrow.parquet as pq
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, f'{REPO}/scripts'); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
SP = (os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '')
P2 = os.environ.get('P2P_OUT', f'{SP}/p2P'); N_B = I.N_B; cl = I.cl; B_BOOT = int(os.environ.get('BOOT', '2000'))
ctx = I.window(); y, ext = ctx['y'], ctx['ext']; O = ~ext; nw = len(y); sc = N_B / nw
mon, apt, clus = ctx['month'].astype(int), ctx['apt'].astype(str), ctx['cluster']
SHW = np.where(mon == 1, -8.778, -3.861)
r = lambda p, m: float(np.sqrt(np.mean((cl(p)[m] - y[m]) ** 2)))
def mean_of(paths, need=1):
    ok = [p for p in paths if os.path.exists(p)]
    return (np.mean([np.load(p) for p in ok], 0), len(ok)) if len(ok) >= need else (None, 0)
# ---------------- members --------------------------------------------------------------------------------------
tr_ids = pd.read_parquet(f'{W}/v15_tr_jul19.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.values
rk_ids = pd.read_parquet(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.values
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
rk2sub = pd.Series(np.arange(len(rk_ids)), index=rk_ids).reindex(sub_ids).values.astype(int)
assert (rk_ids[rk2sub] == sub_ids).all()
SPEC = {  # name: (window files, window seeds needed, ranking loader)
    'L55': ([f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)], 3, 'v55:lgbm_opdi'),
    'P55': ([f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)], 3, 'v55:perapt_opdi'),
    'K32': ([f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)], 3, 'd2:d2_window_rk.parquet'),
    'K128': ([f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)], 3, 'd2:d2_window_rk_K128.parquet'),
    'ID': ([f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)], 2, [f'{P2}/arms/rank_idcat_s{s}.npy' for s in (0, 1)]),
    'RQ': ([f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)], 2, [f'{P2}/arms/rank_rwyq_s{s}.npy' for s in (0, 1)]),
    'CL': ([f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)], 2, [f'{P2}/clean/rank_clean_s{s}.npy' for s in (0, 1)]),
    'L55o': ([f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)], 2, [f'{P2}/trees/rank_oof_pooled_s{s}.npy' for s in (0, 1)]),
    'P55o': ([f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)], 2, [f'{P2}/trees/rank_oof_perapt_s{s}.npy' for s in (0, 1)]),
}
MW, MR, NS = {}, {}, {}
for k, (wf, need, rl) in SPEC.items():
    a, n = mean_of(wf, need)
    if a is None: print(f'  member {k}: window side missing ({n} of {need} seeds)'); continue
    if isinstance(rl, str):
        kind, spec = rl.split(':')
        if kind == 'v55': b = pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)[spec].values.astype(float)
        else: b = pd.read_parquet(f'{W}/{spec}').set_index('MVT_ID_mvt').reindex(sub_ids)['d2_window'].values.astype(float)
    else:
        b, nb = mean_of(rl, need)
        if b is None: print(f'  member {k}: ranking side missing ({nb} of {need} seeds)'); continue
        if k != 'CL': b = b[rk2sub]                       # v15_rk order -> submitting order (clean-room saved in submitting order)
    assert np.isfinite(a).all() and np.isfinite(b).all(), k
    MW[k], MR[k], NS[k] = a.astype(float), b.astype(float), n
SHIP = ['L55', 'P55', 'K32', 'K128']; w0 = np.array([.2230, .2030, .0880, .4860])
CONTROL = os.environ.get('CONTROL', '')
TA = f'{SP}/trackA'
if CONTROL == 'trackA':
    # MACHINERY CONTROL: reproduce scripts/p1v/a_v_refit_k128rs.py's "K128 shipped 49-feature spec" REFIT row
    # (in-sample G_hat +5.9840e7 SE 3.639e7; Jan|Jul +6.8089e7 SE 4.027e7; Q_b 2.0305e8; shipped-weight bulk ord 222.245)
    # with ITS inputs: leaky v55 trees, gate-matrix K32, leaky shipped-spec K128; no SHIFT; base = v99 replayed.
    vid_ = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
    pos_ = pd.Series(np.arange(len(vid_)), index=vid_).reindex(ctx['ids']).values.astype(int)
    MW = {'L55': mean_of([f'{TA}/wc/v55_pooled_s{s}.npy' for s in (0, 1, 2)], 3)[0], 'P55': mean_of([f'{TA}/wc/v55_perapt_s{s}.npy' for s in (0, 1, 2)], 3)[0],
          'K32': np.load(f'{W}/arch_d2_wide_va_K32.npy')[pos_], 'K128': mean_of([f'{TA}/verify/out2/k128rs/k128rs_s{s}.npy' for s in (0, 1, 2)], 3)[0]}
    MR = {'L55': pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)['lgbm_opdi'].values.astype(float),
          'P55': pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)['perapt_opdi'].values.astype(float),
          'K32': pd.read_parquet(f'{W}/d2_window_rk.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)['d2_window'].values.astype(float),
          'K128': pd.read_parquet(f'{W}/d2_window_rk_K128.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)['d2_window'].values.astype(float)}
    NS = {k: 3 for k in SHIP}
    SHW = np.zeros(nw)
    print('*** CONTROL=trackA: leaky members, gate-matrix K32, no SHIFT, base = v99 replayed on the window ***')
if os.environ.get('COV') == '2026':
    # ROBUSTNESS ONLY: give the OPDI tree members the SCORED SET's OPDI coverage (57.0% Jan / 84.3% Jul against the
    # window's 87.0% / 91.0%) by adding track A's seed-0 masked-minus-unmasked difference (p1a_window_v55_masked.py,
    # leaky te_*) to every OPDI-using tree member. Not the decision base.
    zm = np.load(f'{TA}/wc/v55_mask_s0.npz')
    for k, dk in (('L55', 'd_pooled'), ('P55', 'd_perapt'), ('L55o', 'd_pooled'), ('P55o', 'd_perapt')):
        if k in MW: MW[k] = MW[k] + zm[dk]
    print('*** COV=2026: OPDI tree members given the scored set\'s OPDI coverage (robustness row, not the decision) ***')
assert all(k in MW for k in SHIP), 'the four leak-free shipped members are required'
# ---------------- step 1 report: members and the shipped-weight blend, leak-free against leaky ------------------
LK = {'L55': mean_of([f'{TA}/wc/v55_pooled_s{s}.npy' for s in (0, 1, 2)], 3)[0], 'P55': mean_of([f'{TA}/wc/v55_perapt_s{s}.npy' for s in (0, 1, 2)], 3)[0],
      'K32': mean_of([f'{TA}/verify/out2/k32rs/k32rs_s{s}.npy' for s in (0, 1, 2)], 3)[0], 'K128': mean_of([f'{TA}/verify/out2/k128rs/k128rs_s{s}.npy' for s in (0, 1, 2)], 3)[0]}
J, L = mon == 1, mon == 7
print(f'STEP 1. members on window ORDINARY rows (clipped; {int(O.sum()):,} rows), leak-free (this build) against leaky (track A)')
for k in MW:
    lk = LK.get(k)
    print(f'  {k:<5} ({NS[k]} seeds)  leak-free {r(MW[k],O):8.3f}  Jan {r(MW[k],O&J):8.3f}  Jul {r(MW[k],O&L):8.3f}'
          + (f'    leaky {r(lk,O):8.3f}  (leak-free - leaky {r(MW[k],O)-r(lk,O):+.3f})' if lk is not None else ''))
M0 = np.stack([MW[k] for k in SHIP], 1); M0r = np.stack([MR[k] for k in SHIP], 1)
B0 = M0 @ w0 + SHW
blk = np.stack([LK[k] for k in SHIP], 1) @ w0
_vid = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
K32g = np.load(f'{W}/arch_d2_wide_va_K32.npy')[pd.Series(np.arange(len(_vid)), index=_vid).reindex(ctx['ids']).values.astype(int)]
blkg = np.stack([LK['L55'], LK['P55'], K32g, LK['K128']], 1) @ w0
print(f'  shipped-weight blend      leak-free {r(M0 @ w0,O):8.3f}   leaky, all four shipped-spec {r(blk,O):8.3f}   '
      f'leaky with the gate-matrix K32 (the track A verifier row) {r(blkg,O):8.3f} (recorded 222.245)')
print(f'  shipped-weight blend+SHIFT leak-free {r(B0,O):8.3f}  Jan {r(B0,O&J):8.3f}  Jul {r(B0,O&L):8.3f}   leaky {r(blk+SHW,O):8.3f}')
inc0 = I.mix(ctx, B0)
if CONTROL == 'trackA':
    ldw = lambda t_, cs: np.mean([np.load(f'{TA}/wc/{t_}_cs{cs}_s{s}.npy') for s in (0, 1, 2)], 0)
    d99L = 0.2016 * ((ldw('pooled', '0.4') - ldw('pooled', '1.0')) + (ldw('perapt', '0.4') - ldw('perapt', '1.0')))
    a_ = np.abs(inc0 - ctx['vsd']) > 60; inc0 = inc0.copy(); inc0[a_] = cl(np.rint(inc0[a_] + d99L[a_]))
appw = np.abs(inc0 - ctx['vsd']) > 60
print(f'  deployed mixture of the leak-free base: all rows {r(inc0,np.ones(nw,bool)):.4f}  (recorded instrument, gate_a2 bulk: 313.8931); '
      f'rule rows |inc0 - vsd| > 60: {int(appw.sum()):,}')
# ---------------- ranking side ---------------------------------------------------------------------------------
v99 = I.sub('v99'); v96 = I.sub('v96'); v73 = I.sub('v73'); v64 = I.sub('v64')
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
ms = cl(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt)); appr = np.abs(v99 - ms) > 60
print(f'  ranking: v99 unfired rows (|v99 - ms| > 60) {int(appr.sum()):,} of {len(v99):,}')
# ---------------- window/ranking consistency of each member (a member whose ranking-side deviation from the shipped
# blend differs from its window-side deviation is not the same object on both sides) ---------------------------------
rmon = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True).dt.month.values
bw0, br0 = M0 @ w0, M0r @ w0
print('  member minus the shipped-weight blend, RMS s (window Jan / Jul | ranking Jan / Jul), and mean s (window | ranking)')
for k in MW:
    dw, dr_ = MW[k] - bw0, MR[k] - br0
    print(f'    {k:<5} {np.sqrt(np.mean(dw[J]**2)):7.2f} {np.sqrt(np.mean(dw[L]**2)):7.2f} | {np.sqrt(np.mean(dr_[rmon==1]**2)):7.2f} {np.sqrt(np.mean(dr_[rmon==7]**2)):7.2f}'
          f'    mean {dw.mean():+7.2f} | {dr_.mean():+7.2f}')
# ---------------- machinery -------------------------------------------------------------------------------------
def simplex(A, t):
    A = A / 1000.0; t = t / 1000.0
    G = A.T @ A; b = A.T @ t; n = A.shape[1]; best = None
    for rr in range(1, n + 1):
        for S in itertools.combinations(range(n), rr):
            S = list(S); k = len(S); K = np.zeros((k + 1, k + 1)); K[:k, :k] = G[np.ix_(S, S)]; K[:k, k] = 1; K[k, :k] = 1
            q = np.zeros(k + 1); q[:k] = b[S]; q[k] = 1
            try: ws = np.linalg.solve(K, q)[:k]
            except np.linalg.LinAlgError: continue
            if (ws < -1e-10).any(): continue
            wf = np.zeros(n); wf[S] = np.clip(ws, 0, None); wf /= wf.sum()
            f = float(wf @ G @ wf - 2 * wf @ b)
            if best is None or f < best[0]: best = (f, wf)
    return best[1]
t = y - SHW
days = np.array(sorted(set(clus))); perm = np.random.default_rng(0).permutation(len(days))
dfold = pd.Series(np.arange(len(days)) % 6, index=days[perm]).reindex(clus).values
APTS = sorted(set(apt))
def fitted(M, scheme):
    """window bulk (without SHIFT) under a weight-fitting scheme, plus the weights used per group"""
    if scheme == 'in':
        w = simplex(M[O], t[O]); return M @ w, [w]
    out = np.zeros(nw); ws = []
    groups = {'day6': [(dfold != k, dfold == k) for k in range(6)],
              'janjul': [(L, J), (J, L)],
              'loao': [(apt != a, apt == a) for a in APTS]}[scheme]
    for fit, app in groups:
        w = simplex(M[O & fit], t[O & fit]); out[app] = M[app] @ w; ws.append(w)
    return out, ws
def boot_se(contrib, seed=5):
    cs = np.bincount(clus, weights=contrib); k = len(cs); rng = np.random.default_rng(seed)
    return float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(B_BOOT)]).std()) * sc
def price_window(new_bulk_noshift):
    B1 = new_bulk_noshift + SHW; new = inc0.copy(); new[appw] = cl(np.rint(inc0[appw] + (B1 - B0)[appw]))
    d = new - inc0; c = -2.0 * (inc0 - y) * d
    return float(c.sum()) * sc, boot_se(c), float((d ** 2).sum()) * sc, float(c[O].sum()) * sc, float(c[ext].sum()) * sc
def rank_new(delta_r):
    p = v99.copy(); p[appr] = cl(np.rint(v99[appr] + delta_r[appr])); return p
ONE = 1.7717e8
# ---------------- step 3 and 4: arms x schemes ------------------------------------------------------------------
ARMS = [('REFIT', SHIP)] + [(f'REFIT+{k}', SHIP + [k]) for k in ('ID', 'RQ', 'CL') if k in MW]
if 'L55o' in MW and 'P55o' in MW:
    ARMS += [('REFIT-OOFte', ['L55o', 'P55o', 'K32', 'K128']), ('REFIT+OOFte', SHIP + ['L55o', 'P55o'])]
FULL = [k for k in ('L55', 'P55', 'K32', 'K128', 'ID', 'RQ', 'CL', 'L55o', 'P55o') if k in MW]
ARMS += [('FULL', FULL)]
if CONTROL == 'trackA': ARMS = [('REFIT', SHIP)]
print(f'\nSTEPS 3-4. weights and pricing on the leak-free base. G in SSE on the scored set; 1 board-s = {ONE:.4e}. FULL = {FULL}'
      + ('' if len(FULL) == 9 else '   <<< FULL IS INCOMPLETE: not the pre-registered package'))
res = {}
for name, ks in ARMS:
    M = np.stack([MW[k] for k in ks], 1); Mr = np.stack([MR[k] for k in ks], 1)
    w_in = simplex(M[O], t[O])
    dr = Mr @ w_in - M0r @ w0; v100 = rank_new(dr); Qb = float(((v100 - v99) ** 2).sum())
    print(f'\n  {name}: members {ks}\n    full-window weights {np.round(w_in, 4)}   ranking change: rows moved {int((v100 != v99).sum()):,}  '
          f'RMS on unfired rows {np.sqrt(np.mean((v100-v99)[appr]**2)):.2f} s   Q_b {Qb:.4e}')
    for sch in ('in', 'day6', 'janjul', 'loao'):
        bn, ws = fitted(M, sch)
        m2T, se, Qw, m2To, m2Tx = price_window(bn)
        G = m2T - Qb
        res[(name, sch)] = dict(G=G, SE=se, m2T=m2T, Qb=Qb, Qw=Qw, w=w_in, ws=ws)
        wtxt = '' if sch == 'in' else '  weights ' + ' | '.join(str(np.round(w, 3)) for w in ws[:2]) + (' ...' if len(ws) > 2 else '')
        print(f'    {sch:<7} ordinary bulk RMSE {r(bn+SHW,O):8.3f} (Jan {r(bn+SHW,O&J):7.3f} Jul {r(bn+SHW,O&L):7.3f})  -2T_w {m2T:+.4e} '
              f'(ord {m2To:+.3e} ext {m2Tx:+.3e})  Q_w {Qw:.3e}  G_hat {G:+.4e}  SE {se:.3e}  G/SE {G/se:+.2f}  = {G/ONE:+.3f} board-s{wtxt}')
print(f'\n  reference: shipped-weight bulk (old) ordinary RMSE {r(B0,O):.3f} (Jan {r(B0,O&J):.3f} Jul {r(B0,O&L):.3f})')
if ('FULL', 'janjul') in res:
    e = res[('FULL', 'janjul')]
    print(f'\nPRE-REGISTERED DECISION (s492 P): FULL package, January|July cross-fit: G_hat {e["G"]:+.4e}  SE {e["SE"]:.4e}  '
          f'G/SE {e["G"]/e["SE"]:+.3f}  -> {"PASS (G_hat >= 2 SE)" if e["G"] >= 2 * e["SE"] else "FAIL (G_hat < 2 SE)"}'
          + ('' if len(FULL) == 9 else '   [FULL INCOMPLETE -- NOT THE DECISION]'))
if CONTROL == 'trackA' or os.environ.get('COV') == '2026': sys.exit(0)
np.savez(f'{P2}/price_out.npz', FULL=np.array(FULL), **{f'{n}__{s}': np.array([v['G'], v['SE'], v['m2T'], v['Qb']]) for (n, s), v in res.items()},
         **{f'w__{n}': res[(n, 'in')]['w'] for n, _ in ARMS})
# ---------------- retrodictions on the same leak-free base -------------------------------------------------------
print('\nRETRODICTIONS on the leak-free base')
V15 = [f'{SP}/trackE/verify', f'{P2}/v15lf']
def v15(model, cs):
    fs = []
    for s in (0, 1, 2):
        c = [f'{d}/win_base_{model}_cs{cs}_s{s}_teleakfree.npy' for d in V15]
        c = [x for x in c if os.path.exists(x)]
        if c: fs.append(c[0])
    return (np.mean([np.load(x) for x in fs], 0), len(fs)) if fs else (None, 0)
(l10, n1), (l04, n2), (p10, n3), (p04, n4) = v15('pooled', '1.0'), v15('pooled', '0.4'), v15('perapt', '1.0'), v15('perapt', '0.4')
bg = I.board_gain
if l10 is not None and l04 is not None and p10 is not None and p04 is not None:
    d99 = 0.2016 * ((l04 - l10) + (p04 - p10))
    new = inc0.copy(); new[appw] = cl(np.rint(inc0[appw] + d99[appw])); dd = new - inc0; c = -2 * (inc0 - y) * dd
    m2T = float(c.sum()) * sc; se = boot_se(c); Qb = float(((v99 - v96) ** 2).sum()); G = m2T - Qb; Bd = bg(256.9600, 256.8806)
    print(f'  v99 (seeds pooled cs1.0 {n1}, cs0.4 {n2}; per-airport cs1.0 {n3}, cs0.4 {n4}): -2T_w {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  '
          f'SE {se:.3e}  board {Bd:+.4e}  (G_hat - board)/SE {(G-Bd)/se:+.2f}')
else:
    print('  v99: leak-free v15-construction trees missing')
vid = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
pos = pd.Series(np.arange(len(vid)), index=vid).reindex(ctx['ids']).values.astype(int)
Dn = np.load(f'{W}/arch_d_window_va.npy')[pos]
b64 = .2473 * MW['L55'] + .2217 * MW['P55'] + .0849 * Dn + .4462 * MW['K32']
b73 = .2230 * MW['L55'] + .2030 * MW['P55'] + .0880 * MW['K32'] + .4860 * MW['K128']
Bd73 = bg(263.0551, 262.8071); Qb73 = float(((v73 - v64) ** 2).sum())
for lab, sh in (('with SHIFT in the base', SHW), ('without SHIFT (as track A)', 0.0)):
    wo, wn = I.mix(ctx, b64 + sh), I.mix(ctx, b73 + sh); c = -2 * (wo - y) * (wn - wo); m2T = float(c.sum()) * sc; se = boot_se(c)
    print(f'  v73 {lab}: -2T_w {m2T:+.4e}  Q_b {Qb73:.4e}  G_hat {m2T-Qb73:+.4e}  SE {se:.3e}  board {Bd73:+.4e}  (G_hat - board)/SE {(m2T-Qb73-Bd73)/se:+.2f}'
          f'   [D member = arch_d_window_va (gate-matrix D net, clean per s487; not rebuilt)]')
