"""s550: v110 = v108 with the full-year ADS-B tree members (p2p_v55_adsb.py: v100's pooled and per-airport recipe plus
v104's movement features, ADS-B from all twelve 2025 months) added to v100's nine-member bulk blend. Simplex weights are
refit on the window's ordinary rows exactly as for v100 (in-sample weights ship; the Jan|Jul cross-fitted weights give the
pre-registered linear term, s492). The bulk change B1 - B0 is applied on top of v108 where v108 is not copy-fired
(|v108 - ms| > 60), in two variants fixed in advance: ALL such rows, or only rows WITHOUT an ADS-B pushback (where the
ADS-B stage does not act, so no double counting). The variant with the higher Jan|Jul G_hat is v110.
  STANDIN=1 uses v100's own L55/P55 as the "new" members (a machinery test: weights and deltas should barely move).
  uv run --with pandas --with pyarrow --with numpy python scripts/adsb_price_v110.py   -> cand/v110_adsb.npz
"""
import os, sys, itertools, numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE); sys.path.insert(0, HERE); import p1a_instrument as I
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
P2 = os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P'
STANDIN = os.environ.get('STANDIN') == '1'
ctx = I.window(); y, ext, vsd = ctx['y'], ctx['ext'], ctx['vsd']; O = ~ext; nw = len(y); sc = I.N_B / nw; cl = I.cl; ONE = 1.7717e8
mon, clus = ctx['month'].astype(int), ctx['cluster']; J, L = mon == 1, mon == 7; SHW = np.where(mon == 1, float(os.environ.get('SHIFT_1', '-8.778')), float(os.environ.get('SHIFT_7', '-3.861')))
mean_of = lambda ps: np.mean([np.load(p) for p in ps], 0)
rk_ids = pd.read_parquet(os.environ.get('ADSB_RK_IDS', f'{W}/v15_rk_jul19.parquet'), columns=['MVT_ID_mvt']).MVT_ID_mvt.values
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
rk2sub = pd.Series(np.arange(len(rk_ids)), index=rk_ids).reindex(sub_ids).values; assert not np.isnan(rk2sub).any(), 'scored ids missing from the id file'; rk2sub = rk2sub.astype(int)
fp = np.load(os.environ.get('DC26_WF', f'{P2}/full_package.npz')); FULL = [str(k) for k in fp['members']]; wF = fp['w'].astype(float)
WIN = {'L55': [f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)], 'P55': [f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)],
       'K32': [f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)], 'K128': [f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)],
       'ID': [f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)], 'RQ': [f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)],
       'CL': [f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)], 'L55o': [f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)],
       'P55o': [f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)]}
MRk = pd.read_parquet(os.environ.get('DC26_MEMBERS_RK', f'{W}/p2_members_rk.parquet')).set_index('MVT_ID_mvt').reindex(sub_ids)
MW = {k: mean_of(WIN[k]) for k in FULL}; MR = {k: MRk[k].values.astype(float) for k in FULL}
if STANDIN:
    MW['L55A'], MW['P55A'] = MW['L55'].copy(), MW['P55'].copy(); MR['L55A'], MR['P55A'] = MR['L55'].copy(), MR['P55'].copy()
else:
    TA = os.environ.get('ADSB_RANK_DIR', f'{P2}/trees_adsb')
    MW['L55A'] = mean_of([f'{TA}/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)]); MW['P55A'] = mean_of([f'{TA}/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)])
    RSL = [int(s) for s in os.environ.get('ADSB_SEEDS_POOLED', '0,1,2').split(',')]   # scored-side seeds of the full-year
    RSP = [int(s) for s in os.environ.get('ADSB_SEEDS_PERAPT', '0,1,2').split(',')]   # members (s564)
    MR['L55A'] = mean_of([f'{TA}/rank_stored_pooled_s{s}.npy' for s in RSL])[rk2sub]; MR['P55A'] = mean_of([f'{TA}/rank_stored_perapt_s{s}.npy' for s in RSP])[rk2sub]
NEW = FULL + ['L55A', 'P55A']
for k in NEW: assert np.isfinite(MW[k]).all() and np.isfinite(MR[k]).all(), k
r = lambda p, m: float(np.sqrt(np.mean((cl(p)[m] - y[m]) ** 2)))
print(f'window ordinary RMSE: L55 {r(MW["L55"], O):.3f}  L55A {r(MW["L55A"], O):.3f}   P55 {r(MW["P55"], O):.3f}  P55A {r(MW["P55A"], O):.3f}')
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
t = y - SHW; M0w = np.stack([MW[k] for k in FULL], 1); M0r = np.stack([MR[k] for k in FULL], 1)
B0w = M0w @ wF
z8 = np.load(f'{I.CAND}/v108_adsb.npz'); v8w, v8r = z8['newW'].astype(float), I.sub('v108')
assert (z8['win_ids'] == ctx['ids']).all() and np.array_equal(z8['new'].astype(float), v8r)
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(sub_ids)
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
msr = cl(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt))
F4 = pd.read_parquet(os.environ.get('ADSB_FEAT_V104', f'{REPO}/artifacts/adsb_push_feat_v104.parquet')).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
hasw, hasr = F4.ab_push.reindex(ctx['ids']).notna().values, F4.ab_push.reindex(sub_ids).notna().values
VAR = (('ALL unfired rows', np.abs(v8w - vsd) > 60, np.abs(v8r - msr) > 60),
       ('rows WITHOUT ADS-B only', (np.abs(v8w - vsd) > 60) & ~hasw, (np.abs(v8r - msr) > 60) & ~hasr))
def price(tag, extra):
    """extra: the two added members (window, ranking). Returns {variant: (G, SE, nW, nR)}"""
    names = FULL + ['L55A', 'P55A']; M1w = np.column_stack([M0w] + [extra[k][0] for k in ('L55A', 'P55A')]); M1r = np.column_stack([M0r] + [extra[k][1] for k in ('L55A', 'P55A')])
    w_in = simplex(M1w[O], t[O]); B1w = np.zeros(nw); wcf = []
    for fit, app in ((L, J), (J, L)):
        w_ = simplex(M1w[O & fit], t[O & fit]); B1w[app] = M1w[app] @ w_; wcf.append(np.round(w_, 3))
    print(f'\n[{tag}] weights in-sample: ' + '  '.join(f'{k} {w:.4f}' for k, w in zip(names, w_in)) + '\n[' + tag + '] Jan|Jul: ' + ' | '.join(str(w) for w in wcf))
    dW = B1w - B0w; dR = M1r @ w_in - M0r @ wF; out = {}
    for name, aw, ar in VAR:
        nW = v8w.copy(); nW[aw] = cl(np.rint(v8w[aw] + dW[aw])); nR = v8r.copy(); nR[ar] = cl(np.rint(v8r[ar] + dR[ar]))
        lin = -2.0 * (v8w - y) * (nW - v8w); own = (v8w - y) ** 2 - (nW - y) ** 2; Qb = float(((nR - v8r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
        rng = np.random.default_rng(5); cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own); k = len(cs); ii = [rng.integers(0, k, k) for _ in range(2000)]
        SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
        print(f'[{tag}] {name}: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e})  rows R {int((nR != v8r).sum()):,}  max |change| {np.abs(nR - v8r).max():.0f} s\n'
              f'      -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  -> {np.sqrt(231.7292**2 - G/344841):.4f}')
        out[name] = (G, SE, nW, nR, w_in)
    return out
null = price('NULL: v100 own L55/P55 as the added members', {'L55A': (MW['L55'], MR['L55']), 'P55A': (MW['P55'], MR['P55'])})
if STANDIN: sys.exit(0)
real = price('ADS-B members', {'L55A': (MW['L55A'], MR['L55A']), 'P55A': (MW['P55A'], MR['P55A'])})
res = {k: real[k] for k in real}; adj = {k: real[k][0] - null[k][0] for k in real}
for k in adj: print(f'  {k}: G_hat {real[k][0]:+.4e}  minus its null {null[k][0]:+.4e}  = {adj[k]:+.4e}')
pick = max(adj, key=adj.get); print(f'\nPRE-REGISTERED RULE (highest G_hat minus its null) -> v110 = variant "{pick}"')
np.savez(f'{I.CAND}/v110_adsb.npz', ids=sub_ids, v108=v8r, new=res[pick][3], win_ids=ctx['ids'], newW=res[pick][2], w_in=res[pick][4], members=np.array(NEW), variant=pick)
