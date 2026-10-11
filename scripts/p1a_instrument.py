"""p1a: an offline estimator of the BOARD SSE change of a candidate, and its retrodiction record.

USAGE AS A LIBRARY
    import p1a_instrument as I
    ctx = I.window_v55()            # FAITHFUL base: window versions of the tree members that ship (use this)
    #   I.window()                  # the RECORDED instrument (gate_stage_a2 trees, the 313.8931 control)
    est = I.bulk(ctx, win_old, win_new, rank_old, rank_new)         # any bulk change, full vectors
    est = I.bulk_from_deltas(ctx, win_delta, rank_delta)            # an additive bulk delta on top of the
                                                                    # standing (v99), shipped v99-style
    est = I.cell(rank_old, rank_new, rows, p_copy, copy_val, alt_val, alt_samples)  # labelled-prior commitment
  win_old / win_new    the candidate's DEPLOYED-LEVEL predictions on the 344,336 window rows (v15 order,
                       window_ids.npz): after the mixture, after the candidate's own firing rule is replayed
                       on the window. Never a component RMSE (s484's 1.442e8 was 0.2016 x component gains).
  rank_old / rank_new  the two submission vectors on the 344,841 scored rows (submitting order).

WHAT IT RETURNS, and why (each measured in track A):
  Q_b        sum((rank_new - rank_old)^2), EXACT, no labels. The board pays it in full. On the neural-member
             changes v64 and v73 it was 1.44x and 1.47x the window's Q_w; on the tree-only v99, 1.055x.
  m2T_w      -2 * sum((win_old - y) * (win_new - win_old)) over ALL window rows, ordinary AND extreme.
  G_hat      m2T_w * N_b/N_w - Q_b ("Q-anchored"). On the faithful base the linear term transferred at
             1.038 / 1.144 / 1.361 (v64 / v73 / v99) and every bulk retrodiction is inside 1 SE.
  SE         day-cluster bootstrap SE of the linear term (62 days). 2.4e7 SSE for a v99-sized tree
             delta, 4.2e7 for v73, 1.1e8 for v64: |G_hat| < 2 SE is not predictable in sign.
  cell mode  labelled-prior outcome model resampled row by row: E[G], the simulated median and P(G>0).
             Reproduces v96 (dSSE == -Q on the board) at 0.998x.

RUN AS A SCRIPT: prints the retrodiction table for v64, v73, v99 on three window bases, and v95, v96.
"""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.environ['DC26_WORK']; DATA = os.environ['DC26_DATA']
OUT = os.environ.get('P1A_OUT', (os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackA'))
FLOOR, CAP, DAY, N_B = 60., 200000., 86400., 344841
cl = lambda p: np.clip(p, FLOOR, CAP)

def window_labels(ids):
    """The window's labels, read from the challenge extract by movement id. The public release ships the window
    files without labels (it does not redistribute challenge data); where the files carry them they are equal."""
    fs = sorted(glob.glob(f'{DATA}/training_2025-0[17]-01_*.parquet')); assert len(fs) == 2, fs
    T = pd.concat([pq.read_table(f, columns=['MVT_ID_mvt', 'PHASE_mvt', 'TAXITIME_SEC_mvt']).to_pandas() for f in fs])
    T = T[T.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    y = T.reindex(ids).values.astype(float); assert not np.isnan(y).any(), 'window id missing from the extract'
    return y

def window():
    wi = np.load(f'{OUT}/window_ids.npz', allow_pickle=True)
    c = np.load(f'{W}/b4ab_ctx.npz')
    P = np.load(f'{W}/b4ab_P_u.npy'); Qg = P ** 0.90; Qg = Qg / Qg.sum(1, keepdims=True)
    Qg[:, 1:][Qg[:, 1:] < 0.10] = 0.0; Qg[:, 0] = 1.0 - Qg[:, 1:].sum(1)
    Qg = np.clip(Qg, 1e-12, None); Qg = Qg / Qg.sum(1, keepdims=True)
    y = c['yv'].astype(float) if 'yv' in c.files else window_labels(wi['mvt_id'])
    ext = c['ext'].astype(bool) if 'ext' in c.files else y >= 7200.
    ctx = dict(y=y, ext=ext, vsd=c['vsd'], vlb=c['vlb'], bulk=c['bulkv'],
               Qg=Qg, month=wi['month'], day=wi['day'], apt=wi['apt'], ids=wi['mvt_id'])
    ctx['inc'] = mix(ctx, ctx['bulk'])
    r = float(np.sqrt(np.mean((ctx['inc'] - ctx['y']) ** 2)))
    assert abs(r - 313.8931) < 5e-4, f'deployed control {r:.4f} != 313.8931'
    ctx['cluster'] = pd.factorize(pd.Series(ctx['month']).astype(str) + '-' + pd.Series(ctx['day']).astype(str))[0]
    return ctx

def window_v15(ctx=None):
    """The same window and mixture routing, but with the bulk blend rebuilt from the SHIPPED tree
    construction (v15_tr_jul19, 71 features, p1a_window_components.py, colsample 1.0) in place of
    gate_stage_a2's members, which lack six shipped features (since_inblk, type_match, prev_taxi_in,
    arr_stand_mo, arr_rwy_mo, arr_apt_mo) and score 5-6 s worse per member on the window."""
    ctx = dict(ctx or window())
    ld = lambda t, cs: np.mean([np.load(f'{OUT}/wc/{t}_cs{cs:.1f}_s{s}.npy') for s in (0, 1, 2)], 0)
    wm = _members(ctx)
    wm['LGB'], wm['PA'] = ld('pooled', 1.0), ld('perapt', 1.0)
    ctx['members'] = wm
    ctx['bulk'] = .2230 * wm['LGB'] + .2030 * wm['PA'] + .0880 * wm['D2'] + .4860 * wm['K128']
    ctx['inc'] = mix(ctx, ctx['bulk'])
    return ctx

def window_v55(ctx=None):
    """The FAITHFUL base: window-side versions of the tree members that actually ship (v55_trees.parquet:
    pooled at the v38 recipe with feature_fraction .7 and OPDI features, per-airport 600/127/100 with OPDI),
    built by p1a_window_v55.py on the ten non-window months. Returns None until those arrays exist."""
    fs = [f'{OUT}/wc/v55_{t}_s{s}.npy' for t in ('pooled', 'perapt') for s in (0, 1, 2)]
    if not all(os.path.exists(f) for f in fs): return None
    ctx = dict(ctx or window())
    ld = lambda t: np.mean([np.load(f'{OUT}/wc/v55_{t}_s{s}.npy') for s in (0, 1, 2)], 0)
    wm = _members(ctx); wm['LGB'], wm['PA'] = ld('pooled'), ld('perapt')
    ctx['members'] = wm
    ctx['bulk'] = .2230 * wm['LGB'] + .2030 * wm['PA'] + .0880 * wm['D2'] + .4860 * wm['K128']
    ctx['inc'] = mix(ctx, ctx['bulk'])
    return ctx

def window_v55_2026cov(ctx=None):
    """The faithful base with the SCORED SET's OPDI coverage (57.0% January, 84.3% July, against the window's
    87.0% / 91.0%): the seed-0 masked-minus-unmasked difference from p1a_window_v55_masked.py is added to each
    tree member. Returns None until that file exists."""
    fn = f'{OUT}/wc/v55_mask_s0.npz'
    base = window_v55(ctx)
    if base is None or not os.path.exists(fn): return None
    z = np.load(fn); wm = dict(base['members'])
    wm['LGB'] = wm['LGB'] + z['d_pooled']; wm['PA'] = wm['PA'] + z['d_perapt']
    base['members'] = wm
    base['bulk'] = .2230 * wm['LGB'] + .2030 * wm['PA'] + .0880 * wm['D2'] + .4860 * wm['K128']
    base['inc'] = mix(base, base['bulk'])
    return base

def _members(ctx):
    z = np.load(f'{W}/gate_stage_a2.npz'); zm = np.load(f'{W}/gate_mlp.npz')
    vid = pq.read_table(f'{W}/night_va_all.parquet', columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
    pos = pd.Series(np.arange(len(vid)), index=vid).reindex(ctx['ids']).values.astype(int)
    assert np.array_equal(z['yv'][pos], ctx['y'])
    return {'LGB': z['lgb_va'][pos], 'PA': z['pa_va'][pos], 'MLP': zm['mlp_va'][pos],
            'D': np.load(f'{W}/arch_d_window_va.npy')[pos], 'D2': np.load(f'{W}/arch_d2_wide_va_K32.npy')[pos],
            'K128': np.load(f'{W}/arch_d2_wide_va_K128.npy')[pos]}

def v99_rule(ctx, delta):
    """scripts/v99_build.py replayed on the window: add delta where |inc - ms| > 60, rint, clip."""
    inc = ctx['inc']; app = np.abs(inc - ctx['vsd']) > 60
    pv = inc.copy(); pv[app] = cl(np.rint(inc[app] + delta[app])); return pv

def mix(ctx, bulk):
    b = cl(bulk); V = np.stack([b, ctx['vsd'], ctx['vlb'], cl(DAY + b), b], 1)
    return cl(np.rint((ctx['Qg'] * V).sum(1)))

def _terms(ctx, old, new, m):
    d = (new - old)[m]; e = (old - ctx['y'])[m]
    return -2.0 * float((e * d).sum()), float((d ** 2).sum())

def bulk(ctx, win_old, win_new, rank_old, rank_new, B=400, seed=0):
    y, ext = ctx['y'], ctx['ext']; nw = len(y); allm = np.ones(nw, bool)
    m2T, Qw = _terms(ctx, win_old, win_new, allm)
    m2T_o, Qw_o = _terms(ctx, win_old, win_new, ~ext); m2T_x, Qw_x = _terms(ctx, win_old, win_new, ext)
    Qb = float(((np.asarray(rank_new, float) - np.asarray(rank_old, float)) ** 2).sum())
    # day-cluster bootstrap of the linear term
    contrib = -2.0 * (win_old - y) * (win_new - win_old)
    cs = np.bincount(ctx['cluster'], weights=contrib); k = len(cs)
    co = np.bincount(ctx['cluster'], weights=np.where(ext, 0.0, contrib), minlength=k)
    rng = np.random.default_rng(seed); ii = [rng.integers(0, k, k) for _ in range(B)]
    bs = np.array([cs[i].sum() for i in ii]); bo = np.array([co[i].sum() for i in ii])
    sc = N_B / nw
    return dict(m2T_w=m2T, m2T_w_ord=m2T_o, m2T_w_ext=m2T_x, Q_w=Qw, Q_w_ord=Qw_o, Q_w_ext=Qw_x, Q_b=Qb,
                G_window=(m2T - Qw) * sc, G_hat=m2T * sc - Qb, SE=float(bs.std()) * sc, SE_ord=float(bo.std()) * sc,
                G_hat_ordlin=m2T_o * sc - Qb, G_window_ord_only=(m2T_o - Qw_o) * sc)

def bulk_from_deltas(ctx, win_delta, rank_delta, rank_base=None, win_base=None):
    """An additive bulk delta applied the way v99 shipped it: to rows whose deployed prediction is not within
    60 s of ms = MVT - SCHED, then rint and clip, on BOTH sides. rank_base defaults to the standing (v99);
    win_base defaults to v99's rule replayed on ctx with its own delta, i.e. the standing on the window."""
    if rank_base is None: rank_base = sub('v99')
    if win_base is None:
        ld = lambda t, cs: np.mean([np.load(f'{OUT}/wc/{t}_cs{cs:.1f}_s{s}.npy') for s in (0, 1, 2)], 0)
        win_base = v99_rule(ctx, 0.2016 * ((ld('pooled', 0.4) - ld('pooled', 1.0)) + (ld('perapt', 0.4) - ld('perapt', 1.0))))
    wa = np.abs(win_base - ctx['vsd']) > 60
    wn = win_base.copy(); wn[wa] = cl(np.rint(win_base[wa] + win_delta[wa]))
    ids = pq.read_table(f'{DATA}/submitting.parquet').to_pandas().MVT_ID_mvt.values
    R = pq.read_table(f'{DATA}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
    R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(ids)
    E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
    ms = cl(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt)); ra = np.abs(rank_base - ms) > 60
    rn = np.asarray(rank_base, float).copy(); rn[ra] = cl(np.rint(rank_base[ra] + rank_delta[ra]))
    return bulk(ctx, win_base, wn, rank_base, rn)

def cell(rank_old, rank_new, rows, p_copy, copy_val, alt_val, alt_samples=None, sims=20000, seed=0):
    """A commitment of rows to new values. Outcome model per row: with prob p the truth IS copy_val
    (a copy), otherwise it is an ordinary value drawn from alt_samples (the labelled non-copy y) or
    fixed at alt_val. Returns E[G] and the SIMULATED median and P(G>0), which for a small cell is
    the number to quote: v95 moved 8 rows on 28 labelled ones, one of them day-scale."""
    o = np.asarray(rank_old, float)[rows]; n_ = np.asarray(rank_new, float)[rows]
    c = np.broadcast_to(np.asarray(copy_val, float), o.shape); p = np.broadcast_to(np.asarray(p_copy, float), o.shape)
    Qb = float(((np.asarray(rank_new, float) - np.asarray(rank_old, float)) ** 2).sum())
    rng = np.random.default_rng(seed)
    iscopy = rng.random((sims, len(o))) < p
    if alt_samples is not None:
        a = np.asarray(alt_samples, float)[rng.integers(0, len(alt_samples), (sims, len(o)))]
    else:
        a = np.broadcast_to(np.asarray(alt_val, float), (sims, len(o)))
    yy = np.where(iscopy, c, a)
    G = ((o - yy) ** 2 - (n_ - yy) ** 2).sum(1)
    return dict(G_hat=float(G.mean()), G_med=float(np.median(G)), P_pos=float((G > 0).mean()), Q_b=Qb)

def board_gain(before, after): return N_B * (before ** 2 - after ** 2)
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions')   # clean mode reads its own chain of submissions
CAND = os.environ.get('DC26_CAND', f'{W}/adsblol/cand')   # and keeps its own pricing candidates
CLEAN = os.environ.get('CLEAN') == '1'
if CLEAN:   # the defaults are the submitted chain's; a CLEAN run must name its own inputs (s561)
    _miss = [k for k in ('DC26_SUBDIR', 'DC26_CAND', 'SHIFT_1', 'SHIFT_7', 'DC26_WF') if not os.environ.get(k)]
    assert not _miss, f'CLEAN=1 needs {_miss} set (scripts/regen_clean.sh sets them)'
    assert os.path.realpath(SUBDIR) != os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'submissions')), \
        'CLEAN=1 must not read or write the repository submissions/'
    assert os.path.realpath(CAND) != os.path.realpath(f'{W}/adsblol/cand'), 'CLEAN=1 must not use the submitted chain candidates'
def sub(v): return pq.read_table(f'{SUBDIR}/quick-boat_{v}.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)

def _lirf(paths, cols):
    r = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in paths], ignore_index=True)
    for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'):
        if c in r: r[c] = pd.to_datetime(r[c], errors='coerce', utc=True)
    d = r[r.PHASE_mvt.eq('DEP') & r.ADEP_mvt.eq('LIRF')].copy()
    d['ms'] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    return d

def main():
    ctx = window(); y = ctx['y']
    print(f'CONTROL deployed mixture on the window 313.8931 reproduced; rows {len(y):,}, extremes {int(ctx["ext"].sum())}')
    WS = {'v59': {'LGB': .2, 'MLP': .4, 'PA': .4},
          'v64': {'LGB': .2473, 'PA': .2217, 'D': .0849, 'D2': .4462},
          'v73': {'LGB': .2230, 'PA': .2030, 'D2': .0880, 'K128': .4860}}
    DECL = {'v64': 4.5721e8, 'v73': 1.5247e8, 'v99': 1.442e8}   # s247 (+2.4816 x PER), s281 (+0.974 OOF-s), s484
    BOARD = {'v64': board_gain(264.3314, 263.0551), 'v73': board_gain(263.0551, 262.8071), 'v99': board_gain(256.9600, 256.8806)}
    ld = lambda t, cs: np.mean([np.load(f'{OUT}/wc/{t}_cs{cs:.1f}_s{s}.npy') for s in (0, 1, 2)], 0)
    d99 = 0.2016 * ((ld('pooled', 0.4) - ld('pooled', 1.0)) + (ld('perapt', 0.4) - ld('perapt', 1.0)))
    out = {}
    bases = [('RECORDED instrument (gate_stage_a2 trees, 313.8931)', ctx),
             ('V15 trees (v99 as a CLEAN in-base replacement)', window_v15(ctx))]
    v55 = window_v55(ctx)
    if v55 is not None: bases.append(('FAITHFUL base (window v55-equivalent trees, as shipped)', v55))
    v55c = window_v55_2026cov(ctx)
    if v55c is not None: bases.append(('FAITHFUL base at the SCORED SET OPDI coverage', v55c))
    for base_name, cx in bases:
        wm = cx.get('members') or _members(cx)
        bl = lambda w: sum(v * wm[k] for k, v in w.items())
        r_inc = float(np.sqrt(np.mean((cx['inc'] - y) ** 2))); r_ord = float(np.sqrt(np.mean((cl(cx['bulk']) - y)[~cx['ext']] ** 2)))
        rows = []
        for old, new in (('v59', 'v64'), ('v64', 'v73')):
            rows.append((new, bulk(cx, mix(cx, bl(WS[old])), mix(cx, bl(WS[new])), sub(old), sub(new))))
        rows.append(('v99', bulk(cx, cx['inc'], v99_rule(cx, d99), sub('v96'), sub('v99'))))
        out[base_name] = rows
        print(f'\nBULK MODE on the {base_name}: deployed RMSE {r_inc:.4f}, bulk ordinary {r_ord:.3f}')
        print(f'  G = -(dSSE), SSE on the 344,841 scored rows (window terms x N_b/N_w)')
        print(f'  {"case":5} {"declared":>11} {"G_window":>11} {"G_hat":>11} {"SE":>10} {"BOARD":>11} {"(hat-B)/SE":>10}   '
              f'{"decl/B":>7} {"win/B":>7} {"hat/B":>7}   {"-2T_w":>10} {"-2T_b":>10} {"Q_w":>10} {"Q_b":>10}')
        for nm, e in rows:
            gb = BOARD[nm]; m2Tb = gb + e['Q_b']; sc = N_B / len(y)
            print(f'  {nm:5} {DECL[nm]:11.4e} {e["G_window"]:11.4e} {e["G_hat"]:11.4e} {e["SE"]:10.3e} {gb:11.4e} {(e["G_hat"]-gb)/e["SE"]:+10.2f}   '
                  f'{DECL[nm]/gb:7.2f} {e["G_window"]/gb:7.2f} {e["G_hat"]/gb:7.2f}   {e["m2T_w"]*sc:10.3e} {m2Tb:10.3e} '
                  f'{e["Q_w"]*sc:10.3e} {e["Q_b"]:10.3e}')
            print(f'        window split: -2T ord {e["m2T_w_ord"]*sc:+.3e} ext {e["m2T_w_ext"]*sc:+.3e} | Q ord {e["Q_w_ord"]*sc:.3e} ext {e["Q_w_ext"]*sc:.3e}'
                  f' | -2T_b/-2T_w {m2Tb/(e["m2T_w"]*sc):.3f}  Q_b/Q_w {e["Q_b"]/(e["Q_w"]*sc):.3f}'
                  f' | ordinary-only linear: G_hat {e["G_hat_ordlin"]:.3e} (SE {e["SE_ord"]:.2e})')
        tot_hat = sum(e['G_hat'] for _, e in rows); tot_b = sum(BOARD[nm] for nm, _ in rows); tot_d = sum(DECL[nm] for nm, _ in rows)
        se = float(np.sqrt(sum(e['SE'] ** 2 for _, e in rows)))
        print(f'  POOLED over the three: declared {tot_d:.4e} ({tot_d/tot_b:.2f}x)   G_hat {tot_hat:.4e} +- {se:.2e} ({tot_hat/tot_b:.2f}x)   board {tot_b:.4e}')
    # ---- v95 and v96: labelled-prior cell commitments (positive control) ----
    C = ['MVT_ID_mvt', 'ADEP_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt',
         'LOBT_flt', 'STAND_mvt', 'FLIGHT_mvt']
    TR = sorted(glob.glob(f'{DATA}/training_*.parquet'))
    L = _lirf(TR, C); L['y'] = L.TAXITIME_SEC_mvt.astype(float)
    R = _lirf([f'{DATA}/ranking.parquet'], C)
    ids = pq.read_table(f'{DATA}/submitting.parquet').to_pandas().MVT_ID_mvt.values
    pos = pd.Series(np.arange(len(ids)), index=ids)
    # v95: stand-4xx cell, moved to the labelled median; alternative = the labelled y distribution
    S = L[L.LOBT_flt.isna() & (L.ms >= 7200) & L.STAND_mvt.astype(str).str.startswith('4')]
    o91, o95 = sub('v91'), sub('v95'); mv = np.where(o95 != o91)[0]
    Ey = float(S.y.mean())
    c95 = cell(o91, o95, mv, 0.0, 0.0, None, alt_samples=S.y.values)
    # v96: eleven carriers committed to ms, p = pooled labelled copy rate, alternative = non-copy median
    L['cp'] = np.abs(L.y - L.ms) <= 60; L['car'] = L.FLIGHT_mvt.astype(str).str[:3]
    A = L[L.ms > 600]; g = A.groupby('car').agg(n=('cp', 'size'), cp=('cp', 'mean'), mons=('MVT_TIME_UTC_mvt', lambda s: s.dt.month.nunique()))
    HOT = sorted(g[(g.n >= 25) & (g.cp >= 0.95) & (g.mons >= 5)].index)
    H = A[A.car.isin(HOT)]; P = float(H.cp.mean()); ALT = float(H[~H.cp].y.median())
    o96 = sub('v96'); mv6 = np.where(o96 != o95)[0]
    msr = R.set_index('MVT_ID_mvt').ms.reindex(ids[mv6]).values
    c96 = cell(o95, o96, mv6, P, np.clip(msr, FLOOR, CAP), ALT, alt_samples=H[~H.cp].y.values)
    # declared before firing: v95_build.py claims the Q (rows moved to the stand's own level, 0/28 copies);
    # s395 records v96's published predicted median 256.9619
    cases = [('v95', c95, c95['Q_b'], board_gain(258.1975, 257.9390), f'{len(S)} labelled rows, 0 copies, E[y] {Ey:,.0f}, median {S.y.median():,.0f}; {len(mv)} rows moved'),
             ('v96', c96, board_gain(257.9390, 256.9619), board_gain(257.9390, 256.9600), f'{len(HOT)} carriers, {len(H):,} labelled rows, p {P:.4f}; {len(mv6)} rows moved')]
    print(f'\nCELL MODE (labelled prior on the edited population; outcomes resampled row by row, 20,000 draws)')
    print(f'  {"case":5} {"declared":>11} {"Q_b":>11} {"E[G]":>11} {"median G":>11} {"P(G>0)":>7} {"BOARD":>11} {"decl/B":>7} {"E/B":>7} {"med/B":>7} {"B/Q_b":>6}')
    for nm, cc, dec, gb, note in cases:
        print(f'  {nm:5} {dec:11.4e} {cc["Q_b"]:11.4e} {cc["G_hat"]:11.4e} {cc["G_med"]:11.4e} {cc["P_pos"]:7.3f} {gb:11.4e} '
              f'{dec/gb:7.3f} {cc["G_hat"]/gb:7.3f} {cc["G_med"]/gb:7.3f} {gb/cc["Q_b"]:6.3f}   {note}')

if __name__ == '__main__':
    main()
