"""s551: why the pre-registered v110 lost on the window although its full-year member (L55A) beats v100's L55 by 19 s alone.
Reuses adsb_price_v110.py's setup unchanged (same members, window, v108 vectors, fired mask), then measures on ordinary
window rows split by ADS-B coverage:
  1. RMSE of v108, of v100's bulk blend, of the new blend (global in-sample weights), and of L55A alone
  2. variant U2: the simplex refit ON THE ROWS IT IS APPLIED TO (unfired rows without ADS-B), cross-fitted Jan|Jul for
     the window vector, in-sample for the scored side; priced exactly as adsb_price_v110.price() prices
  SAVE=1 writes cand/v110b_adsb.npz in v110_build.py's layout.
  uv run --with pandas --with pyarrow --with numpy python scripts/adsb_v110_diag.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, 'adsb_price_v110.py')).read().split('def price(')[0])
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
unf_w = np.abs(v8w - vsd) > 60; unf_r = np.abs(v8r - msr) > 60
M1w = np.column_stack([M0w, MW['L55A'], MW['P55A']]); M1r = np.column_stack([M0r, MR['L55A'], MR['P55A']]); names = FULL + ['L55A', 'P55A']
w_g = simplex(M1w[O], t[O]); B1g = cl(M1w @ w_g + SHW); B0 = cl(B0w + SHW)
print(f'coverage: window {hasw.mean():.1%} of rows, scored {hasr.mean():.1%}')
print(f'{"ordinary rows":24s} {"n":>8s} {"v108":>8s} {"v100 bulk":>10s} {"new blend":>10s} {"L55A":>8s} {"L55":>8s}')
for nm, m in (('with ADS-B', O & hasw), ('without ADS-B', O & ~hasw), ('without, unfired', O & ~hasw & unf_w), ('all', O)):
    print(f'{nm:24s} {int(m.sum()):8d} {rm(v8w, m):8.2f} {rm(B0, m):10.2f} {rm(B1g, m):10.2f} {rm(cl(MW["L55A"] + SHW), m):8.2f} {rm(cl(MW["L55"] + SHW), m):8.2f}')
fitU = O & ~hasw
w_u = simplex(M1w[fitU], t[fitU]); B1u = np.zeros(nw); wcf = []
for fit, app in ((L, J), (J, L)):
    w_ = simplex(M1w[fitU & fit], t[fitU & fit]); B1u[app] = M1w[app] @ w_; wcf.append(np.round(w_, 3))
print('\nU2 weights in-sample (fit on ordinary rows without ADS-B): ' + '  '.join(f'{k} {w:.4f}' for k, w in zip(names, w_u)) + '\nU2 Jan|Jul: ' + ' | '.join(str(w) for w in wcf))
aw, ar = unf_w & ~hasw, unf_r & ~hasr
dW = B1u - B0w; dR = M1r @ w_u - M0r @ wF
nW = v8w.copy(); nW[aw] = cl(np.rint(v8w[aw] + dW[aw])); nR = v8r.copy(); nR[ar] = cl(np.rint(v8r[ar] + dR[ar]))
lin = -2.0 * (v8w - y) * (nW - v8w); own = (v8w - y) ** 2 - (nW - y) ** 2; Qb = float(((nR - v8r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
rng = np.random.default_rng(5); cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own); k = len(cs); ii = [rng.integers(0, k, k) for _ in range(2000)]
SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
print(f'U2 rows WITHOUT ADS-B, weights refit there: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e})  rows R {int((nR != v8r).sum()):,}  max |change| {np.abs(nR - v8r).max():.0f} s\n'
      f'      -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  -> {np.sqrt(231.7292**2 - G/344841):.4f}')
# ---- variant C: on covered unfired rows REPLACE v108's (v100 bulk + ADS-B stage) by v100 with the new bulk and no stage
w0 = np.array([.2230, .2030, .0880, .4860]); B00 = np.stack([MW[k] for k in ('L55', 'P55', 'K32', 'K128')], 1) @ w0 + SHW
inc0 = I.mix(ctx, B00); app0 = np.abs(inc0 - vsd) > 60
v100w = inc0.copy(); v100w[app0] = cl(np.rint(inc0[app0] + (M0w @ wF + SHW - B00)[app0])); v100r = I.sub('v100')
print(f'\nv100 on the window rebuilt: RMSE {rm(v100w, np.ones(nw, bool)):.4f}; v108 == v100 on {np.mean(v8w == v100w):.1%} of window rows, '
      f'{np.mean(v8w[~hasw] == v100w[~hasw]):.1%} of uncovered; scored side {np.mean(v8r == v100r):.1%}, uncovered {np.mean(v8r[~hasr] == v100r[~hasr]):.1%}')
fitC = O & hasw; w_c = simplex(M1w[fitC], t[fitC]); B1c = np.zeros(nw); wcc = []
for fit, app in ((L, J), (J, L)):
    w_ = simplex(M1w[fitC & fit], t[fitC & fit]); B1c[app] = M1w[app] @ w_; wcc.append(np.round(w_, 3))
print('C weights in-sample (fit on ordinary rows WITH ADS-B): ' + '  '.join(f'{k} {w:.4f}' for k, w in zip(names, w_c)) + '\nC Jan|Jul: ' + ' | '.join(str(w) for w in wcc))
awc, arc = unf_w & hasw, unf_r & hasr
nWc = v8w.copy(); nWc[awc] = cl(np.rint(v100w[awc] + (B1c - M0w @ wF)[awc])); nRc = v8r.copy(); nRc[arc] = cl(np.rint(v100r[arc] + (M1r @ w_c - M0r @ wF)[arc]))
for nm, m in (('covered ordinary', O & hasw), ('covered ordinary unfired', O & awc), ('all ordinary', O), ('all rows', np.ones(nw, bool))):
    print(f'  {nm:26s} v108 {rm(v8w, m):8.2f}  C {rm(nWc, m):8.2f}')
lin = -2.0 * (v8w - y) * (nWc - v8w); own = (v8w - y) ** 2 - (nWc - y) ** 2; Qb = float(((nRc - v8r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own)
SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
print(f'C covered rows, v100 + new bulk, no stage: window own {own.sum()*sc:+.4e} (paired SE {SEo:.2e})  rows R {int((nRc != v8r).sum()):,}  max |change| {np.abs(nRc - v8r).max():.0f} s\n'
      f'      -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  -> {np.sqrt(231.7292**2 - G/344841):.4f}')
aptw = ctx['apt']; print('  per airport window own (C vs v108): ' + '  '.join(f'{a} {own[aptw == a].sum()*sc:+.2e}' for a in pd.unique(aptw)))
top = np.argsort(-np.abs(own))[:12]
print('  largest row changes on the window (y, v108, C, own gain):'); [print(f'    {aptw[i]}  y {y[i]:8.0f}  v108 {v8w[i]:8.0f}  C {nWc[i]:8.0f}  {own[i]:+.3e}') for i in top]
# ---- C family, rule fixed before these were run: C, C without LIRF (s546: LIRF's covered errors are record errors), each
# also as v108 + a (C - v108) with a Jan|Jul cross-fitted; pick the highest G_hat among variants with G/SE > 1
aptr = pd.read_parquet(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.values  # (ids only, for the order check)
Rapt = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt']).to_pandas()
Rapt = Rapt[Rapt.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').ADEP_mvt.reindex(sub_ids).values
LW, LR = aptw == 'LIRF', Rapt == 'LIRF'; assert 0.05 < LW.mean() < 0.12 and 0.05 < LR.mean() < 0.12
def G_of(nW_, nR_):
    lin = -2.0 * (v8w - y) * (nW_ - v8w); own = (v8w - y) ** 2 - (nW_ - y) ** 2; Qb = float(((nR_ - v8r) ** 2).sum()); m2T = float(lin.sum()) * sc
    cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own)
    SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
    return own.sum() * sc, SEo, m2T, Qb, m2T - Qb, SE
fam = {}
for tag, mw_, mr_ in (('C', awc, arc), ('C no LIRF', awc & ~LW, arc & ~LR)):
    nW_ = v8w.copy(); nW_[mw_] = nWc[mw_]; nR_ = v8r.copy(); nR_[mr_] = nRc[mr_]; fam[tag] = (nW_, nR_)
    dW_, dR_ = nW_ - v8w, nR_ - v8r; e = v8w - y; acf = np.zeros(nw)
    a_in = float(-(e * dW_).sum() / (dW_ ** 2).sum())
    for fit, app in ((L, J), (J, L)):
        a_ = float(-(e * dW_)[fit].sum() / (dW_[fit] ** 2).sum()); acf[app] = a_
    fam[tag + ' alpha'] = (cl(np.rint(v8w + acf * dW_)), cl(np.rint(v8r + a_in * dR_)))
    print(f'  {tag}: alpha in-sample {a_in:.3f}, Jan-fit {acf[L][0] if L.any() else 0:.3f} (applied to Jul), Jul-fit {acf[J][0]:.3f} (applied to Jan)')
res = {}
for tag, (nW_, nR_) in fam.items():
    o_, so_, m2T_, Qb_, G_, SE_ = G_of(nW_, nR_); res[tag] = (G_, SE_)
    print(f'  {tag:16s} own {o_:+.4e} ({o_/so_:+.2f} SE)  -2T_w*sc {m2T_:+.4e}  Q_b {Qb_:.4e}  G_hat {G_:+.4e}  G/SE {G_/SE_:+.2f}  -> {np.sqrt(231.7292**2 - G_/344841):.4f}  rows R {int((nR_ != v8r).sum()):,}')
ok = {k: v for k, v in res.items() if v[0] / v[1] > 1}; pick = max(ok, key=lambda k: ok[k][0]) if ok else None
print(f'RULE (highest G_hat with G/SE > 1) -> {pick}')
if os.environ.get('PIN_PICKS') == '1':   # final phase: keep the variant the clean chain shipped; the rule above compares on the scored rows
    pick = os.environ.get('PIN_V110', 'C alpha'); assert pick in fam, pick; print(f'PINNED -> {pick}')
if os.environ.get('SAVE') == 'Cfam' and pick:
    np.savez(f'{I.CAND}/v110_adsb.npz', ids=sub_ids, v108=v8r, new=fam[pick][1], win_ids=ctx['ids'], newW=fam[pick][0], w_in=w_c, members=np.array(names), variant=pick)
    print(f'saved cand/v110_adsb.npz as variant "{pick}"')
if os.environ.get('SAVE') == 'C':
    np.savez(f'{I.CAND}/v110c_adsb.npz', ids=sub_ids, v108=v8r, new=nRc, win_ids=ctx['ids'], newW=nWc, w_in=w_c, members=np.array(names), variant='C covered unfired rows: v100 + new bulk (weights fit on covered rows), no ADS-B stage')
    print('saved cand/v110c_adsb.npz')
if os.environ.get('SAVE') == '1':
    np.savez(f'{I.CAND}/v110b_adsb.npz', ids=sub_ids, v108=v8r, new=nR, win_ids=ctx['ids'], newW=nW, w_in=w_u, members=np.array(names), variant='U2 rows WITHOUT ADS-B, weights refit there')
    print('saved cand/v110b_adsb.npz')
# ---- the pick, per airport: does the scored side move more because more rows are covered (gain scales too) or because
# each covered row moves further (drift)? rows changed and RMS change per airport, window vs scored, and own gain
if pick:
    nW_, nR_ = fam[pick]; dW_, dR_ = nW_ - v8w, nR_ - v8r; own = (v8w - y) ** 2 - (nW_ - y) ** 2; lin = -2.0 * (v8w - y) * dW_
    print(f'\n{pick} per airport:  rows changed W (share) | R (share)   RMS change W | R   own W   coverage-scaled G')
    tot = 0.0
    for a in sorted(pd.unique(aptw)):
        mw, mr = aptw == a, Rapt == a; cw, cr = mw & (dW_ != 0), mr & (dR_ != 0)
        g = (float(lin[cw].sum()) * sc / max(cw.sum(), 1) * cr.sum() - float((dR_[mr] ** 2).sum())) if cw.any() else -float((dR_[mr] ** 2).sum()); tot += g
        rw = np.sqrt(np.mean(dW_[cw] ** 2)) if cw.any() else 0; rr = np.sqrt(np.mean(dR_[cr] ** 2)) if cr.any() else 0
        print(f'  {a}  {cw.sum():7,d} ({cw.sum()/mw.sum():5.1%}) | {cr.sum():7,d} ({cr.sum()/mr.sum():5.1%})   {rw:6.1f} | {rr:6.1f}   {own[mw].sum()*sc:+.2e}   {g:+.2e}')
    print(f'  coverage-scaled total {tot:+.4e} -> {np.sqrt(231.7292**2 - tot/344841):.4f}   (Q-anchored G_hat {res[pick][0]:+.4e})')
