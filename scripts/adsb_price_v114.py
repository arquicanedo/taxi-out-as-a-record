"""s558: v112 blends v108 and C2 with ONE alpha (0.662 in-sample; the cross-fits disagree, 0.834 Jan-fit vs 0.589 Jul-fit) and
LIRF loses under it (s554). Two variants fixed before the run, priced against v112 (226.9779):
  per-airport alpha   a_apt fit on each airport's covered unfired rows, Jan|Jul cross-fitted for the window vector, in-sample
                      for the scored side; an airport with under 200 fitting rows in a fold keeps the global alpha
  no LIRF             v112's global alpha, with LIRF's rows left at v108
Rule: highest G_hat with G/SE > 1. Executes scripts/adsb_stage_full.py up to its pricing (v112 asserted rebuilt).
SAVE=1 writes cand/v114_adsb.npz.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_price_v114.py
"""
import os, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_stage_full.py')).read().split("ok = {k: v for k, v in res2.items()")[0])
v112w, v112r = fam2['C2 alpha']
assert np.array_equal(v112r, I.sub('v112')), 'v112 rebuilt'
dW_, dR_ = C2w - v8w, C2r - v8r; e = v8w - y
a_glob = float(-(e * dW_).sum() / (dW_ ** 2).sum()); print(f'v112 rebuilt; global alpha {a_glob:.3f}')
afit = lambda m: float(-(e * dW_)[m].sum() / (dW_[m] ** 2).sum())
aW, aR = np.zeros(nw), np.zeros(len(sub_ids)); rows = []
for a in sorted(pd.unique(aptw)):
    mw, mr = aptw == a, Rapt == a; ch = mw & (dW_ != 0)
    ai = afit(ch) if ch.sum() >= 200 else a_glob; aR[mr] = ai
    for fit, app in ((L, J), (J, L)):
        f_ = ch & fit; aW[mw & app] = afit(f_) if f_.sum() >= 200 else a_glob
    rows.append(f'{a} {ai:.3f} ({aW[mw & L][0] if (mw & L).any() else 0:.3f}|{aW[mw & J][0] if (mw & J).any() else 0:.3f})')
print('per-airport alpha, in-sample (Jan-fit applied to Jul | Jul-fit applied to Jan): ' + '  '.join(rows))
fam4 = {'per-airport alpha': (cl(np.rint(v8w + aW * dW_)), cl(np.rint(v8r + aR * dR_)))}
nW_, nR_ = v112w.copy(), v112r.copy(); nW_[LW] = v8w[LW]; nR_[LR] = v8r[LR]; fam4['no LIRF'] = (nW_, nR_)
res4 = {}
for tag, (xW, xR) in fam4.items():
    lin = -2.0 * (v112w - y) * (xW - v112w); own = (v112w - y) ** 2 - (xW - y) ** 2; Qb = float(((xR - v112r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
    cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own)
    SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc; res4[tag] = (G, SE)
    print(f'  vs v112  {tag:17s} own {own.sum()*sc:+.4e} ({own.sum()*sc/SEo:+.2f} SE)  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  G/SE {G/SE:+.2f}'
          f'  -> {np.sqrt(226.9779**2 - G/344841):.4f}  rows R {int((xR != v112r).sum()):,}')
    print('      per airport window own: ' + '  '.join(f'{a} {own[aptw == a].sum()*sc:+.1e}' for a in sorted(pd.unique(aptw))))
ok = {k: v for k, v in res4.items() if v[0] / v[1] > 1}; p4 = max(ok, key=lambda k: ok[k][0]) if ok else None
print(f'RULE (highest G_hat with G/SE > 1) -> {p4}')
if os.environ.get('PIN_PICKS') == '1':   # final phase: keep the variant the clean chain shipped
    p4 = os.environ.get('PIN_V114', 'no LIRF'); assert p4 in fam4, p4; print(f'PINNED -> {p4}')
if os.environ.get('SAVE') == '1' and p4:
    np.savez(f'{I.CAND}/v114_adsb.npz', ids=sub_ids, v112=v112r, new=fam4[p4][1], win_ids=ctx['ids'], newW=fam4[p4][0], variant=p4)
    print(f'saved cand/v114_adsb.npz as "{p4}"')
