"""s562: v117 = v114 with the blend weight refitted on the rows v114 actually blends. v112 fitted one alpha (0.662) for the
move from v108 toward the full-year model C2 on ALL covered rows, LIRF included; v114 then left LIRF's rows at v108. LIRF's own
weight is ~0.37 (s558), so it dragged the global alpha down everywhere else. Refitted outside LIRF the weight is stable across
the cross-fit (Jan-fit ~0.83, Jul-fit ~0.84), against 0.834 / 0.589 with LIRF in. Fixed before pricing:
  alpha refit outside LIRF   alpha fitted on covered changed non-LIRF rows; Jan|Jul cross-fitted for the window vector,
                             in-sample for the scored side; LIRF stays at v108 as in v114
Rule: submit-worthy if G/SE > 1 and the window's own gain is positive. Executes scripts/adsb_price_v114.py up to its rule
(v114 asserted rebuilt). SAVE=1 writes cand/v117_adsb.npz.
  uv run --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_price_v117.py
"""
import os, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_price_v114.py')).read().split("ok = {k: v for k, v in res4.items()")[0])
v114w, v114r = fam4['no LIRF']; assert np.array_equal(v114r, I.sub('v114')), 'v114 rebuilt'
print(f'v114 rebuilt; window RMSE {np.sqrt(np.mean((v114w - y) ** 2)):.3f}; v112 alpha (LIRF in) {a_glob:.3f}')
chN = (dW_ != 0) & ~LW
a_in, a_j, a_l = afit(chN), afit(chN & J), afit(chN & L)
print(f'alpha fitted outside LIRF: in-sample {a_in:.3f}; Jan-fit {a_j:.3f} -> Jul, Jul-fit {a_l:.3f} -> Jan')
xW = cl(np.rint(v8w + np.where(J, a_l, a_j) * dW_)); xR = cl(np.rint(v8r + a_in * dR_)); xW[LW] = v8w[LW]; xR[LR] = v8r[LR]
lin = -2.0 * (v114w - y) * (xW - v114w); own = (v114w - y) ** 2 - (xW - y) ** 2; Qb = float(((xR - v114r) ** 2).sum()); m2T = float(lin.sum()) * sc; G = m2T - Qb
cs, co = np.bincount(clus, weights=lin), np.bincount(clus, weights=own)
SE = float(np.array([cs[i].sum() for i in ii]).std()) * sc; SEo = float(np.array([co[i].sum() for i in ii]).std()) * sc
print(f'  vs v114  alpha refit outside LIRF  own {own.sum()*sc:+.4e} ({own.sum()*sc/SEo:+.2f} SE)  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  G/SE {G/SE:+.2f}'
      f'  rows R {int((xR != v114r).sum()):,}')
print('      per airport window own: ' + '  '.join(f'{a} {own[aptw == a].sum()*sc:+.1e}' for a in sorted(pd.unique(aptw))))
print(f'      window RMSE {np.sqrt(np.mean((xW - y) ** 2)):.3f} (v114 {np.sqrt(np.mean((v114w - y) ** 2)):.3f}); ordinary {np.sqrt(np.mean((xW - y)[O] ** 2)):.3f} ({np.sqrt(np.mean((v114w - y)[O] ** 2)):.3f})')
for r in (1.0, 1.5, 1.97): print(f'      if the linear term transfers at {r:.2f}x: {226.8129 - np.sqrt(226.8129**2 - (r*m2T - Qb)/344841):+.4f} s -> {np.sqrt(226.8129**2 - (r*m2T - Qb)/344841):.4f}')
go = G / SE > 1 and own.sum() > 0; print(f'RULE (G/SE > 1 and window own gain > 0) -> {"alpha refit outside LIRF" if go else None}')
if os.environ.get('PIN_PICKS') == '1':   # final phase: the clean chain did not build v117 (G/SE +0.97), so v117 = v114 unless PIN_V117=1
    go = os.environ.get('PIN_V117', '0') == '1'; print(f'PINNED -> {"alpha refit outside LIRF" if go else None}')
if os.environ.get('SAVE') == '1' and go:
    np.savez(f'{I.CAND}/v117_adsb.npz', ids=sub_ids, v114=v114r, new=xR, win_ids=ctx['ids'], newW=xW, variant='alpha refit outside LIRF',
             alpha=np.array([a_in, a_j, a_l]))
    print('saved cand/v117_adsb.npz as "alpha refit outside LIRF"')
