"""s555: does v100's mixture (the copy / day-scale hedge) still pay on rows ADS-B covers? On covered ordinary rows the pure
full-year bulk scores ~177 against v110's ~211 (s554): the hedge costs ~34 s there. It exists for the extremes. Measured on
the window, ALL rows including extremes, on v110's covered unfired region (fired copies untouched), per airport:
  SSE of v110   against   SSE of the pure bulk B2 (no mixture, no hedge)
split into ordinary (y < 7200) and extreme rows, so the hedge's cost and its benefit are both visible.
  uv run --with pandas --with pyarrow --with numpy python scripts/adsb_gate_oracle.py
"""
import os, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_v110_diag.py')).read().split('# ---- the pick, per airport')[0])
v110w, v110r = fam[pick]; assert pick == 'C alpha'
wL, wP = w_c[names.index('L55A')], w_c[names.index('P55A')]; wL, wP = wL / (wL + wP), wP / (wL + wP)
B2w = cl(np.rint(wL * MW['L55A'] + wP * MW['P55A'] + SHW))
e0, e1 = (v110w - y) ** 2, (B2w - y) ** 2
print(f'region: covered unfired (v110 changed there), window {awc.sum():,} rows; scale sc {sc:.4f}')
print(f'{"":6s} {"rows":>8s} {"extremes":>8s}   {"ordinary: v110 SSE - B2 SSE":>28s}   {"extremes: v110 SSE - B2 SSE":>28s}   {"net (positive = drop the hedge)":>32s}')
tot = {'ord': 0.0, 'ext': 0.0}
for a in list(sorted(pd.unique(aptw))) + ['ALL', 'ALL but LIRF']:
    ma = np.ones(nw, bool) if a == 'ALL' else (aptw != 'LIRF') if a == 'ALL but LIRF' else (aptw == a)
    m = ma & awc; mo, me = m & ~ext, m & ext
    if not m.any(): continue
    go, ge = float((e0 - e1)[mo].sum()) * sc, float((e0 - e1)[me].sum()) * sc
    print(f'  {a:12s} {m.sum():8,d} {me.sum():8,d}   {go:+28.4e}   {ge:+28.4e}   {go + ge:+32.4e}')
m = awc & (aptw != 'LIRF'); d = (e0 - e1); co = np.bincount(clus, weights=np.where(m, d, 0.0)); se = float(np.array([co[i].sum() for i in ii]).std()) * sc
print(f'ALL but LIRF net {float(d[m].sum())*sc:+.4e}, paired day-cluster SE {se:.3e} ({float(d[m].sum())*sc/se:+.2f} SE)')
big = np.where(m & ext)[0]; big = big[np.argsort(-(e1 - e0)[big])][:10]
print('extremes where dropping the hedge hurts most (apt, y, v110, B2):'); [print(f'  {aptw[i]}  {y[i]:8.0f}  {v110w[i]:8.0f}  {B2w[i]:8.0f}') for i in big]
