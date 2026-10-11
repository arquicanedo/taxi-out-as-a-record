"""Two window measurements for the paper (v2, sections 5.2 and 6.4) that had no committed source script:

  1. ordinary-row window RMSE of the final model (v121's window vector, scripts/paper_final_window.py; v114's under
     CLEAN=1 or FINAL=v114) by airport, with and without a
     detected pushback (the table in section 6.4)
  2. the rows the trajectory stage leaves alone: a pushback present and the base model at or above 3,600 s; their
     count, the ordinary share by count and by squared distance between the base and its bulk, and what replacing the
     base with the bulk there would cost on the window (section 5.2, from s518)

Reuses adsb_price_v110.py's setup unchanged (window, base model, coverage mask). Default mode reproduces the shipped
chain's figures; CLEAN=1 with DC26_CAND/DC26_SUBDIR/SHIFT_1/SHIFT_7 (scripts/regen_clean.sh) gives the clean model's.
  uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python scripts/paper_pushback_split.py
"""
import contextlib, io, os
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(os.path.join(HERE, 'adsb_price_v110.py')).read().split('def price(')[0])
w0 = np.array([.2230, .2030, .0880, .4860]); B0 = np.stack([MW[k] for k in ('L55', 'P55', 'K32', 'K128')], 1) @ w0 + SHW
inc0 = I.mix(ctx, B0); appw = np.abs(inc0 - vsd) > 60   # the base model on the window, as scripts/adsb_price.py builds it
v100w = inc0.copy(); v100w[appw] = cl(np.rint(inc0[appw] + (B0w + SHW - B0)[appw]))
FINAL = 'v114' if I.CLEAN else os.environ.get('FINAL', 'v126')
z = np.load(f'{I.CAND}/{FINAL}_window.npz' if FINAL in ('v121', 'v124', 'v125', 'v126') else f'{I.CAND}/{FINAL}_adsb.npz'); assert (z['win_ids'] == ctx['ids']).all()
v114w = z['newW'].astype(float)   # the final model's window vector (name kept from the first version of this script)
aw = ctx['apt'].astype(str)
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
print(f'window rows {len(y):,}; with a detected pushback {hasw.mean():.1%}; base RMSE {rm(v100w, np.ones(len(y), bool)):.3f}, final {rm(v114w, np.ones(len(y), bool)):.3f}')
print(f'\n1. ordinary-row window RMSE of the final model\n{"":18s}' + ''.join(f'{a:>8s}' for a in sorted(set(aw))))
for nm, h in (('with a pushback', hasw), ('without', ~hasw)):
    print(f'{nm:18s}' + ''.join(f'{rm(v114w, O & h & (aw == a)) if (O & h & (aw == a)).sum() >= 200 else float("nan"):8.1f}' for a in sorted(set(aw))))
print('   rows: ' + '  '.join(f'{a} {int((O & hasw & (aw == a)).sum())}|{int((O & ~hasw & (aw == a)).sum())}' for a in sorted(set(aw))))
bulk = cl(np.rint(B0w + SHW)); m = hasw & (v100w >= 3600); D2 = (v100w - bulk) ** 2
print(f'\n2. pushback present and base >= 3,600 s: {int(m.sum())} window rows, ordinary {float((m & O).sum() / m.sum()):.1%} by count, '
      f'{float(D2[m & O].sum() / D2[m].sum()):.1%} by squared distance base-bulk; replacing the base with the bulk there: '
      f'window SSE {float(((bulk - y) ** 2 - (v100w - y) ** 2)[m].sum()):+.4e}')
