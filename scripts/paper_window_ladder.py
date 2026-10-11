"""Per-layer ablation of the trajectory stage on the labelled window, with day-clustered intervals, and the window RMSE
without its invisible day-scale rows (paper v2, sections 6.1 and 6.4). Every number the paper takes from here is logged
in analysis/paper-v2/added.tsv with this script as its source.

  base      v100 on the window, rebuilt exactly as scripts/adsb_price.py does
  vNNN      the cross-fitted window vector saved by each pricer (cand/vNNN_adsb.npz, 'newW')
  v121      cand/v121_window.npz (scripts/paper_final_window.py); v118 and v120 change no window row
  v122-126  cand/v126_window.npz (scripts/paper_final_window_v126.py): the join rule, the classifier, the near branch,
            the local-time class, the day-scale floor
            (shipped chain only; CLEAN=1 stops at v114)
  invisible a day-scale row (y >= 20,000 s) whose schedule deviation is ordinary (|ms| < 7,200 s)
Intervals: 95% day-cluster bootstrap over the window's 62 days, 2,000 draws, of the RMSE difference to the previous row.

  uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python scripts/paper_window_ladder.py
  CLEAN chain: DC26_CAND=$DC26_WORK/clean/cand SHIFT_1=-10.638 SHIFT_7=-0.548 CLEAN=1 DC26_SUBDIR=$DC26_WORK/clean/submissions ...
"""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'adsb_stage_exp.py')
exec(open(os.path.join(HERE, 'adsb_price.py')).read().split('# ---------------- 2. ADS-B features')[0])
W = os.environ['DC26_WORK']
def win(v):
    z = np.load(f'{I.CAND}/{v}_adsb.npz'); assert (z['win_ids'] == ctx['ids']).all(); return z['newW'].astype(float)
LADDER = [('base model (v100)', v100w.astype(float))] + [(v, win(v)) for v in ('v102', 'v103', 'v104', 'v107', 'v108', 'v110', 'v112', 'v114')]
if not I.CLEAN:
    zf = np.load(f'{I.CAND}/v121_window.npz'); assert (zf['win_ids'] == ctx['ids']).all()
    z6 = np.load(f'{I.CAND}/v126_window.npz'); assert (z6['win_ids'] == ctx['ids']).all()
    LADDER += [('v117', win('v117')), ('v121', zf['newW'].astype(float)), ('v122', z6['v122W'].astype(float)),
               ('v123', z6['v123W'].astype(float)), ('v124', z6['v124W'].astype(float)), ('v125', z6['v125W'].astype(float)),
               ('v126 (final)', z6['newW'].astype(float))]
SIDE = [] if I.CLEAN else [('v105', 'v104'), ('v106', 'v104'), ('v109', 'v108')]
days = ctx['cluster']; k = days.max() + 1; rng = np.random.default_rng(11); B = 2000; idx = [rng.integers(0, k, k) for _ in range(B)]
allm = np.ones(len(y), bool); ms = ctx['vsd']
inv = (y >= 20000) & (np.abs(ms) < 7200)
print(f'window rows {len(y):,}; invisible day-scale rows {int(inv.sum())}: y = {sorted(y[inv].astype(int).tolist())}')
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
def ci(p0, p1, m):
    e0 = np.bincount(days[m], weights=((p0 - y) ** 2)[m], minlength=k); e1 = np.bincount(days[m], weights=((p1 - y) ** 2)[m], minlength=k)
    n = np.bincount(days[m], minlength=k); d = [np.sqrt(e0[i].sum() / n[i].sum()) - np.sqrt(e1[i].sum() / n[i].sum()) for i in idx]
    return np.percentile(d, [2.5, 97.5])
print(f'\n{"":20s}{"all rows":>10s}{"gain [95%]":>26s}{"ordinary":>10s}{"gain [95%]":>26s}{"no invisible":>14s}')
prev = None; P = dict(LADDER)
for nm, p in LADDER:
    row = f'{nm:20s}{rm(p, allm):10.3f}'
    if prev is None:
        row += f'{"":26s}{rm(p, O):10.3f}{"":26s}'
    else:
        a = ci(prev, p, allm); o = ci(prev, p, O)
        row += f'{rm(prev, allm) - rm(p, allm):+10.3f} [{a[0]:+.3f}, {a[1]:+.3f}]{rm(p, O):10.3f}{rm(prev, O) - rm(p, O):+10.3f} [{o[0]:+.3f}, {o[1]:+.3f}]'
    row += f'{rm(p, ~inv):14.3f}'; print(row); prev = p
print('\nside branches, against their base')
for v, b in SIDE:
    p = win(v); q = P[b]; o = ci(q, p, O)
    print(f'{v} vs {b}: ordinary {rm(q, O):.3f} -> {rm(p, O):.3f}  gain {rm(q, O) - rm(p, O):+.3f} [{o[0]:+.3f}, {o[1]:+.3f}]')
