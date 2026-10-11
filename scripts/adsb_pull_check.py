"""s548: completeness check of the parsed adsb.lol days: every day of 2025 and of Jan+Jul 2026 must have a .done marker
and all four output files, non-empty, and no leftover .tmp file; days far below their month's median in take-offs are
listed. Run during and after a pull.   python3 scripts/adsb_pull_check.py [--final]
"""
import os, sys, glob, datetime as dt, statistics as st
DY = os.path.join(os.environ['DC26_WORK'], 'adsblol/days'); final = '--final' in sys.argv
want = []; d = dt.date(2025, 1, 1)
while d <= dt.date(2026, 7, 31):
    if d.year == 2025 or d.month in (1, 7): want.append(d.strftime('%Y%m%d'))
    d += dt.timedelta(days=1)
missing, broken, done = [], [], {}
for k in want:
    if not os.path.exists(f'{DY}/d{k}.done'): missing.append(k); continue
    parts = [f'{DY}/d{k}_{p}.parquet' for p in ('ground', 'takeoffs', 'landings', 'cat')]
    if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in parts): broken.append(k); continue
    s = open(f'{DY}/d{k}.done').read(); done[k] = int(s.split(' takeoffs')[0].split(', ')[-1]) if ' takeoffs' in s else None
tmp = glob.glob(f'{DY}/*.tmp')
print(f'{len(want)} days expected; done {len(done)}; missing {len(missing)}; broken {len(broken)}; leftover .tmp {len(tmp)}')
if broken: print('  BROKEN:', broken)
if tmp: print('  TMP:', [os.path.basename(t) for t in tmp])
if final and missing: print('  MISSING:', missing)
bym = {}
for k, n in done.items():
    if n is not None: bym.setdefault(k[:6], []).append(n)
thin = [(k, n) for k, n in done.items() if n is not None and n < 0.8 * st.median(bym[k[:6]])]
print(f'  days under 80% of their month median in take-offs: {len(thin)}' + (': ' + ', '.join(f'{k} ({n})' for k, n in sorted(thin)) if thin else ''))
sys.exit(1 if final and (missing or broken or tmp) else 0)
