"""Pull the OPDI flight_list months the shipped model needs. Reproducibility dependency.

WHY. `quick-boat_v74.parquet` and `quick-boat_v76.parquet` both read
`$DC26_WORK/v55_trees.parquet`, which is the pair of tree components retrained with the OPDI
airframe features (s223-s225). So OPDI is a HARD dependency of both shipped models, and prize
eligibility requires that every external dataset be openly accessible and documented. This
script is the "accessible" half; `EXTERNAL-DATA.md` is the "documented" half and states the
licence position exactly rather than favourably.

WHAT IS NEEDED. Only `flight_list`, and only these seventeen months:

    2025-01 .. 2025-12    the learning year
    2026-01, 2026-02      the January ranking month, plus 02 for rotation lookahead
    2026-07               the July ranking month
    2026-03, 2026-06      the final phase (2026-10-08): June, and 03 as February's lookahead
                          (02 and 07 are already above)

`flight_events` and `measurements` are NOT needed. Four `flight_events` files (801 MB) sit in
`data/opdi/` from the s221 turnaround-veto work, which died at -68.9 board-s and is not in any
shipped path. Do not download them to reproduce the model.

SIZE. Seventeen `flight_list` files (fifteen were ~591 MB), plain HTTPS, no authentication.

WHAT IT IS USED FOR, in one line, because a reader should not have to infer it: `flight_list`
supplies `icao24`, the airframe identity that the challenge data does not carry, joined on
`(flt_id, dof)` against `CALLSIGN_flt`. That gives per-airframe turnaround features. The join
reaches 84.7% on `CALLSIGN_flt` and 90.6% of 2025 departures get a registration (s221).

  uv run --quiet --with pandas --with pyarrow python scripts/pull_opdi.py
  DRY=1 ... to list what would be fetched without fetching
"""
import os, sys, urllib.request, urllib.error, hashlib

BASE = 'https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list'
OUT = os.environ.get('OPDI_DIR',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'opdi'))
MONTHS = [f'2025{m:02d}' for m in range(1, 13)] + ['202601', '202602', '202603', '202606', '202607']
DRY = os.environ.get('DRY') == '1'

os.makedirs(OUT, exist_ok=True)
print(f'OPDI flight_list -> {os.path.abspath(OUT)}')
print(f'{len(MONTHS)} months: {" ".join(MONTHS)}\n')

ok = skip = fail = 0
for m in MONTHS:
    name = f'flight_list_{m}.parquet'
    dest = os.path.join(OUT, name)
    if os.path.exists(dest):
        print(f'  have  {name}  {os.path.getsize(dest):>12,} B')
        skip += 1
        continue
    url = f'{BASE}/{name}'
    if DRY:
        print(f'  would fetch  {url}')
        continue
    try:
        tmp = dest + '.part'
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, 'wb') as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(tmp, dest)
        print(f'  got   {name}  {os.path.getsize(dest):>12,} B')
        ok += 1
    except urllib.error.HTTPError as e:
        print(f'  FAIL  {name}  HTTP {e.code}')
        fail += 1
    except Exception as e:
        print(f'  FAIL  {name}  {e}')
        fail += 1

print(f'\nfetched {ok}, already present {skip}, failed {fail}')
if fail:
    sys.exit('some months are missing -- the airframe features cannot be rebuilt without them')
print('\nlicence: see EXTERNAL-DATA.md. The OPDI portal states BOTH free reuse with attribution')
print('and non-commercial focus, with no LICENSE file. Admissibility for this challenge was')
print('granted explicitly by an organiser on 2026-09-06; the licence wording question was asked')
print('the same day and is unanswered. Both facts are recorded rather than resolved in our favour.')
