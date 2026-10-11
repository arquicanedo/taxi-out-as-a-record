"""s548: manifest of every parsed adsb.lol day: which replica it was streamed from (the tag recorded in its .done marker;
days from the s516 pull predate that field and are prod-0) and what it yielded. -> artifacts/adsb_pull/day_manifest.tsv
Re-run after any pull; with the archived PREFERRED_RELEASES_2025.txt it pins the exact inputs of the full-year table.
  python3 scripts/adsb_day_manifest.py
"""
import glob, os, re
DY = os.path.join(os.environ['DC26_WORK'], 'adsblol/days'); OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts', 'adsb_pull', 'day_manifest.tsv')
rows = []
for f in sorted(glob.glob(f'{DY}/d*.done')):
    s = open(f).read().strip(); day = os.path.basename(f)[1:9]
    tag = s.split(' ')[0] if s.startswith('v20') or s.startswith('prod') or s.startswith('staging') else 'prod-0 (s516 pull, tag not recorded)'
    m = re.search(r'(\d+) traces, (\d+) ground, (\d+) takeoffs, (\d+) landings', s)
    rows.append('\t'.join([day, tag] + (list(m.groups()) if m else ['', '', '', ''])))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, 'w').write('day\treplica\ttraces\tground\ttakeoffs\tlandings\n' + '\n'.join(rows) + '\n')
print(f'{len(rows)} days -> {OUT}')
