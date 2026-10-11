"""Pull METAR from Iowa State IEM ASOS for the 10 challenge airports."""
import io, sys, time, urllib.parse, urllib.request
import pandas as pd

APTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
BASE = 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?'
P = dict(data='tmpf,dwpf,drct,sknt,p01i,vsby,skyc1,skyl1,wxcodes',
         year1=2024, month1=12, day1=1, year2=2026, month2=8, day2=15,   # covers Jan+Jul 2025 and 2026
         tz='UTC', format='onlycomma', latlon='no', missing='M', trace='T',
         direct='no', report_type=3)
out = []
for a in APTS:
    url = BASE + urllib.parse.urlencode({**P, 'station': a})
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=180).read().decode()
            df = pd.read_csv(io.StringIO(raw))
            print(f'  {a}: {len(df):,} obs   cols={list(df.columns)[:4]}...')
            out.append(df); break
        except Exception as e:
            print(f'  {a}: attempt {attempt+1} failed: {type(e).__name__}'); time.sleep(5)
    else:
        print(f'  {a}: GAVE UP')
m = pd.concat(out, ignore_index=True)
for c in ['tmpf','dwpf','drct','sknt','vsby','skyl1']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['p01i'] = pd.to_numeric(m['p01i'].replace('T', 0.001), errors='coerce')
for c in ['wxcodes','skyc1','station']:
    m[c] = m[c].astype(str)
m['valid'] = pd.to_datetime(m['valid'], utc=True)
import os; OUT = os.environ.get('METAR_OUT', os.path.join(os.environ['DC26_WORK'], 'metar.parquet')); m.to_parquet(OUT); print(f'wrote {OUT}: {len(m):,} obs')
print(m.head(3).to_string())
