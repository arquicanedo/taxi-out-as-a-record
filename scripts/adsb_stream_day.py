"""Stream one adsb.lol globe_history day straight from GitHub into per-airport surface points, take-offs and
landings, without writing the tar or its extract to disk. Superset of p3q_adsb_parse.py: the same ground and
take-off schema (plus `trk`, and `t_lg`, the last ground sample: the transition may span up to 1800 s
where s512's parser required 120 s, so `t_to - t_lg <= 120` recovers it), and landings, so the labelled 2026 ARRIVALS can check whether 2025 coverage transfers.
  uv run --with pandas --with pyarrow python scripts/adsb_stream_day.py 2026.01.15 [workers]
Output: $DC26_WORK/adsblol/days/dYYYYMMDD_{ground,takeoffs,landings,cat}.parquet and dYYYYMMDD.done
Source: adsb.lol globe_history_YYYY, ODbL 1.0 (scripts/adsb/ATTRIBUTION.md).
"""
import os, sys, gzip, json, math, tarfile, subprocess, time
from multiprocessing import Pool
import pandas as pd
APT={'EHAM':(52.3105,4.7683),'EDDF':(50.0379,8.5622),'EGLL':(51.4700,-0.4543),'LIRF':(41.8003,12.2389),'LFPG':(49.0097,2.5479),
     'EDDM':(48.3538,11.7861),'LSZH':(47.4582,8.5555),'LEMD':(40.4719,-3.5626),'LEBL':(41.2974,2.0833),'LTFM':(41.2753,28.7519)}
R_KM=8.0
FLAVOR=os.environ.get('ADSB_FLAVOR','prod')   # s522: 'staging' / 'mlatonly' feeds probed for LFPG/LTFM coverage
OUT=os.path.join(os.environ['DC26_WORK'], 'adsblol/days' + ('' if FLAVOR=='prod' or os.environ.get('ADSB_OUT_PROD')=='1' else f'_{FLAVOR}'))   # s548: the year pull's fallback writes canonical days/
def near(lat,lon):
    if not (40.1<lat<52.7 and -3.8<lon<29.0): return None
    for a,(la,lo) in APT.items():
        if abs(lat-la)<0.12 and abs(lon-lo)<0.18:
            dx=(lon-lo)*111.32*math.cos(math.radians(la)); dy=(lat-la)*110.57
            if dx*dx+dy*dy<R_KM*R_KM: return a
    return None
def one(raw):
    try: js=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
    except Exception: return None
    hexid=js.get('icao'); ts0=js.get('timestamp',0); tr=js.get('trace') or []
    gpts=[]; tos=[]; lds=[]; cat=None; cs=None; last_g={}; last_a={}
    for p in tr:
        if len(p)<5: continue
        t=ts0+p[0]; lat,lon,alt,gs=p[1],p[2],p[3],p[4]
        trk=p[5] if len(p)>5 and p[5] is not None else float('nan')
        if len(p)>8 and isinstance(p[8],dict):
            cs=(p[8].get('flight') or cs); cat=p[8].get('category') or cat
        if lat is None or lon is None: continue
        a=near(lat,lon)
        if a is None: continue
        if alt=='ground':
            if a in last_a and 0<t-last_a[a]<=1800: lds.append((hexid,a,t,(cs or '').strip(),last_a[a])); del last_a[a]
            gpts.append((hexid,t,a,lat,lon,gs if gs is not None else float('nan'),trk)); last_g[a]=t
        elif isinstance(alt,(int,float)) and alt<3000:
            if a in last_g and 0<t-last_g[a]<=1800: tos.append((hexid,a,t,(cs or '').strip(),last_g[a])); del last_g[a]
            last_a[a]=t
    if not gpts and not tos and not lds: return None
    return gpts,tos,lds,(hexid,cat)
def urls(d):
    tag=f'v{d}-planes-readsb-{FLAVOR}-0'
    for yr in (d[:4], str(int(d[:4])+1)):   # s548: 2025-12-31 is published in globe_history_2026, not _2025
        r=subprocess.run(['gh','api',f'repos/adsblol/globe_history_{yr}/releases/tags/{tag}','--jq',
                          '.assets[] | [.browser_download_url, .size] | @tsv'],capture_output=True,text=True)
        if r.returncode==0 and r.stdout.strip(): break
    r.check_returncode()
    rows=sorted(l.split('\t') for l in r.stdout.strip().splitlines())
    return [u for u,_ in rows],sum(int(s) for _,s in rows)
def members(tf):
    for m in tf:
        if m.isfile() and 'trace_full_' in m.name: yield tf.extractfile(m).read()
if __name__=='__main__':
    d=sys.argv[1]; nw=int(sys.argv[2]) if len(sys.argv)>2 else 5; key='d'+d.replace('.','')
    os.makedirs(OUT,exist_ok=True); t0=time.time()
    if os.environ.get('ADSB_URLS'):   # s548: an explicit replica (PREFERRED_RELEASES.txt, or the fallback) instead of the flavour tag
        us,total=os.environ['ADSB_URLS'].split(','),0
    else:
        us,total=urls(d)
    cmd='; '.join(f'curl -sfL --retry 5 --retry-delay 10 --speed-limit 20000 --speed-time 300 "{u}" || exit 22' for u in us)
    proc=subprocess.Popen(['sh','-c',cmd],stdout=subprocess.PIPE,bufsize=1<<20)
    G=[];T=[];L=[];C=[]; n=0
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tf, Pool(nw) as pool:
        for r in pool.imap_unordered(one,members(tf),chunksize=100):
            n+=1
            if r: G+=r[0]; T+=r[1]; L+=r[2]; C.append(r[3])
    rc=proc.wait()
    if rc!=0 or n<20000: sys.exit(f'{d}: FAILED rc={rc} traces={n}')
    for nm,rows,cols in [('ground',G,['hex','t','apt','lat','lon','gs','trk']),('takeoffs',T,['hex','apt','t_to','callsign','t_lg']),
                         ('landings',L,['hex','apt','t_ld','callsign','t_la']),('cat',C,['hex','category'])]:
        pd.DataFrame(rows,columns=cols).to_parquet(f'{OUT}/{key}_{nm}.parquet.tmp',index=False)
        os.replace(f'{OUT}/{key}_{nm}.parquet.tmp',f'{OUT}/{key}_{nm}.parquet')
    open(f'{OUT}/{key}.done','w').write(f"{us[0].split('/download/')[1].split('/')[0] if '/download/' in us[0] else FLAVOR} " + f'{total} bytes, {n} traces, {len(G)} ground, {len(T)} takeoffs, {len(L)} landings, {time.time()-t0:.0f} s\n')
    print(f'{d}: {total/1e9:.2f} GB, {n:,} traces, {len(G):,} ground pts, {len(T):,} take-offs, {len(L):,} landings, {time.time()-t0:.0f} s',flush=True)
