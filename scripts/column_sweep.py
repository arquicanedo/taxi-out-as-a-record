"""SYSTEMATIC SWEEP: every available column on the cell, measured for copy discrimination.

On cell rows all 16 *_flt columns are null, so only the 14 *_mvt columns exist. Tested so far:
MVT_ID, FLIGHT, ADEP, ADES, RUNWAY, STAND, MVT_TIME, SCHED_TIME. NEVER TESTED: FLIGHT_ID_mvt,
FLIGHT_RULE_mvt. This sweeps all of them and their simple derivations, ranked by AUC.
"""
import os, glob, re, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
D=os.environ['DC26_DATA']; THR=7200.
sch=pq.read_schema(f'{D}/ranking.parquet')
MVT=[c for c in sch.names if c.endswith('_mvt')]
print(f'*_mvt columns: {MVT}')
L=pd.concat([pq.read_table(f,columns=MVT+['LOBT_flt']).to_pandas()
             for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True)
for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','LOBT_flt']:
    L[c]=pd.to_datetime(L[c],errors='coerce',utc=True)
d=L[L.PHASE_mvt.eq('DEP')].copy()
d['y']=d.TAXITIME_SEC_mvt.astype(float)
d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
cell=d[d.ADEP_mvt.eq('LIRF')&d.LOBT_flt.isna()&(d.ms>=THR)].copy()
cell['cp']=(np.abs(cell.y-cell.ms)<=60).astype(int)
print(f'\ncell {len(cell)} rows, copy {cell.cp.mean():.1%}')
print(f'\n=== THE TWO NEVER EXAMINED ===')
for c in ('FLIGHT_ID_mvt','FLIGHT_RULE_mvt'):
    if c not in cell: print(f'  {c}: ABSENT'); continue
    v=cell[c]
    print(f'  {c}: dtype {v.dtype}, null {v.isna().mean():.1%}, distinct {v.nunique()}')
    print(f'    sample: {list(v.dropna().unique()[:6])}')
    if v.nunique()<=12:
        t=cell.groupby(c).cp.agg(['size','mean']).sort_values('size',ascending=False)
        print(t.to_string(formatters={'mean':'{:.1%}'.format}))
# build every candidate feature
F={}
cell['fid']=pd.to_numeric(cell.FLIGHT_ID_mvt,errors='coerce')
cell['mid']=pd.to_numeric(cell.MVT_ID_mvt,errors='coerce')
if cell.fid.notna().any():
    F['FLIGHT_ID raw']=cell.fid
    F['FLIGHT_ID - MVT_ID']=cell.fid-cell.mid
    F['FLIGHT_ID mod 1e3']=cell.fid%1000
    F['FLIGHT_ID mod 1e6']=cell.fid%1000000
    allr=d[d.ADEP_mvt.eq('LIRF')].copy()
    allr['fid']=pd.to_numeric(allr.FLIGHT_ID_mvt,errors='coerce')
    allr['day']=allr.MVT_TIME_UTC_mvt.dt.floor('D')
    pct=allr.groupby('day').fid.rank(pct=True)
    allr['fidpct']=pct
    cell=cell.merge(allr[['MVT_ID_mvt','fidpct']],on='MVT_ID_mvt',how='left')
    F['FLIGHT_ID pct within day']=cell.fidpct
    # is FLIGHT_ID shared with any other movement? (a flight has a DEP and an ARR record)
    n=d.groupby('FLIGHT_ID_mvt').size()
    cell['fid_n']=cell.FLIGHT_ID_mvt.map(n)
    F['FLIGHT_ID row count']=cell.fid_n
F['ms']=cell.ms
F['hour']=cell.MVT_TIME_UTC_mvt.dt.hour
F['dow']=cell.MVT_TIME_UTC_mvt.dt.dayofweek
F['month']=cell.MVT_TIME_UTC_mvt.dt.month
F['sched second']=cell.SCHED_TIME_UTC_mvt.dt.second
F['mvt second']=cell.MVT_TIME_UTC_mvt.dt.second
F['sched minute']=cell.SCHED_TIME_UTC_mvt.dt.minute
F['garbled flight']=cell.FLIGHT_mvt.map(lambda s: bool(re.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d',str(s).strip().upper()))).astype(int)
F['flight len']=cell.FLIGHT_mvt.astype(str).str.len()
F['flight has letter tail']=cell.FLIGHT_mvt.astype(str).str.match(r'.*[A-Z]$').astype(int)
F['stand numeric']=pd.to_numeric(cell.STAND_mvt.astype(str).str.extract(r'(\d+)')[0],errors='coerce')
F['stand prefix']=pd.to_numeric(cell.STAND_mvt.astype(str).str[:1],errors='coerce')
F['stand is letter']=cell.STAND_mvt.astype(str).str[:1].str.isalpha().astype(int)
F['runway 25']=(cell.RUNWAY_mvt.astype(str)=='25').astype(int)
F['ADES in ten']=cell.ADES_mvt.isin(set(d.ADEP_mvt.unique())).astype(int)
F['MVT_ID pct within day']=None
print(f'\n=== every feature, ranked by |AUC - 0.5| ===')
res=[]
for k,v in F.items():
    if v is None: continue
    v=pd.to_numeric(v,errors='coerce')
    m=v.notna()
    if m.sum()<100 or v[m].nunique()<2: continue
    a=roc_auc_score(cell.cp[m.values],v[m])
    res.append((abs(a-0.5),a,k,int(m.sum())))
for s,a,k,n in sorted(res,reverse=True):
    print(f'  {k:28} AUC {a:.4f}  |lift| {s:.4f}  n={n}')
