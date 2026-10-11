"""AUTOMATED SLICE SEARCH AT SCALE, with a permutation null so it cannot manufacture signal.

column_sweep.py exhausted single columns on the cell: best |AUC-0.5| is 0.094 (ADES in ten), and
FLIGHT_ID_mvt is 100% null there while FLIGHT_RULE_mvt has one value. So a mechanism, if it exists,
is in COMBINATIONS.

THE METHOD, and the reason it is not just more p-hacking:
  1. build every categorical slice and every numeric bin from the *_mvt columns
  2. evaluate each slice OUT OF FOLD by month -- the rate comes from the other eleven months
  3. score by D^2-WEIGHTED lift, because squared error is what pays, not AUC (s237)
  4. run the ENTIRE search again on PERMUTED labels, 200 times, and record the best slice found
     each time. That distribution is what a null search finds by luck on 627 rows.
  5. report only slices beating the 99th percentile of the null

Step 4 is the whole point. Without it a search this wide always returns something.
"""
import os, glob, re, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
D=os.environ['DC26_DATA']; THR=7200.; TYP=1021.
sch=pq.read_schema(f'{D}/ranking.parquet'); MVT=[c for c in sch.names if c.endswith('_mvt')]
L=pd.concat([pq.read_table(f,columns=MVT+['LOBT_flt']).to_pandas()
             for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True)
for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']: L[c]=pd.to_datetime(L[c],errors='coerce',utc=True)
d=L[L.PHASE_mvt.eq('DEP')].copy(); d['y']=d.TAXITIME_SEC_mvt.astype(float)
d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
c=d[d.ADEP_mvt.eq('LIRF')&d.LOBT_flt.isna()&(d.ms>=THR)].copy()
c['cp']=(np.abs(c.y-c.ms)<=60).astype(int); c['mon']=c.MVT_TIME_UTC_mvt.dt.month
c['D2']=(c.ms-TYP)**2
TEN=set(d.ADEP_mvt.unique())
# ---- build the feature space ----
CAT={}
CAT['carrier']=c.FLIGHT_mvt.astype(str).str[:3]
CAT['ades']=c.ADES_mvt.astype(str)
CAT['ades_region']=c.ADES_mvt.astype(str).str[:2]
CAT['ades_in_ten']=c.ADES_mvt.isin(TEN).map({True:'TEN',False:'OUT'})
CAT['runway']=c.RUNWAY_mvt.astype(str)
CAT['stand_pre']=c.STAND_mvt.astype(str).str[:1]
CAT['garbled']=c.FLIGHT_mvt.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d').map({True:'G',False:'C'})
CAT['flt_tail']=c.FLIGHT_mvt.astype(str).str.match(r'.*[A-Z]$').map({True:'L',False:'D'})
CAT['flt_len']=c.FLIGHT_mvt.astype(str).str.len().astype(str)
NUMB={}
NUMB['ms']=pd.cut(c.ms,[7200,10000,14400,25000,50000,1e9],labels=['a','b','c','d','e'])
NUMB['hour']=pd.cut(c.MVT_TIME_UTC_mvt.dt.hour,[-1,5,11,17,23],labels=['n','m','a','e'])
NUMB['dow']=pd.cut(c.MVT_TIME_UTC_mvt.dt.dayofweek,[-1,0,3,5,6],labels=['mo','mid','fr','we'])
NUMB['schedmin']=pd.cut(c.SCHED_TIME_UTC_mvt.dt.minute,[-1,5,20,40,55,60],labels=['q0','q1','q2','q3','q4'])
NUMB['stand_n']=pd.cut(pd.to_numeric(c.STAND_mvt.astype(str).str.extract(r'(\d+)')[0],errors='coerce'),
                       [0,300,500,700,900,1e4],labels=['s1','s2','s3','s4','s5'])
ALL={**{k:v.astype(str) for k,v in CAT.items()},**{k:v.astype(str) for k,v in NUMB.items()}}
keys=list(ALL)
print(f'cell {len(c)} rows, copy {c.cp.mean():.1%}   {len(keys)} categoricals')
MONTHS=sorted(c.mon.unique())
def oof_lift(lab,cp):
    """for one slice-labelling, the best single level's OOF D^2-weighted copy rate minus base"""
    best=(0.,None,0)
    base=float((c.D2*cp).sum()/c.D2.sum())
    for lv in pd.unique(lab):
        sel=(lab==lv).values
        if sel.sum()<25: continue
        num=0.;den=0.
        for m in MONTHS:
            te=(c.mon.values==m)&sel; tr=(c.mon.values!=m)&sel
            if te.sum()==0 or tr.sum()<10: continue
            r=float((c.D2[tr]*cp[tr]).sum()/max(c.D2[tr].sum(),1e-9))
            num+=r*float(c.D2[te].sum()); den+=float(c.D2[te].sum())
        if den<=0: continue
        lift=abs(num/den-base)
        if lift>best[0]: best=(lift,lv,int(sel.sum()))
    return best
def search(cp):
    out=[]
    for i,k in enumerate(keys):
        l,lv,n=oof_lift(ALL[k],cp)
        if lv is not None: out.append((l,k,str(lv),n))
        for k2 in keys[i+1:]:
            comb=ALL[k]+'|'+ALL[k2]
            l,lv,n=oof_lift(comb,cp)
            if lv is not None: out.append((l,f'{k} x {k2}',str(lv),n))
    return sorted(out,reverse=True)
real=search(c.cp.values)
print(f'  slices evaluated: {len(real):,}')
rng=np.random.default_rng(0); null=[]
for b in range(200):
    perm=rng.permutation(c.cp.values)
    r=search(perm)
    null.append(r[0][0] if r else 0.)
null=np.array(null)
thr=np.percentile(null,99)
print(f'\n=== PERMUTATION NULL over 200 label shuffles ===')
print(f'  best lift a NULL search finds: median {np.median(null):.4f}  p95 {np.percentile(null,95):.4f}  '
      f'p99 {thr:.4f}  max {null.max():.4f}')
print(f'\n=== the real search, top 15, against that threshold ===')
print(f'  {"lift":>7} {"n":>5}  slice')
for l,k,lv,n in real[:15]:
    mark=' *** BEATS NULL p99' if l>thr else ''
    print(f'  {l:>7.4f} {n:>5}  {k} = {lv}{mark}')
beat=[x for x in real if x[0]>thr]
print(f'\n  slices beating the null p99: {len(beat)} of {len(real):,}')
print(f'  expected by chance at p99: {0.01*len(real):.0f}')
