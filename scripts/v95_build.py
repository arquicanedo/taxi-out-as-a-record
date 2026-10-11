"""v95 = v91 with the LIRF stand-4xx cell rows moved off the copy branch.

FOUND BY AUTOMATED SLICE SEARCH WITH A PERMUTATION NULL (scripts/slice_sweep.py). 101 slices over
every *_mvt categorical and binned numeric, scored on OUT-OF-FOLD D^2-weighted copy rate by month,
then the ENTIRE search re-run on 200 permuted label sets to measure what a null search finds by
luck. Null: median 0.4177, p99 0.5769, MAX 0.5818. The single real hit is 0.6472 -- above the null's
maximum, p < 0.005.

  LIRF cell copy rate by stand prefix
    2: 89.1%   3: 51.6%   5: 48.3%   6: 73.7%   7: 67.0%   8: 74.1%   9: 73.1%
    4:  0.0%   <- 28 rows, ZERO copies, in 9 separate months, no month with a single one

Stands 402-412, a contiguous block: one pier, one handling contract, whose system reports off-block
where the rest of LIRF's does not. Their y median is 780 s and 96.4% are under 2,000 -- ordinary
taxis. `STAND_mvt` is a movements column, present on the scored set (null on 0.2% of cell rows).

EVIDENCE: labelled-prior, a counted rate on the edited population. 0 of 28 gives a Rule-of-Three
95% upper bound of 10.7% on the true copy rate.

WHAT IS CLAIMED AND WHAT IS NOT. This does NOT commit the rows to a value; it removes copy mass the
evidence says is not there, moving each row to the ordinary-taxi level the stand's own labelled rows
show. Q is 0.348 board-s.

CONTROL: FIRE=0 reproduces v91 byte-identically.
"""
import hashlib, os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D=os.environ['DC26_DATA']
BASE='submissions/quick-boat_v91.parquet'; OUT=os.environ.get('V95_OUT','submissions/quick-boat_v95.parquet')
FIRE=os.environ.get('FIRE','1')=='1'
THR,FLOOR,CAP,PER=7200.,60.,200000.,1.842e8; SHA91='ff6c2dc932e2ec61'
C=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt',
   'LOBT_flt','STAND_mvt']
def prep(p,lab):
    r=pd.concat([pq.read_table(f,columns=C).to_pandas() for f in p],ignore_index=True)
    for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']: r[c]=pd.to_datetime(r[c],errors='coerce',utc=True)
    d=r[r.PHASE_mvt.eq('DEP')&r.ADEP_mvt.eq('LIRF')].copy()
    d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['sp']=d.STAND_mvt.astype(str).str[:1]
    return d
# ---- the labelled support, re-derived here so the script carries its own evidence ----
L=prep(sorted(glob.glob(f'{D}/training_*.parquet')),'2025'); L['y']=L.TAXITIME_SEC_mvt.astype(float)
S=L[L.LOBT_flt.isna()&(L.ms>=THR)&L.sp.eq('4')]
ncp=int((np.abs(S.y-S.ms)<=60).sum())
TGT=float(np.median(S.y))
print(f'labelled support: {len(S)} LIRF stand-4xx cell rows, {ncp} copies, '
      f'{S.MVT_TIME_UTC_mvt.dt.month.nunique()} distinct months')
assert ncp==0, f'support broken: {ncp} copies among the labelled stand-4xx rows'
assert len(S)>=25, 'support too small'
print(f'  their y: median {TGT:,.0f}  q10 {S.y.quantile(.1):,.0f}  q90 {S.y.quantile(.9):,.0f}')
# ---- apply ----
R=prep([f'{D}/ranking.parquet'],'2026')
assert R.MVT_TIME_UTC_mvt.dt.year.eq(2026).all()
Z=R[R.LOBT_flt.isna()&(R.ms>=THR)&R.sp.eq('4')].copy()
b=pq.read_table(BASE).to_pandas()
assert hashlib.sha256(open(BASE,'rb').read()).hexdigest()[:16]==SHA91, 'base is not v91'
tpl=pq.read_table(f'{D}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
assert bool((b.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
Z=Z.merge(b.rename(columns={'TAXITIME_SEC_mvt':'inc'}),on='MVT_ID_mvt',how='left')   # s390: MVT_ID join
assert Z.inc.notna().all(), 'incumbent join incomplete'
Z['inc']=Z.inc.astype(float)
Z['fire']=Z.inc>TGT+60        # only move rows we currently predict ABOVE the stand's own level
print(f'\n2026 stand-4xx cell rows {len(Z)}   firing {int(Z.fire.sum())}   target {TGT:,.0f}')
p=b.TAXITIME_SEC_mvt.values.astype(float).copy()
idx={int(v):i for i,v in enumerate(b.MVT_ID_mvt.values)}
for _,r in Z[Z.fire].iterrows():
    assert str(r.STAND_mvt).startswith('4') and r.ms>=THR
    if FIRE: p[idx[int(r.MVT_ID_mvt)]]=TGT
q=np.clip(np.rint(p),FLOOR,CAP).astype('int32'); b32=b.TAXITIME_SEC_mvt.values.astype('int32')
nd=int((q!=b32).sum()); Q=float(np.sum((q.astype(float)-b32.astype(float))**2))
print(f'  rows changed {nd}   Q {Q:.4e} = {Q/PER:.3f} board-s')
if nd: print(f'  single-row mass {float(np.max((q.astype(float)-b32.astype(float))**2)/Q):.1%}')
assert (nd>0)==FIRE and nd<=len(Z)
o=pd.DataFrame({'MVT_ID_mvt':tb.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':q})
t=pa.Table.from_pandas(o,schema=tpl.schema,preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t,OUT)
print(f'  wrote {OUT}  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
if not FIRE:
    assert (pq.read_table(OUT).to_pandas().TAXITIME_SEC_mvt.values==b32).all(), 'CONTROL FAILED'
    print('  CONTROL: FIRE=0 reproduces v91 exactly')
