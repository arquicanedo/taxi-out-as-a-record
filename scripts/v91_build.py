"""v91 = v90 with ONE row corrected: TAY9LA cannot be a date rollover.

MECHANISM (Fable 2026-09-18, verified independently here). On all 1,488 LIRF LOBT-null departures
of 2025, in Europe/Rome local time:

    every one of the 12 rollovers has  localdate(BLOCK) == localdate(SCHED)
                                  and  localdate(MVT)   == localdate(SCHED) + 1
    0 of the 1,320 NON-crossing rows ever rolled over;  all 12 rollovers are among the 168 crossing
    crossing rows with ms >= 20000:  36 copy, 11 roll, 0 ordinary
    NON-crossing rows with ms >= 30000:  8 of 8 copy, 0 roll

So BLOCK on a rollover is the SCHEDULE's local DATE joined to the actual off-block's local
TIME-OF-DAY, and a rollover therefore REQUIRES the take-off to cross the local date relative to the
schedule. TAY9LA (MVT_ID 202687258) is scheduled 2026-01-22 06:45 local and departs 2026-01-22
21:37 local -- the same local day -- so ROLL is structurally impossible for it, and the v34 cell's
flat 0.40 ROLL weight is wrong on that row for a reason, not by a coin flip.

WHY THIS IS NOT YESTERDAY'S KILLED RE-WEIGHTING. That one moved weights on counted rates of 3/5 and
5/13, whose sampling variance (0.048, 0.018) exceeded the bias they corrected (0.040, 0.0002). Here
the rate is 0 of 1,320: variance ~0 against a bias of (0.40 - 0)^2 = 0.16.

DOMINANCE. The two admissible truths are copy (y = ms = 53,520) and ordinary taxi. Writing 53,520
is better than v90's 67,165 under BOTH -- it is not a bet between them. It loses only under
rollover, which the mechanism excludes.

THE PREDICATE, not a row picked by hand: the v34 cell (LIRF, LOBT null, ms >= 50,000 s) minus every row whose take-off
crosses the local date relative to its schedule. In the 2026 set the cell holds six rows and exactly one does not cross;
the code below applies the predicate to the whole cell and asserts that it selects that one row.

CONTROL: with APPLY=0 the output reproduces v90's values exactly.
"""
import hashlib, os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D=os.environ['DC26_DATA']
OUT=os.environ.get('V91_OUT','submissions/quick-boat_v91.parquet')
APPLY=os.environ.get('APPLY','1')=='1'
MVT_ID=202687258.0; FLOOR,CAP=60.,200000.
rk=pq.read_table(f'{D}/ranking.parquet',columns=['MVT_ID_mvt','FLIGHT_mvt','ADEP_mvt','PHASE_mvt',
    'MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']).to_pandas()
dep=rk[rk.PHASE_mvt.eq('DEP')].copy()
_m=pd.to_datetime(dep.MVT_TIME_UTC_mvt,utc=True); _s=pd.to_datetime(dep.SCHED_TIME_UTC_mvt,utc=True)
cell=dep[dep.ADEP_mvt.eq('LIRF') & dep.LOBT_flt.isna() & ((_m-_s).dt.total_seconds()>=50000)]
nocross=cell[_m[cell.index].dt.tz_convert('Europe/Rome').dt.date.values==_s[cell.index].dt.tz_convert('Europe/Rome').dt.date.values]
print(f'v34 cell {len(cell)} rows; not crossing the local date: {len(nocross)}')
assert len(cell)==6 and list(nocross.MVT_ID_mvt)==[MVT_ID], 'the predicate no longer selects exactly this row'
r=rk[rk.MVT_ID_mvt.eq(MVT_ID)]
assert len(r)==1, f'expected exactly one row for {MVT_ID}, got {len(r)}'
r=r.iloc[0]
mvt=pd.to_datetime(r.MVT_TIME_UTC_mvt,utc=True); sch=pd.to_datetime(r.SCHED_TIME_UTC_mvt,utc=True)
ml,sl=mvt.tz_convert('Europe/Rome'),sch.tz_convert('Europe/Rome')
ms=(mvt-sch).total_seconds()
assert r.ADEP_mvt=='LIRF' and pd.isna(r.LOBT_flt), 'row is not a LIRF LOBT-null row'
assert ml.date()==sl.date(), f'row DOES cross the local date ({sl.date()} -> {ml.date()}): mechanism does not apply'
print(f'{r.FLIGHT_mvt}  SCHED {sl:%Y-%m-%d %H:%M} local   MVT {ml:%Y-%m-%d %H:%M} local'
      f'   same local day -> ROLL impossible   ms {ms:,.0f}')
v90=pq.read_table('submissions/quick-boat_v90.parquet').to_pandas()
cur=float(v90.loc[v90.MVT_ID_mvt.eq(MVT_ID),'TAXITIME_SEC_mvt'].iloc[0])
tgt=float(np.clip(round(ms),FLOOR,CAP))
p=v90.TAXITIME_SEC_mvt.values.astype(float).copy()
if APPLY: p[v90.MVT_ID_mvt.values==MVT_ID]=tgt
p=np.clip(np.rint(p),FLOOR,CAP)
d=p-v90.TAXITIME_SEC_mvt.values.astype(float); Q=float(np.sum(d**2))
print(f'  v90 {cur:,.0f}  ->  v91 {tgt:,.0f}   delta {tgt-cur:+,.0f}   rows moved {int((d!=0).sum())}   Q {Q:.4e}')
N=344841; S=259.2408
for lab,truth in (('copy     y = ms',ms),('ordinary y = 1,000',1000.),('rollover y = 87,300 (EXCLUDED)',87300.)):
    b=N*S*S+ (tgt-truth)**2-(cur-truth)**2
    print(f'    if {lab:<30} board {S:.4f} -> {np.sqrt(b/N):8.4f}  ({S-np.sqrt(b/N):+7.4f})')
tpl=pq.read_table(f'{D}/submitting.parquet')
o=pd.DataFrame({'MVT_ID_mvt':v90.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':p.astype('int32')})
tb=pa.Table.from_pandas(o,schema=tpl.schema,preserve_index=False); assert tb.schema.equals(tpl.schema)
pq.write_table(tb,OUT)
print(f'wrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
if not APPLY:
    ref=pq.read_table('submissions/quick-boat_v90.parquet').to_pandas()
    ok=bool((ref.values==o.values).all()); print(f'  CONTROL APPLY=0: values {"IDENTICAL" if ok else "DIFFER -- FAIL"}')
    assert ok
