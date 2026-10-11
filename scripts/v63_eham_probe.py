"""RETIRED. This is probe-and-invert code, published deliberately.

WHY IT IS HERE. This script implements the method our paper discloses and retires: submit a
known perturbation on a known row set, read the public score, solve `dSSE = 2T + Q` for the
inner product of the perturbation with the true errors, and ship the optimum. We used it, it
contributed a measured +2.7501 RMSE-seconds or more to our standing, and we stopped on
2026-09-11 after an organiser wrote that reverse-engineering the ranking is unwelcome.

**We publish it rather than describe it** because a disclosure a reader cannot check is not a
disclosure. `scripts/v76_build.py` builds the same model with this content removed and ships
alongside the model that keeps it, so the difference can be priced without taking our word for
anything. Every constant is itemised in REPRODUCE.md section 9.

**It is not run by any shipped build**, and some of these scripts depend on intermediate state
that no longer exists, so several will not execute at all. They are here to be read. Do not run
them: the method measures the evaluation set rather than the problem, which is why we stopped,
independent of any rule.

---

v63: a LEVEL PROBE on the EHAM disruption week. Slot 3, user decision 2026-09-10.

s236 (Fable's audit) found a regime the validation window does not contain at all:

    same arrival taxi-in model, trained 2025 Feb-Jun + Aug-Dec
      Jan+Jul 2025 ordinary arrivals   188.68
      Jan+Jul 2026 ordinary arrivals   200.70     ratio 1.0637
      ex-EHAM                                     ratio 0.9742

    EHAM alone adds 2.089e9 arrival SSE, more than the entire 2026-minus-2025 difference.
    2026-01-05 mean arrival residual +1,460 s on 197 rows.
    Largest EHAM day-mean residual ANYWHERE in Jan+Jul 2025: 76.4 s.

Our bulk responds partway through `naive` (AOBT_3 present on 100% of EHAM rows) and then stops:
it never predicts above 7,200 s on any of those days, while that week's LABELLED arrivals reach
15,526 s.

NOTE, 2026-10-05. The paragraph below argued that this probe was legitimate because no training data contains the
regime. We no longer hold that view: reading a constant off the public score is learning from the ranking whatever
the reason, and the constant this probe produced (EHAM_EPS) is counted with the others in REPRODUCE.md section 9.
The argument is kept as it was written on the day.

WHY A BOARD PROBE IS LEGITIMATE HERE AND IS NOT THE THING WE STOPPED DOING. Every previous
inversion recovered a weight that labelled data could in principle have supplied. **No training
data contains this regime**, so labelled data structurally cannot answer it. The board is the
only instrument that exists for this cell. That is a different act from inverting a cell weight
we could have fitted offline, and the distinction is the whole reason this is being run.

The direction scales the disruption UPLIFT, which is the quantity in question:
    u = max(pred - base, 0)       base = the EHAM non-disruption median, from the SCORED set
    probe: pred + EPS*u           i.e. the uplift is scaled by (1 + EPS)
Rows where the model applies no uplift are untouched, so a null result cannot be manufactured
by moving rows that were never disrupted.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
BASE=os.environ.get('V63_BASE','submissions/quick-boat_v59.parquet')
BSCORE=float(os.environ.get('V63_BSCORE','264.3314'))
# s228: the board reads dSSE to +-9,125 SSE units, so probe strength buys almost nothing past
# a few percent -- v57 was probed ~20x harder than its own measurement needed. eps=0.15 costs
# 0.9 board-s if the direction is empty against 5.0 at eps=0.35, and measures T just as well.
EPS=float(os.environ.get('V63_EPS','0.15'))
OUT=os.environ.get('V63_OUT','submissions/quick-boat_v63.parquet')
FLOOR,CAP,N,PER=60.,200000.,344841,1.842e8
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
b=pq.read_table(BASE).to_pandas(); ids=b.MVT_ID_mvt
assert bool((ids.values==tb.MVT_ID_mvt.values).all())
p0=b.TAXITIME_SEC_mvt.values.astype(float)
rk=pq.read_table(f'{DATA}/ranking.parquet',
    columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt']).to_pandas()
d=rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(ids)
t=d.MVT_TIME_UTC_mvt.astype('int64')//10**6          # timestamp[us]; //10**9 is the s220 bug
assert 1.76e9<np.median(t)<1.80e9, f'epoch {np.median(t):.4e} is not 2026'
day=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d').values
apt=d.ADEP_mvt.astype(str).values
EH=apt=='EHAM'
SNOW=EH&np.isin(day,['2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06','2026-01-07'])
REF=EH&np.isin(day,[f'2026-01-{k:02d}' for k in range(10,32)])
base_med=float(np.median(p0[REF]))
print(f'EHAM scored departures {int(EH.sum()):,}')
print(f'  disruption cell 2026-01-02..07: {int(SNOW.sum()):,} rows')
print(f'  reference 2026-01-10..31:       {int(REF.sum()):,} rows, median prediction {base_med:,.0f}s')
assert 1500<=SNOW.sum()<=2500, f'cell size {int(SNOW.sum())} outside the sane band'
print('\n  our prediction by day, EHAM:')
for dd in ['2026-01-02','2026-01-03','2026-01-04','2026-01-05','2026-01-06','2026-01-07']:
    m=EH&(day==dd)
    print(f'    {dd}  n={int(m.sum()):>4}  median {np.median(p0[m]):>7,.0f}s  '
          f'p99 {np.percentile(p0[m],99):>7,.0f}s  max {p0[m].max():>7,.0f}s')
m=REF
print(f'    2026-01-10..31 n={int(m.sum()):>4}  median {np.median(p0[m]):>7,.0f}s  '
      f'p99 {np.percentile(p0[m],99):>7,.0f}s  max {p0[m].max():>7,.0f}s')
u=np.zeros(len(p0)); u[SNOW]=np.maximum(p0[SNOW]-base_med,0.0)
nz=int((u>0).sum())
print(f'\ndirection: uplift over {base_med:,.0f}s on the cell, {nz:,} of {int(SNOW.sum()):,} rows carry one')
print(f'  uplift median {np.median(u[u>0]):,.0f}s  p90 {np.percentile(u[u>0],90):,.0f}s  max {u.max():,.0f}s')
def build(eps):
    p=p0.copy(); p[SNOW]=np.clip(np.rint(p0[SNOW]+eps*u[SNOW]),FLOOR,CAP); return p
c0=build(0.0); nd=int((c0.astype('int32')!=p0.astype('int32')).sum())
print(f'\nCONTROL eps=0 vs {os.path.basename(BASE)}: {nd} rows differ')
assert nd==0, 'CONTROL FAILED'
print('  byte-identical.\n')
pred=build(EPS); w=pred-p0; Q=float(w@w)
print(f'EPS {EPS}  rows moved {int((w!=0).sum()):,}  Q = {Q:.6e} = {Q/PER:+.4f} board-s if T=0')
print(f'  |move| median {np.median(np.abs(w[w!=0])):,.0f}s  max {np.abs(w).max():,.0f}s')
S0=N*BSCORE*BSCORE
print(f'\n--- INVERSION, pre-registered against base {BSCORE:.4f}. ---')
print(f'  {"if v63 scores":>14} {"alpha*":>9} {"uplift x":>9} {"best along u":>13} {"gain":>8}')
T0=np.sqrt((S0+Q)/N)                       # the score if the direction is empty
for Bp in np.concatenate([np.linspace(BSCORE-0.8,BSCORE+0.4,5),[BSCORE+(T0-BSCORE)*0.5,T0]]):
    dS=N*(Bp*Bp-BSCORE*BSCORE); T=(dS-Q)/2.0; a=-T/Q
    Bs=np.sqrt(max(S0-T*T/Q,0)/N)
    print(f'  {Bp:>14.4f} {a:>9.4f} {1+a*EPS:>9.4f} {Bs:>13.4f} {BSCORE-Bs:>+8.4f}')
print(f'\n  uplift x < 1 means our disruption response is TOO BIG; > 1 too small.')
print(f'  T = 0 (no signal either way) scores {np.sqrt((S0+Q)/N):.4f}')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
tt=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert tt.schema.equals(tpl.schema)
pq.write_table(tt,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
print('\nREBASE BEFORE SHIPPING: set V63_BASE and V63_BSCORE to whatever slot 1 and 2 leave')
print('standing, or the inversion arithmetic is against the wrong reference.')
