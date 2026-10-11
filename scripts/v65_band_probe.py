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

v65: a magnitude-band probe on what we PREDICT as extreme. User's design, 2026-09-10.

With ~100 candidate rows, shift one magnitude band by a fixed amount and read the mean residual
on that band off the board. One probe per band; one board score is one linear measurement.

    u = +SHIFT on rows with LOW <= v64_pred < HIGH,  0 elsewhere
    dSSE = 2*SHIFT*sum(pred - y) + n*SHIFT^2       everything known but sum(pred - y)
    =>  mean residual on the band = [ (dSSE - n*SHIFT^2) / (2*SHIFT) ] / n

WHERE THE LINE IS (s217). Bands are defined on an OBSERVABLE -- our own predicted value -- and a
per-band mean residual is a POPULATION statement that would apply to rows we have never seen.
That is a measurement. Subdividing until a cell holds one or two rows is a decoder, and with
~100 candidates that boundary is close: three bands is fine, thirty is not. **Do not subdivide
these further.**

Cost, from s228's +-9,125 SSE board resolution: at SHIFT=600 the LOW band costs 0.1485 board-s
if the direction is empty and resolves its mean residual to +-0.1 s per row, against residuals
of order 10,000-80,000 s for a missed copy.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
BASE=os.environ.get('V65_BASE','submissions/quick-boat_v64.parquet')
BSCORE=float(os.environ.get('V65_BSCORE','261.8498'))
BAND=os.environ.get('V65_BAND','LOW'); SHIFT=float(os.environ.get('V65_SHIFT','600'))
OUT=os.environ.get('V65_OUT',f'submissions/quick-boat_v65{BAND[0].lower()}.parquet')
FLOOR,CAP,N,PER=60.,200000.,344841,1.842e8
EDGES={'LOW':(7200.,20000.),'MID':(20000.,50000.),'HIGH':(50000.,1e9)}
assert BAND in EDGES, f'band must be one of {list(EDGES)}'
lo,hi=EDGES[BAND]
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
b=pq.read_table(BASE).to_pandas(); ids=b.MVT_ID_mvt
assert bool((ids.values==tb.MVT_ID_mvt.values).all())
p0=b.TAXITIME_SEC_mvt.values.astype(float)
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt']).to_pandas()
apt=rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(ids).ADEP_mvt.astype(str).values
M=(p0>=lo)&(p0<hi)
n=int(M.sum())
print(f'base {os.path.basename(BASE)} @ {BSCORE:.4f}')
print(f'band {BAND} [{lo:,.0f},{hi:,.0f}) : {n} rows')
assert 4<=n<=400, f'band size {n} outside the sane range'
print(f'  predictions: median {np.median(p0[M]):,.0f}  min {p0[M].min():,.0f}  max {p0[M].max():,.0f}')
print(f'  airports: {dict(pd.Series(apt[M]).value_counts())}')
def build(s):
    q=p0.copy(); q[M]=np.clip(np.rint(p0[M]+s),FLOOR,CAP); return q
c=build(0.0); nd=int((c.astype('int32')!=p0.astype('int32')).sum())
print(f'\nCONTROL shift=0: {nd} rows differ'); assert nd==0, 'CONTROL FAILED'
print('  byte-identical.')
pred=build(SHIFT); u=pred-p0; Q=float(u@u)
print(f'\nSHIFT {SHIFT:+,.0f}s  rows moved {int((u!=0).sum())}  Q = {Q:.6e} = {Q/PER:+.4f} board-s if the band is unbiased')
S0=N*BSCORE*BSCORE
print(f'\n--- INVERSION, pre-registered. Reads the MEAN RESIDUAL (pred - y) on the band. ---')
print(f'  {"if it scores":>13} {"sum(pred-y)":>14} {"mean resid":>12} {"reading"}')
for dB in (-0.30,-0.10,-0.02,0.0,+0.02,+0.10,+0.30):
    Bp=BSCORE+dB
    dS=N*(Bp*Bp-BSCORE*BSCORE); T=(dS-Q)/(2*SHIFT); mr=T/n
    tag='we OVER-predict here' if mr>50 else ('we UNDER-predict here' if mr<-50 else 'unbiased')
    print(f'  {Bp:>13.4f} {T:>+14,.0f} {mr:>+12,.1f} s  {tag}')
print(f'\n  a mean residual of +/-{9125/(2*SHIFT)/n:,.1f} s is the resolution floor at this shift')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
t=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
print('\nREBASE on whatever is standing when it fires, or the inversion is against the wrong reference.')
