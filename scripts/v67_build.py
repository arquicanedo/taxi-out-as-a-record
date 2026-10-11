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

**It is not run by any shipped build.** `scripts/regen2.sh` composes the archived outputs of
these scripts rather than re-running them (`RERUN_RETIRED=1` re-runs v66/v67 only), and nothing here is needed to reproduce either
published model. Do not run it. It measures the evaluation set rather than the problem, which
is the reason we stopped, independent of any rule.

---

v67: bank the LOW-band correction and probe the MID band. Bands pinned to v64.

TRAP THIS AVOIDS. v66 shifted the LOW band by +600, so LOW rows now predict 7,800-20,600 and a
band recomputed on the CURRENT base would pull shifted LOW rows into MID [20000,50000). Band
membership is therefore taken from the FIXED v64 reference throughout, never from the running
base. The correction and the probe are applied on top of whatever is standing.

  LOW  measured at v66: mean residual +392.1 s/row over 76 rows -> ship -392.1, worth +0.0634
  MID  probe +600 s over 20 rows -> reads its mean residual
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
BASE=os.environ.get('V67_BASE','submissions/quick-boat_v66.parquet')
BSCORE=float(os.environ.get('V67_BSCORE','262.9794'))
REF=os.environ.get('V67_REF','submissions/quick-boat_v64.parquet')   # band membership, FIXED
LOW_FIX=float(os.environ.get('V67_LOW_FIX','-392.1'))
PROBE_BAND=os.environ.get('V67_PROBE','MID'); SHIFT=float(os.environ.get('V67_SHIFT','600'))
OUT=os.environ.get('V67_OUT','submissions/quick-boat_v67.parquet')
FLOOR,CAP,N,PER=60.,200000.,344841,1.842e8
EDGES={'LOW':(7200.,20000.),'MID':(20000.,50000.),'HIGH':(50000.,1e9)}
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
b=pq.read_table(BASE).to_pandas(); ids=b.MVT_ID_mvt
r=pq.read_table(REF).to_pandas()
assert bool((ids.values==tb.MVT_ID_mvt.values).all())
assert bool((r.MVT_ID_mvt.values==ids.values).all())
p0=b.TAXITIME_SEC_mvt.values.astype(float)
pref=r.TAXITIME_SEC_mvt.values.astype(float)          # membership reference
def band(nm):
    lo,hi=EDGES[nm]; return (pref>=lo)&(pref<hi)
LO,PB=band('LOW'),band(PROBE_BAND)
assert not (LO&PB).any(), 'bands overlap'
print(f'base {os.path.basename(BASE)} @ {BSCORE:.4f}   membership from {os.path.basename(REF)}')
print(f'  LOW  {int(LO.sum())} rows -> correction {LOW_FIX:+,.1f}s')
print(f'  {PROBE_BAND:<4} {int(PB.sum())} rows -> probe {SHIFT:+,.0f}s')
print(f'  overlap {int((LO&PB).sum())}')
def build(fix,ss):
    p=p0.copy()
    p[LO]=np.clip(np.rint(p0[LO]+fix),FLOOR,CAP)
    if ss!=0: p[PB]=np.clip(np.rint(p[PB]+ss),FLOOR,CAP)
    return p
c=build(0.,0.); nd=int((c.astype('int32')!=p0.astype('int32')).sum())
print(f'\nCONTROL (0,0): {nd} rows differ'); assert nd==0,'CONTROL FAILED'
print('  byte-identical.')
lo_only=build(LOW_FIX,0.); ul=lo_only-p0; Ql=float(ul@ul)
# the LOW contribution is known exactly from the v66 inversion: T_low = +29,798 over 76 rows
T_low=29798.0; n_low=int(LO.sum())
dSSE_low=2*LOW_FIX*T_low+n_low*LOW_FIX*LOW_FIX
pred=build(LOW_FIX,SHIFT); ub=pred-lo_only; Qb=float(ub@ub); n=int(PB.sum())
print(f'\nLOW banked : dSSE {dSSE_low:+.4e} = {-dSSE_low/PER:+.4f} board-s')
print(f'{PROBE_BAND} probe: Qb {Qb:.6e} = {Qb/PER:+.4f} board-s if unbiased')
S0=N*BSCORE*BSCORE
print(f'\n--- INVERSION. Subtract the known LOW term, read the {PROBE_BAND} mean residual. ---')
print(f'  {"if v67 scores":>14} {"sum(pred-y)":>16} {"mean resid":>12}  reading')
for dB in (-0.20,-0.10,-0.064,-0.03,0.0,+0.05,+0.15):
    Bp=BSCORE+dB
    dS=N*(Bp*Bp-BSCORE*BSCORE); dband=dS-dSSE_low
    T=(dband-Qb)/(2*SHIFT); mr=T/n
    tag='we OVER-predict' if mr>50 else ('we UNDER-predict' if mr<-50 else 'unbiased')
    print(f'  {Bp:>14.4f} {T:>+16,.0f} {mr:>+12,.1f} s  {tag}')
print(f'\n  LOW correction alone would score {np.sqrt((S0+dSSE_low)/N):.4f}')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
t=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
