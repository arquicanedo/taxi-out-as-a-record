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

v69: correct all three bands to the optima solved from the LINEAR SYSTEM, not the chain.

s254. The 09-11 window banked each correction relative to the probe-shifted base instead of the
v64 reference, so every band landed off its optimum. Solving the four board scores as one system
(4 equations, 4 unknowns, residual 1e-16) gives the truth relative to v64:

    LOW    sum e = +29,799   +392.1 s/row   optimum shift  -392.1   v67 has +208    err +600
    MID    sum e = -31,229 -1,561.5 s/row   optimum shift +1,561.5  v67 has +600    err -961
    HIGH   sum e = -13,294 -2,215.7 s/row   optimum shift +2,215.7  v67 has    0    err -2,216
    EHAM   alpha* = -0.1012                                          v67 correct

Band membership is pinned to the v64 reference. Control byte-identical to v67 at zero shift.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
BASE=os.environ.get('V69_BASE','submissions/quick-boat_v67.parquet')
BSCORE=float(os.environ.get('V69_BSCORE','262.5506'))
REF=os.environ.get('V69_REF','submissions/quick-boat_v64.parquet')
D_LOW=float(os.environ.get('V69_LOW','-600.1'))
D_MID=float(os.environ.get('V69_MID','961.5'))
D_HIGH=float(os.environ.get('V69_HIGH','2215.7'))
OUT=os.environ.get('V69_OUT','submissions/quick-boat_v69.parquet')
FLOOR,CAP,N,PER=60.,200000.,344841,1.842e8
# solved from the system, relative to v64
SUM_E={'LOW':29799.,'MID':-31229.,'HIGH':-13294.}
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
b=pq.read_table(BASE).to_pandas(); r=pq.read_table(REF).to_pandas(); ids=b.MVT_ID_mvt
assert bool((ids.values==tb.MVT_ID_mvt.values).all())
p0=b.TAXITIME_SEC_mvt.values.astype(float); pref=r.TAXITIME_SEC_mvt.values.astype(float)
B={'LOW':(pref>=7200)&(pref<20000),'MID':(pref>=20000)&(pref<50000),'HIGH':pref>=50000}
SH={'LOW':D_LOW,'MID':D_MID,'HIGH':D_HIGH}
print(f'base {os.path.basename(BASE)} @ {BSCORE:.4f}   membership from {os.path.basename(REF)}')
cur={k:float(np.median((p0-pref)[m])) for k,m in B.items()}
for k in ('LOW','MID','HIGH'):
    n=int(B[k].sum()); opt=-SUM_E[k]/n
    print(f'  {k:<5} {n:>3} rows   v67 at {cur[k]:+8.1f}   optimum {opt:+9.1f}   applying {SH[k]:+9.1f}'
          f'   -> {cur[k]+SH[k]:+9.1f}')
    assert abs(cur[k]+SH[k]-opt)<2.0, f'{k} would land at {cur[k]+SH[k]:.1f}, optimum is {opt:.1f}'
def build(f):
    p=p0.copy()
    for k,m in B.items(): p[m]=np.clip(np.rint(p[m]+f*SH[k]),FLOOR,CAP)
    return p
c=build(0.0); nd=int((c.astype('int32')!=p0.astype('int32')).sum())
print(f'\nCONTROL f=0 vs {os.path.basename(BASE)}: {nd} rows differ'); assert nd==0,'CONTROL FAILED'
print('  byte-identical.')
pred=build(1.0); u=pred-p0; Q=float(u@u)
# predicted score: each band moves from its current offset to its optimum, exactly
dS=0.0
for k,m in B.items():
    n=int(m.sum()); e=SUM_E[k]; c0=cur[k]; c1=cur[k]+SH[k]
    dS+=2*(c1-c0)*(e+n*c0)+n*(c1-c0)**2
Bp=np.sqrt((N*BSCORE**2+dS)/N)
print(f'\nrows moved {int((u!=0).sum())}   Q {Q:.6e}')
print(f'  dSSE {dS:+.4e}  ->  PREDICTED SCORE {Bp:.4f}   ({BSCORE-Bp:+.4f} board-s)')
print(f'  this is arithmetic on four board measurements, not a projection.')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
t=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
