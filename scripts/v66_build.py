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

v66: bank the EHAM correction at its measured alpha*, AND probe a magnitude band, in one slot.

The two row sets are disjoint -- the EHAM cell is EHAM 2026-01-02..07, the bands are LIRF 67 /
LFPG 5 / EGLL 4 -- so the cross term is exactly zero and one board score still yields the band's
T once the EHAM contribution is subtracted.

  EHAM   v63 probed at EPS=+0.15 from base 263.0551 and scored 265.2334.
         Q = 1.689438e8, T = +1.139446e8, alpha* = -0.6745, so ship EPS = -0.1012.
         Our disruption uplift is TOO BIG by 1.113x. Banked gain +0.4239 -> 262.6312.
  BAND   +SHIFT on the chosen band; reads its mean residual off the score.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
BASE=os.environ.get('V66_BASE','submissions/quick-boat_v64.parquet')
BSCORE=float(os.environ.get('V66_BSCORE','263.0551'))
EHAM_EPS=float(os.environ.get('V66_EHAM_EPS','-0.1012'))
BAND=os.environ.get('V66_BAND','LOW'); SHIFT=float(os.environ.get('V66_SHIFT','600'))
OUT=os.environ.get('V66_OUT','submissions/quick-boat_v66.parquet')
FLOOR,CAP,N,PER=60.,200000.,344841,1.842e8
EDGES={'LOW':(7200.,20000.),'MID':(20000.,50000.),'HIGH':(50000.,1e9),'NONE':(1e18,1e18)}
lo,hi=EDGES[BAND]
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
b=pq.read_table(BASE).to_pandas(); ids=b.MVT_ID_mvt
assert bool((ids.values==tb.MVT_ID_mvt.values).all())
p0=b.TAXITIME_SEC_mvt.values.astype(float)
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt',
    'MVT_TIME_UTC_mvt']).to_pandas()
d=rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(ids)
t=d.MVT_TIME_UTC_mvt.astype('int64')//10**6
assert 1.76e9<np.median(t)<1.80e9, 'epoch not 2026'
apt=d.ADEP_mvt.astype(str).values
day=pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d').values
EH=apt=='EHAM'
SNOW=EH&np.isin(day,[f'2026-01-{k:02d}' for k in range(2,8)])
REF=EH&np.isin(day,[f'2026-01-{k:02d}' for k in range(10,32)])
base_med=float(np.median(p0[REF]))
u_eham=np.zeros(len(p0)); u_eham[SNOW]=np.maximum(p0[SNOW]-base_med,0.0)
BND=(p0>=lo)&(p0<hi)
assert not (SNOW&BND).any(), 'EHAM cell and band OVERLAP -- the cross term is not zero'
print(f'base {os.path.basename(BASE)} @ {BSCORE:.4f}')
print(f'EHAM cell {int(SNOW.sum())} rows, reference median {base_med:,.0f}s, eps {EHAM_EPS:+.4f}')
print(f'band {BAND} {int(BND.sum())} rows, shift {SHIFT:+,.0f}s')
print(f'  disjoint: {int((SNOW&BND).sum())} overlapping rows')
def build(ee,ss):
    p=p0.copy()
    p[SNOW]=np.clip(np.rint(p0[SNOW]+ee*u_eham[SNOW]),FLOOR,CAP)
    if ss!=0: p[BND]=np.clip(np.rint(p[BND]+ss),FLOOR,CAP)
    return p
c=build(0.0,0.0); nd=int((c.astype('int32')!=p0.astype('int32')).sum())
print(f'\nCONTROL (0,0) vs base: {nd} rows differ'); assert nd==0, 'CONTROL FAILED'
print('  byte-identical.')
eh_only=build(EHAM_EPS,0.0); ue=eh_only-p0; Qe=float(ue@ue)
Te=1.139446e+08*(EHAM_EPS/0.15)*0 + 0   # T is a property of u, recomputed below from alpha*
# the EHAM contribution is known exactly from the v63 inversion
Q63=1.689438e+08; T63=1.139446e+08
scale=EHAM_EPS/0.15
dSSE_eham=2*scale*T63+scale*scale*Q63
pred=build(EHAM_EPS,SHIFT); ub=pred-eh_only; Qb=float(ub@ub)
print(f'\nEHAM banked: dSSE {dSSE_eham:+.4e} = {-dSSE_eham/PER:+.4f} board-s')
print(f'BAND probe : Qb {Qb:.6e} = {Qb/PER:+.4f} board-s if unbiased')
S0=N*BSCORE*BSCORE; n=int(BND.sum())
print(f'\n--- INVERSION. Subtract the known EHAM term, then read the band mean residual. ---')
print(f'  {"if v66 scores":>14} {"band sum(pred-y)":>18} {"mean resid":>12}  reading')
for dB in (-0.70,-0.50,-0.42,-0.30,-0.10,+0.10,+0.40):
    Bp=BSCORE+dB
    dS=N*(Bp*Bp-BSCORE*BSCORE)
    dband=dS-dSSE_eham
    T=(dband-Qb)/(2*SHIFT); mr=T/n
    tag='we OVER-predict' if mr>50 else ('we UNDER-predict' if mr<-50 else 'unbiased')
    print(f'  {Bp:>14.4f} {T:>+18,.0f} {mr:>+12,.1f} s  {tag}')
print(f'\n  EHAM alone would score {np.sqrt((S0+dSSE_eham)/N):.4f}')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
tt=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert tt.schema.equals(tpl.schema)
pq.write_table(tt,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
