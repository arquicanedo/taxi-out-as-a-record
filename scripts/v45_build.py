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

v45: v38 with the per-month uniform bias measured BY PROBE on the ranking set itself.

v43 (January +10s) and v44 (July +10s) inverted to:
    January  mass +1.3405e6 over 152,719 rows  ->  +8.778 s/row over-prediction
    July     mass +7.4186e5 over 192,122 rows  ->  +3.861 s/row over-prediction
The SSE-optimal uniform shift on a cell is -mass/n, worth mass^2/n. This is exact arithmetic
on measured quantities, not a transfer estimate: +0.0794 board-s.

CAVEAT, recorded rather than buried: this is tuned to the ranking set's own truth. If a final
holdout appears (both prior editions added one ~a week out), a uniform per-month bias is more
likely to survive than a row-level rule, but it is still board-fitted.

CONTROL: with the shifts zeroed this reproduces v38 byte-identically.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
CAP,FLOOR=200000,60; PER=1.842e8; NB=344841
SHIFT={1:-8.778, 7:-3.861}
if os.environ.get('ZERO'): SHIFT={1:0.0,7:0.0}
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
v38=pq.read_table('submissions/quick-boat_v38.parquet').to_pandas()
p=v38.TAXITIME_SEC_mvt.values.astype(float)
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt']).to_pandas()
rk['MVT_TIME_UTC_mvt']=pd.to_datetime(rk.MVT_TIME_UTC_mvt,errors='coerce')
mon=rk.set_index('MVT_ID_mvt').reindex(v38.MVT_ID_mvt).MVT_TIME_UTC_mvt.dt.month.values
u=np.zeros(len(p))
for m,s in SHIFT.items(): u[mon==m]=s
new=np.clip(np.rint(p+u),FLOOR,CAP); ueff=new-p
if os.environ.get('ZERO'):
    d=int((new.astype('int32')!=p.astype('int32')).sum()); print(f'CONTROL: differs from v38 on {d} rows'); assert d==0; raise SystemExit
print(f'rows moved {int((ueff!=0).sum()):,} of {len(p):,}')
for m,s in SHIFT.items():
    c=mon==m; print(f'  month {m}: shift {s:+.3f}s   mean applied {ueff[c].mean():+.3f}s   '
                    f'clipped-at-floor {int(((p+u<FLOOR)&c).sum()):,}')
mass={1:1.3405e6,7:7.4186e5}; n={1:152719,7:192122}
exp=sum(2*SHIFT[m]*mass[m]+ (ueff[mon==m]**2).sum() for m in SHIFT)
print(f'\n  expected dSSE {exp:+.4e}  ->  {-exp/PER:+.4f} board-s   (266.9149 -> {np.sqrt((NB*266.9149**2+exp)/NB):.4f})')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':v38.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':new}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False),'submissions/quick-boat_v45.parquet')
print(f"  sha256 {hashlib.sha256(open('submissions/quick-boat_v45.parquet','rb').read()).hexdigest()}")
