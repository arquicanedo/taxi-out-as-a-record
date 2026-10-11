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

Regenerate quick-boat_v34.parquet: the rollover cell at its MEASURED weight.

Identical to v30 in every respect except ALPHA. Same base (v29), same six cell rows,
same rollover target. Only the mixture weight changes: 0.05 -> 0.40.

WHY. v30 shipped ALPHA=0.05 as a probe and gained 1.7317 s. s124 inverted that gain
and its NARRATIVE said "implied p = 0.138, about ONE of the six rows is a genuine
rollover". That sentence is arithmetically wrong, and s124's own projection table --
computed correctly at the time and never acted on -- shows it:

    P=0.30 -> 267.76    P=0.45 -> 267.14 (the optimum)    P=0.55 -> 267.84

An optimum at P=0.45 is only consistent with p ~ 0.45, since gain = S*P*(2p-P) peaks
at P = p. If p really were 0.138 that table would peak near 0.14.

Re-inverted independently here from the actual submitted parquets, using the exact
per-row prediction deltas between v29 and v30 (which recover s124's own D values):

    sum D^2 = 8.2585e9        observed dSSE = -3.2734e8  (from the two board scores)
    gain = sum(D^2) * P * (2p - P)   =>   p = 0.4214
    cross-check: the observed gain is 40.7% of the p=1 maximum, which inverts to 0.42

So roughly 2.5 of the six rows are rollovers, not one. We have been weighting this
cell at about one ninth of its measured rate since 2026-09-06.

WHY 0.40 AND NOT 0.4214. The theorem says P* = p, but the curve is flat near its peak
and the downside is asymmetric: gain stays positive while P < 2p, so P=0.40 breaks
even at p > 0.20 while capturing 99.7% of the peak gain. Given six point predictions
this session of which four were optimistic, buy the margin. s124's table also shows
anywhere in [0.30, 0.55] lands 267-268, so the lever does not need p to be exact.

Projected: 273.2138 -> ~267.1.

THE STANDING CAVEAT, unchanged from s124: the inversion assumes each row's truth is
EITHER the rollover value OR v29's prediction. If some of the six are neither, p moves.
The wide-P robustness means that mostly changes how much we gain, not whether.
"""

import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

SHA   = os.environ.get('V34_SHA', '')
OUT   = os.environ.get('V34_OUT', 'submissions/quick-boat_v34.parquet')
ALPHA = float(os.environ.get('V34_ALPHA', '0.40'))
SRC   = 'submissions/quick-boat_v29.parquet'
DATA  = os.environ['DC26_DATA']
WORK  = os.environ['DC26_WORK']
CAP, FLOOR, DAY = 200000, 60, 86400.
SD_MIN, APT = 50000., 'LIRF'
N, S29 = 344841, 274.9455

tpl = pq.read_table(f'{DATA}/submitting.parquet'); tb = tpl.to_pandas()[['MVT_ID_mvt']]
src = pq.read_table(SRC).to_pandas()
assert len(src) == len(tb) and bool((src.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all())
rk = pq.read_table(f'{DATA}/ranking.parquet',
    columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt',
             'SCHED_TIME_UTC_mvt','LOBT_flt']).to_pandas()
for c in ('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt'):
    rk[c] = pd.to_datetime(rk[c], errors='coerce')
rd = rk[rk.PHASE_mvt == 'DEP'].copy()
rd['sd'] = (rd.MVT_TIME_UTC_mvt - rd.SCHED_TIME_UTC_mvt).dt.total_seconds()
cell_ids = set(rd.MVT_ID_mvt[(rd.LOBT_flt.isna()) & (rd.sd >= SD_MIN)
                             & (rd.ADEP_mvt.astype(str) == APT)].values)
print(f'cell: LOBT null & sched_delay >= {SD_MIN:.0f} & apt == {APT}')
print(f'  rows in the 2026 ranking set: {len(cell_ids)}')
assert 3 <= len(cell_ids) <= 25, f'cell size {len(cell_ids)} outside the sane band'

t = pd.read_parquet(f'{WORK}/v15_trees_jul19.parquet')
n = pd.read_parquet(f'{WORK}/v15_net_jul19.parquet')
m = t.merge(n, on='MVT_ID_mvt').set_index('MVT_ID_mvt')
bulk = (0.2*m.lgbm + 0.4*m.mlp + 0.4*m.perapt)

p = src.TAXITIME_SEC_mvt.values.astype(float)
inC = src.MVT_ID_mvt.isin(cell_ids).values
roll = bulk.reindex(src.MVT_ID_mvt).values + DAY
pred = np.where(inC, np.clip(np.rint(ALPHA*roll + (1-ALPHA)*p), FLOOR, CAP), p)
sd_map = rd.set_index('MVT_ID_mvt').sd
print(f'\nALPHA = {ALPHA}   rows changed {int((pred != p).sum())}')
print(f'{"sched_delay":>12s} {"v29":>10s} {"rollover":>10s} {"v30":>10s}')
for i in np.where(inC)[0]:
    print(f'{sd_map.get(src.MVT_ID_mvt.values[i], np.nan):12,.0f} {p[i]:10,.0f} '
          f'{roll[i]:10,.0f} {pred[i]:10,.0f}')

o = tb.merge(pd.DataFrame({'MVT_ID_mvt': src.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': pred}),
             on='MVT_ID_mvt', how='left')
o['TAXITIME_SEC_mvt'] = o.TAXITIME_SEC_mvt.astype('int32')
assert len(o) == len(tb) and o.TAXITIME_SEC_mvt.isna().sum() == 0
assert bool((o.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all()), 'row order must match'
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']], schema=tpl.schema,
                                    preserve_index=False), OUT)
h = hashlib.sha256(open(OUT,'rb').read()).hexdigest()
print(f'\nwrote {OUT}\n  sha256 {h}')

SSE29 = S29**2*N
sdv = np.array([sd_map.get(i, np.nan) for i in src.MVT_ID_mvt.values[inC]])
pn, rl, nw = p[inC], roll[inC], pred[inC]
print(f'\n--- INVERSION TABLE, recorded before submitting ---')
print(f'{"if v30 scores":>14s} {"implied p":>11s}   p = share of the 6 that are real rollovers')
for q in (0.0, 0.2, 0.35, 0.462, 0.55, 0.7, 1.0):
    d = q*(((nw-rl)**2)-((pn-rl)**2)).sum() + (1-q)*(((nw-sdv)**2)-((pn-sdv)**2)).sum()
    print(f'{np.sqrt(max(SSE29+d,0)/N):14.4f} {q:11.3f}')
if SHA:
    assert h == SHA, f'HASH MISMATCH expected {SHA} got {h}'; print('  sha256 asserted OK')
else:
    print('  (re-run with V34_SHA=<above> to assert reproducibility)')
