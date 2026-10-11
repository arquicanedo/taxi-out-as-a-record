"""v79 = v74 clipped onto a feasibility interval that the data proves, on 69 rows.

THE CONSTRAINT. For every one of the 2,062,577 labelled 2025 departures carrying `LOBT_flt`:

    | (MVT_TIME - LOBT_flt) - y |  <=  3606 s      0 violations, max exactly 3606

The bound is specific to LOBT, which is what makes it a pipeline invariant rather than a tail:

    clock        n           max |deviation|    violations > 3606
    LOBT      2,062,577          3,606                   0
    IOBT      2,062,577         10,737                  58
    EOBT_1    2,062,577         10,383                 614
    AOBT_3    2,062,577         20,657               1,769

3606 = 3600 + 6, an hour plus the rounding slack, which reads as a Network Manager filing rule.
`LOBT_flt` is present on 98.47% of the scored set, so the interval is computable there.

WHY THIS IS NOT A MODEL. It is arithmetic on two given timestamps. No weight is fitted, no
distribution is assumed, no leaderboard number is read. Clipping a prediction onto an interval
that provably contains the truth reduces the absolute error on every row it touches:

    p < lo <= y   =>   |y - lo| <= |y - p|,  and the SSE reduction is
    (y-p)^2 - (y-lo)^2 = (lo-p)(2y - p - lo)  >=  (lo-p)^2

with equality only when y sits exactly on the boundary. **So sum (clipped - p)^2 is a LOWER BOUND
on the gain, not an estimate of it.** Measured against the shipped v74:

    rows outside the interval          69 of 339,551   (63 below, 6 above)
    sum (clipped - p)^2                8.3248e+07
    guaranteed board gain             >= +0.4608 s

By airport: EHAM 25, LFPG 14, LEMD 10, EGLL 5, LIRF 5, LEBL 4, LSZH 3, LTFM 3. The largest single
row is an EHAM prediction of 4,943 s where MVT - LOBT is 13,414, so the truth is at least 9,808.

THE ONE RISK, STATED. The constraint is verified on 2025 only. Departure labels are blanked in
2026 and the arrival side carries no analogous bound (`BLOCK - ARVT_3` has 432 violations above
3606 in 2025), so 2026 cannot be checked directly. If the filing rule loosened in 2026 then some
of the 69 clips are wrong. The exposure is 69 rows and the mechanism looks like a hard pipeline
cap, so this submission also serves as the test of whether the invariant survived the year.

    credit: the LOBT interval was noticed in a public competitor repository
    (javidmardanov/PRC-Data-Challenge-2026, scripts/queue_features.py) during a survey of
    published challenge code. The bound above, the violation count and the gain are our own
    measurements on our own data; the observation that LOBT admits such a bound is theirs.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DATA = os.environ['DC26_DATA']
SUB  = os.environ.get('V79_SUB', 'submissions')
OUT  = os.environ.get('V79_OUT', f'{SUB}/quick-boat_v79.parquet')
SLACK = float(os.environ.get('V79_SLACK', '3606'))

tpl = pq.read_table(f'{DATA}/submitting.parquet'); tb = tpl.to_pandas()[['MVT_ID_mvt']]
rk = pq.read_table(f'{DATA}/ranking.parquet',
                   columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','LOBT_flt']).to_pandas()
rk = rk[rk.PHASE_mvt.eq('DEP')].copy()
for c in ('MVT_TIME_UTC_mvt','LOBT_flt'): rk[c] = pd.to_datetime(rk[c], errors='coerce')
rk['ml'] = (rk.MVT_TIME_UTC_mvt - rk.LOBT_flt).dt.total_seconds()
ml = pd.Series(rk.ml.values, index=rk.MVT_ID_mvt).reindex(tb.MVT_ID_mvt).values

base = pq.read_table(f'{SUB}/quick-boat_v74.parquet').to_pandas()
assert bool((base.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all())
p = base.TAXITIME_SEC_mvt.values.astype(float)

have = np.isfinite(ml)
lo = np.where(have, ml - SLACK, -np.inf)
hi = np.where(have, ml + SLACK,  np.inf)
q = np.clip(p, lo, hi)
q = np.clip(np.rint(q), 60., 200000.)          # the shipped floor and cap still bind

mv = q != p
print(f'LOBT present on {int(have.sum()):,} of {len(p):,} scored rows ({100*have.mean():.2f}%)')
print(f'rows outside the interval: {int(mv.sum())}   below {int((p<lo).sum())}  above {int((p>hi).sum())}')
d2 = float(((q - p)**2).sum())
N = len(p); B = 261.9189
print(f'sum (clipped - predicted)^2 = {d2:.4e}   guaranteed board gain >= {d2/(2*N*B):+.4f} s')
ap = pd.Series(rk.ADEP_mvt.astype(str).values, index=rk.MVT_ID_mvt).reindex(tb.MVT_ID_mvt).values
print(f'by airport: {dict(pd.Series(ap[mv]).value_counts())}')

# CONTROL 1: with an infinite slack nothing may move. The interval must be built through the
# same `have` mask as the real path, or NaN propagates on LOBT-null rows and the control fails
# on its own arithmetic rather than on the logic.
_lo = np.where(have, ml - 1e9, -np.inf); _hi = np.where(have, ml + 1e9, np.inf)
assert int((np.clip(np.rint(np.clip(p, _lo, _hi)), 60., 200000.) != p).sum()) == 0, \
    'CONTROL FAILED: an unbounded interval changed a row'
# CONTROL 2: every clip must move the prediction TOWARD the interval, never away
assert bool((np.abs(q[mv] - np.clip(p[mv], lo[mv], hi[mv])) < 1.01).all()), \
    'CONTROL FAILED: a clip landed outside the interval'
# CONTROL 3: rows without LOBT must be untouched
assert int((q[~have] != p[~have]).sum()) == 0, 'CONTROL FAILED: a LOBT-null row moved'
print('CONTROLS: unbounded slack moves 0 rows; every clip lands in the interval; LOBT-null rows untouched')

o = pd.DataFrame({'MVT_ID_mvt': tb.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': q.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT)
got = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
print(f'\nwrote {OUT}\n  sha256 {got}')
if os.environ.get('SHA79'):
    assert got == os.environ['SHA79'], f'v79 does not reproduce: {got}'
    print('  reproduces the shipped file')
