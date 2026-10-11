"""v77 = v74 with the v34 rollover hedge moved from the BOARD INVERSION to the LABELLED PRIOR.

    A34   0.40  ->  0.55

WHY THIS IS A DISTINCT HYPOTHESIS AND NOT A VARIANT. `v76_build.py` line 12 lists A34 among the
quantities that survive its strip because they come from labelled data:

    A34 = 0.40      the v34 rollover hedge      labelled 2025 prior, 11/20 rows

**That attribution is false.** `v34_build.py:40` records where 0.40 came from:

    sum D^2 = 8.2585e9    observed dSSE = -3.2734e8  (from the two board scores)
    gain = sum(D^2) * P * (2p - P)   =>   p = 0.4214          then shrunk to 0.40 for margin

That is the probe-and-invert programme, retired 2026-09-11. And 11/20 is **0.55**, not 0.40, so
v76 -- the entry whose whole premise is "every board-fitted weight solved from one realisation is
gone" -- kept one, mislabelled as labelled-data content. The disclosure of 2.7501 board-s
therefore understates the board-fitted content, because A34 sat in BOTH arms of that measurement.

v77 is the correction: the hedge at the rate the labelled data actually supports.

THE EVIDENCE, all labelled 2025, no board input (2026-09-14):

    ADEP=LIRF & LOBT_flt null & sched_delay >= 50,000       n=20   11 rollover  9 fallback
    p = 0.5500 +- 0.1112

    leave-one-out log-likelihood on those 20 rows      higher is better
      constant 0.55 (pooled)      -13.7628     <- best
      constant 0.40 (shipped)     -14.6766
      LOO sched_delay buckets     -14.7712     <- worse than a constant. Per-row hedging is DEAD.

The per-row route was tested first and killed. In sample p rises with sched_delay
(0.333 / 0.667 / 0.800 over three buckets, rank corr +0.3399, permutation one-sided p 0.0698),
and leave-one-out does not support it. That is the fifth hand-built LIRF rule to die and it
belongs with the four in CLAUDE.md.

WHAT IT IS WORTH. E[SSE] = sum(D^2) * [p(A-1)^2 + (1-p)A^2], minimised at A = p. Over the six
scored rows, sum(D^2) = 8.1889e9 and one board-second is 1.806e8 SSE:

    truth ->            p=0.55    p=0.50    p=0.45    p=0.40
    A=0.40 (v74)        12.240    11.790    11.330    10.880     board-s lost vs a perfect oracle
    A=0.55 (v77)        11.220    11.330    11.670    11.900
    gain                +1.020    +0.460    -0.340    -1.020

**Symmetric bet, about +-1 board-s**: 261.9189 - 1.020 is 260.90. It is worth a slot because the board takes
best-per-team so a loss cannot cost the standing, and because the ANSWER is what the paper needs:
it is the only test available of whether a board inversion or the labelled prior is the better
estimator of this cell, and that question is the paper's central claim.

THE COUNTER-EVIDENCE, stated because it is real. v75 fired A34=1.00 and scored 276.7902 where the
two-point model at p=0.5455 predicted 270.1929, a 6.60 board-s overshoot implying fewer than
0.5455 of the six are rollovers (s272). That is a BOARD observation. Using it to set A34 is the
retired programme -- and so is keeping 0.40, which IS a board-set value. The consistent position
is to use the labelled data, which says 0.55.

ARITHMETIC. A34 touches only the six V34 rows; v64_build.py masks C49/C51/V54/EX with ~V34, so
nothing else moves. With B = bulk + 86,400 and P = p29, v75 = B and v74 = 0.40B + 0.60P, so

    v_A = v74 + ((A - 0.40)/0.60) * (v75 - v74)       and at A=0.55 that is 0.75*v74 + 0.25*v75

Controls below: A=0.40 must reproduce v74 byte-identically and A=1.00 must reproduce v75.
"""
import hashlib, os, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DATA = os.environ['DC26_DATA']
SUB  = os.environ.get('V77_SUB', 'submissions')
OUT  = os.environ.get('V77_OUT', f'{SUB}/quick-boat_v77.parquet')
A34_OLD = 0.40
A34 = float(os.environ.get('A34', '0.55'))       # the labelled prior, 11/20

tpl = pq.read_table(f'{DATA}/submitting.parquet'); tb = tpl.to_pandas()[['MVT_ID_mvt']]
def rd(f):
    t = pq.read_table(f).to_pandas()
    assert bool((t.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all()), f
    return t.TAXITIME_SEC_mvt.values.astype(float)

v74, v75 = rd(f'{SUB}/quick-boat_v74.parquet'), rd(f'{SUB}/quick-boat_v75.parquet')
cell = v74 != v75
n = int(cell.sum())
assert n == 6, f'the v34 cell is {n} rows, expected 6 -- A34 must not be touching anything else'
D = (v75 - v74) / (1.0 - A34_OLD)                 # the FALL -> ROLL gap, exact in the two parquets
print(f'v34 cell: {n} rows   sum D^2 = {float((D[cell]**2).sum()):.4e}')

def build(a):
    return np.clip(np.rint(v74 + ((a - A34_OLD) / (1.0 - A34_OLD)) * (v75 - v74)), 60., 200000.)

for a, ref, nm in ((A34_OLD, v74, 'v74'), (1.00, v75, 'v75')):
    nd = int((build(a).astype('int32') != ref.astype('int32')).sum())
    print(f'CONTROL A34={a:.2f} vs {nm}: {nd} rows differ'); assert nd == 0, f'CONTROL FAILED vs {nm}'

p = build(A34)
d = p - v74
print(f'\nA34 {A34_OLD:.2f} -> {A34:.2f}:  {int((d != 0).sum())} rows move, '
      f'sum|d| {np.abs(d).sum():,.0f}s, Q = {float((d**2).sum()):.4e}')
N = len(p); B74 = 261.9189
S = float((D[cell]**2).sum())
print('\n  expected board-s lost vs a perfect oracle:')
print('    truth ->' + ''.join(f'   p={q:.2f}' for q in (0.55, 0.50, 0.45, 0.40)))
for a, nm in ((A34_OLD, 'A=0.40 v74'), (A34, f'A={A34:.2f} v77')):
    print(f'    {nm:11s}' + ''.join(
        f'{S*(q*(a-1)**2+(1-q)*a**2)/(2*N*B74):>8.3f}' for q in (0.55, 0.50, 0.45, 0.40)))

o = pd.DataFrame({'MVT_ID_mvt': tb.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': p.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT)
got = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
print(f'\nwrote {OUT}\n  sha256 {got}')
SHA77 = os.environ.get('SHA77', '')
if SHA77:
    assert got == SHA77, f'v77 does not reproduce: {got} != {SHA77}'
    print('  reproduces the shipped file')
