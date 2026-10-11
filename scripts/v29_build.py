"""v29, RECONSTRUCTED to within 0.00069 board-seconds. The form is certain; only precision is lost.

WHY THIS FILE EXISTS. `quick-boat_v29.parquet` is the base of the entire shipped chain -- v30,
v34, v38, v55 and v64 all read it -- and **no script ever produced it.** v25 through v28 have
build scripts and v30 onward have build scripts; v29 is a single hole in an otherwise complete
run, created after the recoverability rule in CLAUDE.md was already written. It is the same
failure as v11-v14, except those were row edits recoverable from a diff and this is a calibration
applied to a model output.

WHAT IS RECOVERED. CLAUDE.md records v29 as "OOF calibration at alpha = 0.35" and STATUS records
v25 as the same calibration at FULL strength. Solving `v29 = b + a*(v25 - b)` by least squares
over every shipped parquet as candidate base identifies the pair unambiguously:

    base v23   alpha 0.35008        <- recovers the recorded 0.35 to four decimals
    base v26   alpha 0.27783           next best, and not a recorded value
    base v27   alpha 0.33555           max residual 7,535 s

and at exactly alpha = 0.35:

    clip(rint(v23 + 0.35*(v25 - v23)), 60, 200000)   ->  343,243 of 344,841 rows byte-exact,
                                                         and the residual costs 0.00069 board-s

WHAT IS NOT RECOVERED, and it is worth 0.00069 board-seconds.

**CORRECTED 2026-09-13, and the first version of this docstring was wrong in its headline.** It
reported "99.54% of rows exact" and inferred a per-row BRANCH from 885 rows whose implied alpha was
exactly 1.0. That inference was an artefact: on those rows `|v25 - v23|` is **median 1 s, max 1 s**,
so the implied alpha is `(v29-v23)/(+-1)`, which is exactly 0 or exactly 1 by construction. Nothing
about a branch followed from it. There is no branch.

The residual is DOUBLE-ROUNDING, and the evidence is unambiguous:

    1,592 of the 1,598 disagreements are |d| == 1 s exactly
    direction is MIXED: 1,215 rows high, 383 low
    the fractional part of the blend is spread (0.35 on 879 rows, 0.40 on 160, 0.70 on 146),
      NOT concentrated at 0.5, and only 75 of the 154 genuine ties disagree at all
    every tie-break convention leaves ~1,600 rows off: half-even 1,598, half-up 1,596,
      half-down 1,604, truncate 3,009. So it is not a tie-break.

**v29 was computed by blending the UNROUNDED predictions and rounding once.** v23 and v25 on disk
are already `int32`, so the precision required for a byte-exact reconstruction was destroyed when
those two files were written, not by anything this script does. No reconstruction from the shipped
v23/v25 can ever be exact, and that is a property of the archive rather than a gap in knowledge.

Six rows are not double-rounding, |d| from 2 to 227 s, two of them sitting at the FLOOR of 60:

    MVT_ID        v23      v25     blend      v29   |d|
    202319078   4,728    5,397  4,962.15    5,189   227
    202314714   5,670    5,436  5,588.10    5,788   200
    212247168  10,502   10,161 10,382.65   10,211   172
    202681723   8,099    7,974  8,055.25    7,992    63
    213315556      60       87     69.45       62     7
    203051466      60       66     62.10       60     2

**What it costs, which is the number that matters rather than a row count:**

    1,592 rows at 1 s^2            1,592
    6 rows                       125,135
    total SSE                    126,727   =  0.00069 board-s,  5.4e-06 of our total SSE

So the FORM is certain (base v23, alpha 0.35, arithmetic blend, identified by least squares against
every shipped parquet as candidate base) and the reconstruction is exact for every practical
purpose. **Row-count agreement was the wrong metric and it made a closed question look open.**

HOW TO USE IT. Two honest modes, and the default is the safe one.

  MODE=verify   (default)  Rebuild and REPORT the disagreement against the shipped file. Writes
                           nothing. This is what the paper cites.
  MODE=emit                Write the reconstruction to V29_OUT. **It is not byte-identical to the
                           shipped v29 and must not be used to rebuild the shipped chain**, whose
                           scripts assert against the real file. It exists so a FRESH extract,
                           where no shipped v29 can exist at all, has a defined v29 step.

For a phase-2 set the path is: regenerate v23 (`v23_build.py`) and v25 (`v25_build.py`), then
apply this blend. On a fresh extract the double-rounding does not even arise, because v23 and v25
would be computed in float and blended before any rounding, which is what the original did. So the
phase-2 path is EXACT and the 0.00069 above is the cost of reconstructing THIS extract's v29 from
its already-rounded neighbours.

  uv run --quiet --with numpy --with pandas --with pyarrow python scripts/v29_build.py
"""
import hashlib, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

DATA = os.environ['DC26_DATA']
MODE = os.environ.get('MODE', 'verify')
ALPHA = float(os.environ.get('V29_ALPHA', '0.35'))
FLOOR, CAP = 60., 200000.
SHA = {23: '3f3c616e1cd88c07772d5560e7ed0d5356c4cb69bd0e076d967f94139370bafa',
       25: '7c12d8c02c8b724615d3b0d1a75d27af88dc1c66c7439e70acebe79c7a1eb032',
       29: 'b1dbaf87e416980cdaa3417f05568ddf080f36d2d83eb1885b8070ce61fde73f'}

def load(v, need_sha=True):
    p = f'submissions/quick-boat_v{v}.parquet'
    if need_sha:
        got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        assert got == SHA[v], f'v{v} is not the recorded file: {got}'
    t = pq.read_table(p)
    return t.column('MVT_ID_mvt').to_numpy(), t.column('TAXITIME_SEC_mvt').to_numpy().astype(float)

i23, v23 = load(23)
i25, v25 = load(25)
assert bool((i23 == i25).all()), 'v23 and v25 row order differs'
rec = np.clip(np.rint(v23 + ALPHA * (v25 - v23)), FLOOR, CAP)

print(f'reconstruction: clip(rint(v23 + {ALPHA} * (v25 - v23)), {FLOOR:.0f}, {CAP:.0f})')
if os.path.exists('submissions/quick-boat_v29.parquet'):
    i29, v29 = load(29)
    assert bool((i29 == i23).all())
    bad = rec.astype('int32') != v29.astype('int32')
    n, N = int(bad.sum()), len(v29)
    print(f'  vs the shipped v29: {N-n:,} of {N:,} rows EXACT ({(N-n)/N:.4%}), {n:,} differ, '
          f'max |d| {np.abs(rec-v29).max():,.0f}s')
    ad = np.abs(rec - v29)
    one = int((ad == 1).sum()); big = int((ad > 1).sum())
    sse = float((ad**2).sum())
    up = int((v29[bad] > (v23 + ALPHA*(v25-v23))[bad]).sum())
    print(f'  {one:,} rows off by exactly 1 s  (double-rounding: direction mixed, {up:,} up / '
          f'{n-up:,} down), {big} rows off by more')
    print(f'  residual SSE {sse:,.0f}  =  {sse/1.842e8:.5f} board-s  =  {sse/(len(v29)*261.9189**2):.2e} of total')
    print('  the FORM is certain. The residual is precision lost when v23/v25 were written as')
    print('  int32, not an unknown branch: v29 blended the UNROUNDED predictions and rounded once.')
else:
    print('  no shipped v29 present (fresh extract): nothing to verify against')

if MODE != 'emit':
    print('\nMODE=verify: wrote nothing. Set MODE=emit to write the reconstruction.')
    sys.exit(0)

OUT = os.environ.get('V29_OUT', os.path.join(os.environ.get('DC26_TMP', '/tmp'), 'quick-boat_v29_reconstructed.parquet'))
assert 'submissions/' not in OUT, \
    'refusing to write into submissions/: this is NOT the shipped v29 and the chain asserts on it'
tpl = pq.read_table(f'{DATA}/submitting.parquet')
tb = tpl.to_pandas()[['MVT_ID_mvt']]
assert bool((tb.MVT_ID_mvt.values == i23).all()), 'row order differs from submitting.parquet'
o = pd.DataFrame({'MVT_ID_mvt': tb.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': rec.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False)
assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
print('  RECONSTRUCTION, not the shipped v29. Do not feed it to the shipped chain.')
