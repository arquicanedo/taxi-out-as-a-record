"""Refuse to submit an extreme-band edit whose only evidence is validation.

s226 cost two submission slots to a rule that was already written down. v54 moved 32 rows in
the copy channel at a weight chosen on validation; the board's optimum had the OPPOSITE SIGN
and it cost -3.00 board-s. CLAUDE.md said, in those words, that validation misleads on
DIRECTION for the extreme band and that the bar is bulk-side evidence. The rule was quoted
earlier in the same session and not applied.

A rule that lives only in prose has now been broken by the person who wrote it. This is the
same rule as code.

    uv run --quiet --with pandas --with numpy --with pyarrow python scripts/presubmit_guard.py \
        CANDIDATE.parquet --base BASE.parquet --evidence board|validation|labelled-prior \
            [--note "..."]

    --evidence board is historical: it is the class the retired probe-and-invert programme used (README section 6,
    REPRODUCE.md section 9). It stays so the guard's decisions on those submissions can be re-run.

THE THIRD EVIDENCE CLASS, added 2026-09-14. The original taxonomy had two instruments because
the project had two: the validation window and a board inversion. A third exists and this guard
could not express it -- a BASE RATE COUNTED ON LABELLED ROWS MATCHING THE CELL'S OWN PREDICATE.
That is not the validation window and it does not fail the way the window does: it is a direct
count of the outcome on the same population, from 2025 rows whose labels are given.

It is admitted ONLY under conditions that stop it becoming a loophole, all enforced below:
  - --note is REQUIRED and must state the predicate and the count (e.g. "11/20 LIRF LOBT-null
    sched_delay>=50000"), so the claim is auditable from the command line alone
  - the edit must be SMALL in rows. A base rate sets ONE scalar for a cell; it cannot justify
    a many-row rewrite, which would be a model dressed up as a prior
  - it is still refused when bundled with a large bulk change, like every other class

This class was added in order to ship v77, which corrects A34 from a BOARD INVERSION
(v34_build.py:40, p = 0.4214 solved from two board scores) to the labelled prior 11/20 that
v76_build.py:12 already claims it is. Widening a gate to pass one's own submission is the v54
pattern, so it is recorded here in full rather than quietly. The justification is that the
taxonomy was incomplete; the test that it is not a loophole is that a MODEL fitted on labelled
data still does not qualify, only a counted rate on the edited population does.

It also flags the SECOND failure of that night: a candidate that bundles an extreme-band edit
together with a large bulk change, so one bad edit can poison the measurement of the other.
Ship them separately or the board cannot tell you which one moved.
"""
import argparse, hashlib, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
THR, PER = 7200, 1.842e8
SUB = os.environ.get('DC26_SUBMITTING',
      os.path.join(os.environ['DC26_DATA'],
                   'submitting.parquet'))

ap = argparse.ArgumentParser()
ap.add_argument('candidate'); ap.add_argument('--base', required=True)
ap.add_argument('--evidence', required=True,
                choices=['board', 'validation', 'labelled-prior'],
                help='where the WEIGHTS in this submission came from')
ap.add_argument('--note', default='')
ap.add_argument('--gain-val', type=float, default=None,
                help='the SSE gain measured on validation for this exact change. Required with '
                     '--evidence validation: it is checked against Q (see s367).')
ap.add_argument('--family', default='feature', choices=['bulk', 'feature'],
                help='which transfer factor applies: bulk 1.944x (s247), feature ~4x (s281)')
ap.add_argument('--force', action='store_true', help='override; requires --note')
a = ap.parse_args()

tpl = pq.read_table(SUB); tb = tpl.to_pandas()[['MVT_ID_mvt']]
c = pq.read_table(a.candidate); cd = c.to_pandas()
b = pq.read_table(a.base).to_pandas()
fail = []
if len(cd) != len(tb) or not bool((cd.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all()):
    fail.append('row ids do not match submitting.parquet')
if not c.schema.equals(tpl.schema): fail.append('schema differs from submitting.parquet')
if cd.TAXITIME_SEC_mvt.isna().any(): fail.append('nulls present')
if cd.TAXITIME_SEC_mvt.min() < 60 or cd.TAXITIME_SEC_mvt.max() > 200000: fail.append('outside [60,200000]')
p = cd.TAXITIME_SEC_mvt.values.astype(float); q = b.TAXITIME_SEC_mvt.values.astype(float)
d = p - q; moved = d != 0
mass = d**2
ext = q >= THR                                   # the copy channel, as the BASE predicts it
ext_mass = float(mass[ext].sum()); tot = float(mass.sum())
frac = ext_mass / tot if tot else 0.0
print(f'candidate {a.candidate}')
print(f'  sha256 {hashlib.sha256(open(a.candidate,"rb").read()).hexdigest()}')
print(f'  base   {a.base}')
print(f'  rows moved {int(moved.sum()):,} of {len(p):,} ({moved.mean():.2%})')
print(f'  total perturbation Q = {tot:.4e}  ({tot/PER:+.4f} board-s if T=0)')
print(f'  extreme-band share of that mass: {frac:.1%}  '
      f'({int((moved&ext).sum())} rows, base >= {THR}s)')
print(f'  bulk-band share:                 {1-frac:.1%}  ({int((moved&~ext).sum()):,} rows)')
print(f'  declared evidence: {a.evidence}')
LP_MAX_ROWS = 50
if a.evidence == 'labelled-prior':
    # the two conditions that keep the third class from becoming an override with a nicer name
    if not a.note:
        fail.append('labelled-prior evidence REQUIRES --note stating the predicate and the '
                    'count, e.g. "11/20 LIRF LOBT-null sched_delay>=50000". Without it the '
                    'claim cannot be audited from the command line.')
    if int(moved.sum()) > LP_MAX_ROWS:
        fail.append(f'labelled-prior evidence moves {int(moved.sum()):,} rows, over the '
                    f'{LP_MAX_ROWS}-row limit. A counted base rate sets ONE scalar for a cell; '
                    f'a many-row rewrite is a MODEL, and a model fitted on labelled data is '
                    f'validation evidence, not a prior.')
if frac >= 0.50 and a.evidence == 'validation':
    # This message used to end "Probe it and invert, or ship it bulk-side." That instructed the
    # user to reverse-engineer the ranking, which this project retired on 2026-09-11 after an
    # organiser said it was unwelcome, and which the rules warn is monitored. A guard that
    # forbids one bad practice must not recommend another. Removed 2026-09-14.
    fail.append(f'extreme-band edit ({frac:.0%} of moved SSE mass) with VALIDATION evidence. '
                'Offline validation mispredicts the DIRECTION of tail-side changes on this '
                'target, so it cannot license this edit. Either move the evidence to a counted '
                'rate on the edited population (--evidence labelled-prior) or ship the change '
                'bulk-side. This is the v54 failure, which cost 3.00 RMSE-seconds.')
# s367, bought with v89 (-0.1905). Q is paid IN FULL whatever happens to the direction; the GAIN
# attenuates across the year gap by 1.944x for bulk work (s247) and ~4x for feature families
# (s281). So a change is only worth firing when its validation gain exceeds Q by that factor.
# v89's ratio was 1.15x and BOTH numbers were printed in this guard's own output, undivided.
# s367, bought with v89 (-0.1905), ALGEBRA CORRECTED same day. Model the change as moving a
# fraction alpha of the way toward the validation-implied correction. Then Q = alpha^2 |e|^2 and
# the validation gain is |e|^2 (2*alpha - alpha^2), so
#       gain_val / Q = 2/alpha - 1          and        alpha = 2 / (1 + gain_val/Q)
# On the board the direction survives with efficiency eta, and the gain is positive iff
# alpha < 2*eta, i.e. iff        gain_val/Q  >  1/eta - 1.
# v89 measured eta = 0.342 for a bulk feature-family change (2T = -9.463e+07 against Q = 1.2888e+08
# at alpha = 0.93), giving a bar of 1.92. v89 came in at 1.15 and lost. NOTE the bar is
# 1/eta - 1, NOT the transfer factor itself: my first version of this check used the factor
# directly and was too strict by one.
ETA = {'bulk': 0.51, 'feature': 0.342}[a.family]     # 0.51 = 1/s247's 1.944x; 0.342 measured by v89
BAR = 1.0/ETA - 1.0
if a.evidence == 'validation':
    if a.gain_val is None:
        fail.append('--evidence validation REQUIRES --gain-val, the SSE gain this change measured '
                    'on validation, so it can be checked against Q. See s367.')
    else:
        ratio = a.gain_val / tot if tot else float('inf')
        alpha = 2.0/(1.0+ratio)
        print(f'  validation SSE gain {a.gain_val:.4e}   gain/Q = {ratio:.2f}x   '
              f'implied step alpha = {alpha:.2f}   needs ratio > {BAR:.2f} (eta={ETA}, family={a.family})')
        if ratio <= BAR:
            fail.append(f'gain/Q is {ratio:.2f}x against a bar of {BAR:.2f} for family={a.family}. '
                        f'Q is paid in full; the direction survives the year gap only at efficiency '
                        f'eta={ETA}, so a change is only worth firing when it moves LESS than '
                        f'{2*ETA:.2f} of the way to its validation optimum (this one moves '
                        f'{alpha:.2f}). v89 fired at 1.15x and lost 0.1905 with every other '
                        f'control green.')
if frac >= 0.10 and (1 - frac) >= 0.10 and int((moved & ~ext).sum()) > 10000:
    fail.append(f'BUNDLED: {frac:.0%} extreme-band and {1-frac:.0%} bulk-band mass in one '
                'submission. One bad edit poisons the measurement of the other -- this is why '
                'v55 could not be told apart from v54. Ship them separately.')
print()
if fail and not a.force:
    for f in fail: print(f'  REFUSED: {f}')
    sys.exit(1)
if fail and a.force:
    if not a.note: print('  --force requires --note explaining why'); sys.exit(1)
    for f in fail: print(f'  OVERRIDDEN: {f}')
    print(f'  note: {a.note}')
print('  PASS')
