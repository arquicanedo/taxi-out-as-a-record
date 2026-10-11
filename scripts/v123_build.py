"""v123 = v122 with Fiumicino's coin-flip cell re-weighted by a classifier built only from the provided data (lever L5 of
the independent agent session, scripts/fable_lirf/, s586).

THE CELL: LIRF departures with no flight-plan join, delayed 1 to 4 h (3600 <= ms < 14400), whose plan is not recoverable
from the destination arrival (v122's L2). In 2025 about half copied the schedule. v121/v122 hedge them with the mixture.
THE CLASSIFIER (scripts/fable_lirf/cf_clf.py, LightGBM, trained on 2025 with carrier rates fitted month-out): P(copy)
from the carrier's copy and DLA-filing habits (Bayes-factored P(copy | join-null)), the day's count of join-null delayed
departures and delayed fraction, carrier-day DLA discipline from join-present rows, the carrier's 2026 arrival copy rate
(labelled), a malformed-callsign flag, stand/runway/destination. No departure label of a scored month.
THE MOVE: prediction = 0.5 * (taxi_hat + P * (ms - taxi_hat)) + 0.5 * v122, taxi_hat the 2025 median real taxi by
3-hour bucket. Beta 0.5 is the midpoint, not fitted to a score.
Window (Jan|Jul 2025, after L1-L4, leave-one-month-out classifier): beta 0.5 +4.7846e8 [1.541e8, 8.923e8] on 306 rows;
beta cross-fitted between the months (0.85 / 0.25) +2.5402e8 [5.745e7, 4.926e8].
Inputs: scripts/fable_lirf/run_all.sh then l5_export.py (run in $FABLE_DIR) -> l5_2026.parquet, sha-asserted below.
CONTROL: FIRE=0 writes v122 back byte-identically.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v123_build.py
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
FD = os.environ.get('FABLE_DIR', 'artifacts/fable')   # archived; scripts/fable_lirf/run_all.sh regenerates it
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V123_BASE'] if CLEAN else 'submissions/quick-boat_v122.parquet'
OUT = os.environ.get('V123_OUT', 'submissions/quick-boat_v123.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v123_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA122 = '29cdf104e680abbeda21fc225df6e9cf4269e0c29061e2113fe5a5d7e41bf28e'
SHA_L5 = os.environ.get('SHA_L5', 'd7d45facac913a23dbad6e382fac1af607e47eba88ae8c4b2637679edf813050')      # l5_2026.parquet, recorded from the first run
SHA123 = '9f2f9f4bf8e6948d241ab53218da55b839f409179b39630c9ba51d252d0ece21'      # recorded from the first run
BETA, FLOOR, CAP = 0.5, 60.0, 200000.0
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA122, 'v122 on disk is not the shipped file'
l5f = os.path.join(FD, 'l5_2026.parquet'); h5 = sha(l5f); print(f'  l5_2026.parquet sha256 {h5}')
assert not SHA_L5 or h5 == SHA_L5, 'the classifier output changed'
L5 = pd.read_parquet(l5f).set_index('MVT_ID_mvt')
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
v121 = pq.read_table(os.environ['V121_REF'] if CLEAN else 'submissions/quick-boat_v121.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)
m = pd.Index(ids).isin(L5.index)
assert (inc[m] == v121[m]).all(), 'v122 moved a row of the cell: the levers overlap'
out = inc.copy()
if FIRE:
    mix = L5.reindex(ids[m]).mix.values
    out[m] = np.clip(np.rint(BETA * mix + (1 - BETA) * inc[m]), FLOOR, CAP)
d = out - inc; Q = float((d ** 2).sum()); N = len(ids); s0 = N * 224.7674 ** 2
print(f'  cell rows {int(m.sum())}; rows changed {int((d != 0).sum())}; Q_b {Q:.4e}; largest row share {float((d ** 2).max() / Q) if Q else 0:.1%}')
if FIRE:
    for k in (1.0, 2.0):
        print(f'  if the gain is {k:.0f}x Q: {np.sqrt((s0 - k * Q) / N):.4f}')
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA122), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v122 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA123:
    assert h == SHA123, f'v123 does not reproduce: {h}'; print('  sha256 matches the recorded v123')
