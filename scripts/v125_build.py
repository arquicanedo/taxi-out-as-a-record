"""v125 = v124 with Fiumicino's local-time record class (lever L3 of the independent agent session, scripts/fable_lirf/,
s586, s592).

THE CELL: LIRF departures with no flight-plan join and ms < 1800 whose plan is not recoverable from the destination
arrival. In 2025 (76 rows) their records were real taxis, schedule copies, or a fourth class: the real off-block written
in LOCAL time, so y = taxi + the UTC offset (3,600 s in January, 7,200 s in July); 20 of 76, 17 of 21 for BEL, AEA, AEZ,
AEE, LZB and ETH. No local-time error appears on any arrival in either year, so it is not readable from 2026.
THE MOVE: the class mixture p_real * 1000 + p_tz * (1000 + offset) + p_copy * clip(ms), class rates counted on the 2025
cell per group (scripts/fable_lirf/l3_export.py). Window (the agent's, class rates cross-fitted by month): +1.6398e8
[3.935e7, 3.324e8] on 13 rows, mostly extremes.
Inputs: scripts/fable_lirf/run_all.sh then l3_export.py (run in $FABLE_DIR) -> l3_2026.parquet, sha-asserted below.
CONTROL: FIRE=0 writes v124 back byte-identically.
  uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v125_build.py
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']
FD = os.environ.get('FABLE_DIR', 'artifacts/fable')   # archived; scripts/fable_lirf/run_all.sh regenerates it
FIRE = os.environ.get('FIRE', '1') == '1'
CLEAN = os.environ.get('CLEAN') == '1'   # v127: the same rule on another base (scripts/v127_build.sh)
BASE = os.environ['V125_BASE'] if CLEAN else 'submissions/quick-boat_v124.parquet'
OUT = os.environ.get('V125_OUT', 'submissions/quick-boat_v125.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v125_FIRE0.parquet'))
assert not CLEAN or not os.path.realpath(OUT).startswith(os.path.realpath('submissions')), 'CLEAN=1 writes outside submissions/'
SHA124 = 'a99fef2878dab2d2c457bf96f1a597cc8f6705c4bde35ddeae7ddb45f1ad70fd'
SHA_L3 = os.environ.get('SHA_L3', '955acde3d8038de50dac25291f8c9afb7afa74172a632c6dc1ad6c21f4f0bc44')      # l3_2026.parquet, recorded from the first run
SHA125 = '64eb3556d3fa7c93c8b7f8b891b6f71bc800021e681fc01901cd841cd8ab9612'      # recorded from the first run
FLOOR, CAP = 60.0, 200000.0
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA124, 'v124 on disk is not the shipped file'
l3f = os.path.join(FD, 'l3_2026.parquet'); h3 = sha(l3f); print(f'  l3_2026.parquet sha256 {h3}')
assert not SHA_L3 or h3 == SHA_L3, 'the class mixture changed'
L3 = pd.read_parquet(l3f).set_index('MVT_ID_mvt')
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(); inc = b.TAXITIME_SEC_mvt.values.astype(float)
v121 = pq.read_table(os.environ['V121_REF'] if CLEAN else 'submissions/quick-boat_v121.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)
m = pd.Index(ids).isin(L3.index)
assert (inc[m] == v121[m]).all(), 'an earlier lever moved a row of the cell: the levers overlap'
out = inc.copy()
if FIRE:
    out[m] = np.clip(np.rint(L3.reindex(ids[m]).l3.values), FLOOR, CAP)
d = out - inc; Q = float((d ** 2).sum()); N = len(ids); s0 = N * 222.9558 ** 2
print(f'  cell rows {int(m.sum())}; rows changed {int((d != 0).sum())}; Q_b {Q:.4e}; largest row share {float((d ** 2).max() / Q) if Q else 0:.1%}')
if FIRE:
    for k in (0.5, 1.0):
        print(f'  if the gain is {k:.1f}x Q: {np.sqrt((s0 - k * Q) / N):.4f}')
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': out.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == (sha(BASE) if CLEAN else SHA124), 'FIRE=0 must reproduce its base'; print('  CONTROL: FIRE=0 reproduces v124 byte-identically')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA125:
    assert h == SHA125, f'v125 does not reproduce: {h}'; print('  sha256 matches the recorded v125')
