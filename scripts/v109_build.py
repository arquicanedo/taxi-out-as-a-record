"""v109 = v104 with its ADS-B stage an equal-thirds blend of LightGBM, XGBoost and CatBoost (tree families only; no MLP)
on v104's rows and movement features (findings s543). No fitted weights.
THE CHAIN
    artifacts/adsb_push_feat_v104.parquet (ODbL); artifacts/v107_xgb_params.json
    scripts/adsb_price_v109.py -> $DC26_WORK/adsblol/cand/v109_adsb.npz   (validated Jan|Jul instrument)
    scripts/v109_build.py      this file -> submissions/quick-boat_v109.parquet
REPRODUCE: scripts/regen_adsb.sh.  CONTROL: FIRE=0 writes v108 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
BASE = 'submissions/quick-boat_v108.parquet'
OUT = os.environ.get('V109_OUT', 'submissions/quick-boat_v109.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v109_FIRE0.parquet'))
SHA108 = '92c83411bab72a062a9fc52c51f1089c3139ce09bcc557db52de87c3b6694100'
SHA109 = 'c94839f9e1dc0f962381f3d8038d8eb9d91de07752d40d3c5590fad5ac94dc70'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert sha(BASE) == SHA108, 'v108 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(f'{W}/adsblol/cand/v109_adsb.npz')
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v108'], inc), 'candidate priced against a different v108'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v108 {nd:,}; Q_b = sum((v109 - v108)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == SHA108, 'FIRE=0 must reproduce v108'
    print('  CONTROL: FIRE=0 reproduces v108 byte-identically')
elif SHA109 and h != SHA109 and os.environ.get('V109_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v109 does not reproduce: {h} != {SHA109} (V109_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif SHA109:
    print('  sha256 matches the recorded v109')
