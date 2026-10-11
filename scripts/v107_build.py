"""v107 = v104 with its ADS-B stage replaced by a 50/50 blend of the LightGBM stage and an XGBoost stage (findings s536,
s537): same rows, same v104 movement features; XGBoost settings found once by FLAML and frozen.
THE CHAIN
    artifacts/adsb_push_feat_v104.parquet (ODbL) ; artifacts/v107_xgb_params.json (scripts/v107_flaml_search.py, run once)
    scripts/adsb_price_v107.py -> $DC26_WORK/adsblol/cand/v107_adsb.npz   (validated Jan|Jul instrument)
    scripts/v107_build.py      this file -> submissions/quick-boat_v107.parquet
REPRODUCE: scripts/regen_adsb.sh.  CONTROL: FIRE=0 writes v104 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v104.parquet'
OUT = os.environ.get('V107_OUT', f'{SUBDIR}/quick-boat_v107.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v107_FIRE0.parquet'))
SHA104 = 'bd4b46e26b9b2e525ced684b84ecbff56c64cf378feac71d15d31e823734f103'
SHA107 = 'cdf78063715006bdaf151e754cc64a92901deda5b514057a301cf33c6a9d9743'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA104, 'v104 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v107_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v104'], inc), 'candidate priced against a different v104'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v104 {nd:,}; Q_b = sum((v107 - v104)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v104'   # BASE is asserted to be the shipped v104 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v104 byte-identically')
elif not CLEAN and SHA107 and h != SHA107 and os.environ.get('V107_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v107 does not reproduce: {h} != {SHA107} (V107_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA107:
    print('  sha256 matches the recorded v107')
