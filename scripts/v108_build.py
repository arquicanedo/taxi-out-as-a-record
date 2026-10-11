"""v108 = v104 with its ADS-B stage an equal-quarters blend of LightGBM, XGBoost, CatBoost and a PyTorch MLP on v104's rows
and movement features (findings s539, s540). No fitted weights.
THE CHAIN
    artifacts/adsb_push_feat_v104.parquet (ODbL); artifacts/v107_xgb_params.json; artifacts/v108_stage_mlp.npz
    (scripts/stage_mlp_export.py -> scripts/stage_mlp.py, CPU deterministic, byte-identical on rerun)
    scripts/adsb_price_v108.py -> $DC26_WORK/adsblol/cand/v108_adsb.npz   (validated Jan|Jul instrument)
    scripts/v108_build.py      this file -> submissions/quick-boat_v108.parquet
REPRODUCE: scripts/regen_adsb.sh.  CONTROL: FIRE=0 writes v107 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v107.parquet'
OUT = os.environ.get('V108_OUT', f'{SUBDIR}/quick-boat_v108.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v108_FIRE0.parquet'))
SHA107 = 'cdf78063715006bdaf151e754cc64a92901deda5b514057a301cf33c6a9d9743'
SHA108 = '92c83411bab72a062a9fc52c51f1089c3139ce09bcc557db52de87c3b6694100'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA107, 'v107 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v108_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v107'], inc), 'candidate priced against a different v107'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v107 {nd:,}; Q_b = sum((v108 - v107)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v107'   # BASE is asserted to be the shipped v107 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v107 byte-identically')
elif not CLEAN and SHA108 and h != SHA108 and os.environ.get('V108_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v108 does not reproduce: {h} != {SHA108} (V108_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA108:
    print('  sha256 matches the recorded v108')
