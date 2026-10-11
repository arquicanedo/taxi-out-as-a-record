"""v112 = v108 + a (C2 - v108) on v110's covered unfired rows, where C2 is v100 with its bulk swapped for the full-year ADS-B
members (B2 = 0.7419 L55A + 0.2581 P55A) PLUS a full-year ADS-B stage (LightGBM + XGBoost halves) trained on month-fold
out-of-fold residuals of B2 (findings s554). Replaces v110's C alpha, whose remaining 38% of v108 carried a stage trained on one
month per fold.
THE CHAIN (after v110's, scripts/v110_build.py)
    scripts/oof_folds_v112.sh      p2p_v55_adsb.py MODE=oof, NEST=win (5 folds) and NEST=all (6 folds), seed 0
    scripts/adsb_stage_full.py     SAVE=1 -> $DC26_WORK/adsblol/cand/v112_adsb.npz   (priced against v110, rule pre-stated)
    scripts/v112_build.py          this file -> submissions/quick-boat_v112.parquet
CONTROL: FIRE=0 writes v110 back byte-identically.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v110.parquet'
OUT = os.environ.get('V112_OUT', f'{SUBDIR}/quick-boat_v112.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v112_FIRE0.parquet'))
SHA110 = '21c29e8f88ca647ae4c270c78a090e36584e4be8df80e7c96e7ab2f00facb1d5'
SHA112 = '8fe729be1d0388095eae9075d46f96d0e6f8ec2983003dc0117e8a1decb5ccc8'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA110, 'v110 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v112_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v110'], inc), 'candidate priced against a different v110'
print(f"  variant {z['variant']}; base weights L55A/P55A {np.round(z['wLP'], 4).tolist()}")
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v110 {nd:,}; Q_b = sum((v112 - v110)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v110'   # BASE is asserted to be the shipped v110 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v110 byte-identically')
elif not CLEAN and SHA112 and h != SHA112 and os.environ.get('V112_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v112 does not reproduce: {h} != {SHA112} (V112_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA112:
    print('  sha256 matches the recorded v112')
