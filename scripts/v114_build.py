"""v114 = v112 with LIRF's rows left at v108: the full-year ADS-B change is withheld at the one airport whose covered errors are
record errors the sensor cannot see (s546); the per-airport cross-fit wants alpha 0.370 there against 0.74-0.95 elsewhere (s558).
THE CHAIN (after v112's, scripts/v112_build.py)
    scripts/adsb_price_v114.py     SAVE=1 -> $DC26_WORK/adsblol/cand/v114_adsb.npz   (priced against v112, rule pre-stated)
    scripts/v114_build.py          this file -> submissions/quick-boat_v114.parquet
CONTROL: FIRE=0 writes v112 back byte-identically.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v112.parquet'
OUT = os.environ.get('V114_OUT', f'{SUBDIR}/quick-boat_v114.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v114_FIRE0.parquet'))
SHA112 = '8fe729be1d0388095eae9075d46f96d0e6f8ec2983003dc0117e8a1decb5ccc8'
SHA114 = '7ec3f9d66202f64766d087512cea26aa945fa08eb541b2853033fdbc84381e2b'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA112, 'v112 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v114_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v112'], inc), 'candidate priced against a different v112'
print(f"  variant {z['variant']}")
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v112 {nd:,}; Q_b = sum((v114 - v112)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v112'   # BASE is asserted to be the shipped v112 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v112 byte-identically')
elif not CLEAN and SHA114 and h != SHA114 and os.environ.get('V114_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v114 does not reproduce: {h} != {SHA114} (V114_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA114:
    print('  sha256 matches the recorded v114')
