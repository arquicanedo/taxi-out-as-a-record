"""v117 = v114 with the blend weight refitted outside LIRF (s562): v112 fitted alpha 0.662 with LIRF in, and v114 then left
LIRF at v108, so the weight outside LIRF was dragged down by rows it no longer moves. LIRF stays at v108.
THE CHAIN (after v114's, scripts/v114_build.py)
    scripts/adsb_price_v117.py     SAVE=1 -> $DC26_WORK/adsblol/cand/v117_adsb.npz   (priced against v114, rule pre-stated)
    scripts/v117_build.py          this file -> submissions/quick-boat_v117.parquet
CONTROL: FIRE=0 writes v114 back byte-identically.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v114.parquet'
OUT = os.environ.get('V117_OUT', f'{SUBDIR}/quick-boat_v117.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v117_FIRE0.parquet'))
SHA114 = '7ec3f9d66202f64766d087512cea26aa945fa08eb541b2853033fdbc84381e2b'
SHA117 = '47c5f0309c6a415617c615125c8160435cb38aa3ba918e6c4fba51b13406ee11'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA114, 'v114 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v117_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v114'], inc), 'candidate priced against a different v114'
print(f"  variant {z['variant']}")
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v114 {nd:,}; Q_b = sum((v117 - v114)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v114'   # BASE is asserted to be the shipped v114 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v114 byte-identically')
elif not CLEAN and SHA117 and h != SHA117 and os.environ.get('V117_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v117 does not reproduce: {h} != {SHA117} (V117_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA117:
    print('  sha256 matches the recorded v117')
