"""v104 = v103's extended ADS-B stage plus the INBOUND-PARKING features (findings s520): for each departure track, where
the same airframe was last seen parked at the same airport in the 18 h before (its inbound arrival), the distance to the
first departure sighting, the time between, and a taxi corrected for the unseen part of a late first sighting.

THE CHAIN (v103's, s518, with the parking table)
    ADSB_TIME_MATCH=1 ADSB_PARK=1 scripts/adsb_push_features.py  -> push_feat_tm_pk.parquet, archived as
                                                                    artifacts/adsb_push_feat_v104.parquet (ODbL)
    scripts/adsb_price.py -> adsb_price_v103.py (saves v103's window vector) -> scripts/adsb_price_v104.py
                                                                 -> $DC26_WORK/adsblol/cand/v104_adsb.npz
    scripts/v104_build.py                                        this file -> submissions/quick-boat_v104.parquet
REPRODUCE from the repository: scripts/regen_adsb.sh (rebuilds v102, v103 and v104 with sha256 asserts).
CONTROL: FIRE=0 writes v103 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v103.parquet'
OUT = os.environ.get('V104_OUT', f'{SUBDIR}/quick-boat_v104.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v104_FIRE0.parquet'))
SHA103 = 'e192a2f6dd9cf49a8360e654b6b750137686d371c50e4f174993a89aac79e7df'
SHA104 = 'bd4b46e26b9b2e525ced684b84ecbff56c64cf378feac71d15d31e823734f103'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA103, 'v103 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v104_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v103'], inc), 'candidate priced against a different v103'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v103 {nd:,}; Q_b = sum((v104 - v103)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v103'   # BASE is asserted to be the shipped v103 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v103 byte-identically')
elif not CLEAN and SHA104 and h != SHA104 and os.environ.get('V104_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v104 does not reproduce: {h} != {SHA104} (V104_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA104:
    print('  sha256 matches the recorded v104')
