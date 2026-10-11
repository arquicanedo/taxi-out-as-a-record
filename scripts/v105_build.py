"""v105 = v104 + the REFINED ADS-B stage (operator, wake class, parked / first-sighting position; s523) + the hour-level
WEATHER stage on rows ADS-B cannot see (s524). Findings s527.

THE CHAIN
    ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_XY=1 scripts/adsb_push_features.py -> artifacts/adsb_push_feat_v105.parquet (ODbL)
    scripts/pull_metar.py -> $DC26_WORK/metar.parquet (IEM ASOS, public domain)
    scripts/adsb_price_v105.py -> $DC26_WORK/adsblol/cand/v105_adsb.npz
    scripts/v105_build.py      this file -> submissions/quick-boat_v105.parquet
REPRODUCE: scripts/regen_adsb.sh.  CONTROL: FIRE=0 writes v104 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
BASE = 'submissions/quick-boat_v104.parquet'
OUT = os.environ.get('V105_OUT', 'submissions/quick-boat_v105.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v105_FIRE0.parquet'))
SHA104 = 'bd4b46e26b9b2e525ced684b84ecbff56c64cf378feac71d15d31e823734f103'
SHA105 = 'cba68ec95407002d72911bec5d7e512defdf174ca51b82c87e0bb2d8af088c7d'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert sha(BASE) == SHA104, 'v104 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(f'{W}/adsblol/cand/v105_adsb.npz')
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v104'], inc), 'candidate priced against a different v104'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v104 {nd:,}; Q_b = sum((v105 - v104)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == SHA104, 'FIRE=0 must reproduce v104'
    print('  CONTROL: FIRE=0 reproduces v104 byte-identically')
elif SHA105 and h != SHA105 and os.environ.get('V105_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v105 does not reproduce: {h} != {SHA105} (V105_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif SHA105:
    print('  sha256 matches the recorded v105')
