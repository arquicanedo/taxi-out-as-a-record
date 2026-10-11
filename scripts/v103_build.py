"""v103 = v100 + the EXTENDED ADS-B stage (findings s518), built on the same chain as v102 (s516) with:
time-matched coverage (ab_tm), copy-fired rows included, hedged rows (v100 >= 3600) excluded, training on
callsign-matched rows only, and three detection features (observed path, push-to-taxi time, last-15-min stop).

THE CHAIN
    scripts/adsb_pull_all.sh -> adsb_stream_day.py         adsb.lol globe_history, Jan+Jul 2025 and 2026
    ADSB_TIME_MATCH=1 scripts/adsb_push_features.py         -> push_feat_tm.parquet, archived as
                                                               artifacts/adsb_push_feat_v103.parquet (ODbL)
    scripts/adsb_price.py (v102's pricer, saves v102's window vector) then scripts/adsb_price_v103.py
                                                            -> $DC26_WORK/adsblol/cand/v103_adsb.npz
    scripts/v103_build.py                                   this file: npz -> submissions/quick-boat_v103.parquet
REPRODUCE from the repository:
    ADSB_FEAT=artifacts/adsb_push_feat_v102.parquet python scripts/adsb_price.py
    ADSB_FEAT=artifacts/adsb_push_feat_v103.parquet python scripts/adsb_price_v103.py && python scripts/v103_build.py
CONTROL: FIRE=0 writes v102 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v102.parquet'
OUT = os.environ.get('V103_OUT', f'{SUBDIR}/quick-boat_v103.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v103_FIRE0.parquet'))
SHA102 = '167d00e82a3e110b9849fdeedd9aa8abfbb51eb60f88a7c8b326b97fedcba167'
SHA103 = 'e192a2f6dd9cf49a8360e654b6b750137686d371c50e4f174993a89aac79e7df'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA102, 'v102 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v103_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v102'], inc), 'candidate priced against a different v102'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v102 {nd:,}; Q_b = sum((v103 - v102)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v102'   # BASE is asserted to be the shipped v102 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v102 byte-identically')
elif not CLEAN and SHA103 and h != SHA103 and os.environ.get('V103_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v103 does not reproduce: {h} != {SHA103} (V103_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA103:
    print('  sha256 matches the recorded v103')
