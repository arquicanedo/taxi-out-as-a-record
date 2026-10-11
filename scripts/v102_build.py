"""v102 = v100 + an ADS-B pushback stage: a LightGBM on v100's residual, fed by pushback features detected from raw
adsb.lol surface tracks (open data, ODbL 1.0; attribution in scripts/adsb/ATTRIBUTION.md). Findings s516.

THE CHAIN (each step regenerates the next from public inputs):
    scripts/adsb_pull_all.sh            adsb.lol globe_history, Jan+Jul 2025 and 2026 (+ the four previous days of each
                                        month's 1st), streamed; per day, surface points / take-offs / landings
    scripts/adsb_push_features.py       per departure: callsign + take-off match, pushback detection, 11 features
    scripts/adsb_price.py               v100 rebuilt on the window (reproduces p2p_price FULL__in exactly); the stage
                                        trained on the window's covered ordinary rule rows, 3 seeds; applied to 2026
                                        departures with a pushback where |v100 - clip(MVT - SCHED)| > 60; writes
                                        $DC26_WORK/adsblol/cand/v102_adsb.npz
    scripts/v102_build.py               this file: npz -> submissions/quick-boat_v102.parquet, sha256 asserted

REPRODUCE from the repository without the 389 GB pull: the feature table v102 was priced from is archived as
artifacts/adsb_push_feat_v102.parquet (ODbL, scripts/adsb/ATTRIBUTION.md), so
    ADSB_FEAT=artifacts/adsb_push_feat_v102.parquet python scripts/adsb_price.py && python scripts/v102_build.py
CONTROL: FIRE=0 writes v100 back byte-identically from the same code path.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v100.parquet'
OUT = os.environ.get('V102_OUT', f'{SUBDIR}/quick-boat_v102.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v102_FIRE0.parquet'))
SHA100 = '30ca298a56825a78762312936b39229f32c1fdc7dab2659ecd7f69957f939332'
SHA102 = '167d00e82a3e110b9849fdeedd9aa8abfbb51eb60f88a7c8b326b97fedcba167'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA100, 'v100 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v102_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v100'], inc), 'candidate built on a different v100'
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; ADS-B stage rows {int(z["user"].sum()):,}; rows changed {nd:,}; Q_b = sum((v102 - v100)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v100'   # BASE is asserted to be the shipped v100 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v100 byte-identically')
elif not CLEAN and SHA102 and h != SHA102 and os.environ.get('V102_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v102 does not reproduce: {h} != {SHA102} (V102_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA102:
    print('  sha256 matches the recorded v102')
