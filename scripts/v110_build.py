"""v110 = v108 with two full-year ADS-B bulk members (L55A pooled, P55A per-airport: v100's p2p_v55 recipe plus the ADS-B
pushback features, trained on all twelve months of 2025) replacing v100's bulk plus the ADS-B stage on covered rows,
blended with v108 at a cross-fitted alpha (findings s550, s551).
THE CHAIN
    scripts/adsb_pull_year.sh + adsb_pull_all.sh (streamed, nothing raw stored) -> scripts/adsb_pull_check.py --final
    scripts/adsb_push_features.py  ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_ALL_MONTHS=1 -> artifacts/adsb_push_feat_full.parquet (ODbL)
    scripts/p2p_v55_adsb.py        MODE=win TE=lf and MODE=rank TE=stored, pooled + perapt, seeds 0,1,2
    scripts/adsb_price_v110.py     the pre-registered variants; BOTH LOST on the window (s551: weights fit on all rows applied
                                   to a subset, and the ADS-B stage double-counted), archived as cand/v110_prereg_adsb.npz
    scripts/adsb_v110_diag.py      SAVE=Cfam -> $DC26_WORK/adsblol/cand/v110_adsb.npz, variant "C alpha": on covered
                                   unfired rows v108 + a (C - v108), C = v100 with the new bulk and no ADS-B stage (s551)
    scripts/v110_build.py          this file -> submissions/quick-boat_v110.parquet
The first three steps run in scripts/overnight_v110.sh (FROM=train resumes at training). CONTROL: FIRE=0 writes v108 back.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
SUBDIR = os.environ.get('DC26_SUBDIR', 'submissions'); CLEAN = os.environ.get('CLEAN') == '1'
assert not CLEAN or (os.path.realpath(SUBDIR) != os.path.realpath('submissions') and os.environ.get('DC26_CAND')), \
    'CLEAN=1 needs its own DC26_SUBDIR and DC26_CAND (scripts/regen_clean.sh)'
BASE = f'{SUBDIR}/quick-boat_v108.parquet'
OUT = os.environ.get('V110_OUT', f'{SUBDIR}/quick-boat_v110.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v110_FIRE0.parquet'))
SHA108 = '92c83411bab72a062a9fc52c51f1089c3139ce09bcc557db52de87c3b6694100'
SHA110 = '21c29e8f88ca647ae4c270c78a090e36584e4be8df80e7c96e7ab2f00facb1d5'   # recorded from the first run
FLOOR, CAP = 60., 200000.
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert CLEAN or sha(BASE) == SHA108, 'v108 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
z = np.load(os.path.join(os.environ.get('DC26_CAND', W + '/adsblol/cand'), 'v110_adsb.npz'))
assert (z['ids'] == ids).all(), 'candidate row order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); assert np.array_equal(z['v108'], inc), 'candidate priced against a different v108'
print(f"  variant {z['variant']}; members {list(z['members'])}; weights in-sample {np.round(z['w_in'], 4).tolist()}")
p = z['new'].astype(float) if FIRE else inc.copy()
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  {len(inc):,} scored rows; rows changed vs v108 {nd:,}; Q_b = sum((v110 - v108)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == sha(BASE), 'FIRE=0 must reproduce v108'   # BASE is asserted to be the shipped v108 outside CLEAN mode
    print('  CONTROL: FIRE=0 reproduces v108 byte-identically')
elif not CLEAN and SHA110 and h != SHA110 and os.environ.get('V110_ALLOW_DRIFT') != '1':
    raise AssertionError(f'v110 does not reproduce: {h} != {SHA110} (V110_ALLOW_DRIFT=1 for a from-scratch rebuild)')
elif CLEAN:
    print('  CLEAN mode: not checked against the shipped hash')
elif SHA110:
    print('  sha256 matches the recorded v110')
