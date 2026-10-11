"""v100 = v99 with the bulk blend refitted as the s492 P package: leak-free window versions of the shipped members
plus s489's new members, simplex weights, applied additively to v99's unfired rows the way v99 shipped.

THE CHANGE. v99's bulk on its unfired rows is .2230 pooled + .2030 per-airport + .0880 K32 + .4860 K128 (+ SHIFT,
+ v99's own colsample delta). Those weights were fitted against window members that are not the ship (s487).
Refitted on LEAK-FREE window replicas of all four shipped members (ten non-window months, ten-month te_*) together
with the new members, non-negative and summing to 1, SHIFT kept as shipped (scripts/p2p_price.py):

    L55  .0494   P55  .0240   K32  .0000   K128 .2703          (the four shipped members)
    ID   .1562   identity categoricals member                   (scripts/p2p_arm.py ARM=idcat)
    RQ   .1131   same-runway neighbours' naive member           (scripts/p2p_arm.py ARM=rwyq)
    CL   .2770   clean-room bulk, f3 features                   (scripts/p2p_clean.py)
    L55o .0667   P55o .0431   the v55 trees on out-of-fold te_* (scripts/p2p_v55.py TE=oof)

    v100 = v99                                   where |v99 - clip(MVT - SCHED, 60, 200000)| <= 60   (copy-fired)
    v100 = clip(rint(v99 + M w_new - M0 w0))     elsewhere (313,332 rows)

Ranking-side members: $DC26_WORK/p2_members_rk.parquet (scripts/p2p_save_members.py).

PREDICTED BOARD GAIN, stated before any upload. s488's estimator on the leak-free base, the pre-registered January|July
cross-fit of the FULL package (s492 P): G_hat = -2T_w N_b/N_w - Q_b = +3.4646e+08 SSE, SE 1.0307e+08 (day-cluster,
window only), G/SE +3.36 -> the ship rule (G_hat >= 2 SE) PASSES.
    predicted board 254.9175 from 256.8806 (+1.9631);  +-1 SE: 255.5031 .. 254.3306
    Q_b (exact, paid in full) 6.6279e+08, 33.0% of it on January 2-7 2026 (8.5% of rows), 62.6% on the top 1% of rows.
KILL (s492): a negative board delta kills the package. A board score above 256.5703 (delta below +0.3103, i.e. more
than 2*sqrt(2) SE below G_hat) means the instrument failed. Recorded either way.
CONTROL: FIRE=0 reproduces v99 byte-identically.
"""
import hashlib, os, tempfile
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
FIRE = os.environ.get('FIRE', '1') == '1'
BASE = 'submissions/quick-boat_v99.parquet'
# FIRE=0 never writes over the candidate: its default output is a temp file
OUT = os.environ.get('V100_OUT', 'submissions/quick-boat_v100.parquet' if FIRE else os.path.join(tempfile.gettempdir(), 'quick-boat_v100_FIRE0.parquet'))
SHA99 = '3f6db0be605edf9dd84932f3d57ceb7f45756f4cb17040f18516d6cae2cb4f26'
SHA100 = '30ca298a56825a78762312936b39229f32c1fdc7dab2659ecd7f69957f939332'   # recorded from the first run
FLOOR, CAP = 60., 200000.
MEMBERS = ['L55', 'P55', 'K32', 'K128', 'ID', 'RQ', 'CL', 'L55o', 'P55o']
W_NEW = np.array([0.04940166675079049, 0.02403578920803634, 0.0, 0.27031709271031573, 0.15624409872162232,
                  0.11314475019151718, 0.2770489040304774, 0.06672151150248899, 0.0430861868847515])
SHIP = ['L55', 'P55', 'K32', 'K128']; W0 = np.array([.2230, .2030, .0880, .4860])
assert abs(W_NEW.sum() - 1.0) < 1e-12 and (W_NEW >= 0).all()
sha = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
assert sha(BASE) == SHA99, 'v99 on disk is not the shipped file'
tpl = pq.read_table(f'{D}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(BASE).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base row order'
M = pd.read_parquet(os.environ.get('V100_MEMBERS', f'{W}/p2_members_rk.parquet')).set_index('MVT_ID_mvt').reindex(ids)
assert M[MEMBERS].notna().all().all(), 'member join incomplete'
# V100_MEMBERS0: the four shipped members as v99's bulk holds them (default: the same file). Re-seeded members go in
# V100_MEMBERS while v99 keeps the old ones, so the delta must remove the old four-member term, not the new (s564).
M0 = pd.read_parquet(os.environ['V100_MEMBERS0']).set_index('MVT_ID_mvt').reindex(ids) if os.environ.get('V100_MEMBERS0') else M
Mr = np.ascontiguousarray(M[MEMBERS].to_numpy(float)); M0r = np.ascontiguousarray(M0[SHIP].to_numpy(float))
delta = Mr @ W_NEW - M0r @ W0
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(ids)
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values
ms = np.clip(sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt), FLOOR, CAP)
inc = b.TAXITIME_SEC_mvt.values.astype(float); app = np.abs(inc - ms) > 60
print(f'  {len(inc):,} scored rows   copy-fired (|v99 - ms| <= 60) {int((~app).sum()):,}   delta applied to {int(app.sum()):,}')
print(f'  delta on applied rows: median {np.median(delta[app]):+.2f} s  RMS {np.sqrt(np.mean(delta[app]**2)):.2f} s  '
      f'q01 {np.quantile(delta[app], .01):+.0f}  q99 {np.quantile(delta[app], .99):+.0f}')
p = inc.copy()
if FIRE: p[app] = np.clip(np.rint(inc[app] + delta[app]), FLOOR, CAP)
nd = int((p != inc).sum()); Q = float(((p - inc) ** 2).sum())
print(f'  rows changed {nd:,}   Q_b = sum((v100 - v99)^2) = {Q:.4e}')
assert (nd > 0) == FIRE
o = pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': p.astype('int32')})
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t, OUT); h = sha(OUT); print(f'  wrote {OUT}  sha256 {h}')
if not FIRE:
    assert h == SHA99, 'FIRE=0 must reproduce v99'
    print('  CONTROL: FIRE=0 reproduces v99 byte-identically')
else:
    if h == SHA100:
        print('  sha256 matches the recorded v100')
    elif os.environ.get('V100_ALLOW_DRIFT') == '1':
        # from-scratch regeneration: members re-drawn, so compare numerically against the shipped file
        s100 = pq.read_table('submissions/quick-boat_v100.parquet').to_pandas().TAXITIME_SEC_mvt.values.astype(float)
        dd = o.TAXITIME_SEC_mvt.values.astype(float) - s100
        print(f'  DRIFT ALLOWED: {int((dd != 0).sum()):,} rows differ from the shipped v100, RMS {np.sqrt(np.mean(dd ** 2)):.3f} s, '
              f'sum d^2 {float((dd ** 2).sum()):.4e}')
    else:
        raise AssertionError(f'HASH MISMATCH {h} (recorded {SHA100}); set V100_ALLOW_DRIFT=1 for a from-scratch regeneration')
