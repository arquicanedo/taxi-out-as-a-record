"""Regenerate quick-boat_v25.parquet: v23 with the classifier calibrated OUT OF FOLD.

The only change from v23 is how `Pc` is produced.

  v23   Pt = classifier's IN-SAMPLE predictions on the training rows
        Pc = Isotonic(Pt -> ext).predict(Pr)
  v25   Pt = the same classifier CROSS-FITTED over the training rows
        Pc = Isotonic(Pt_oof -> ext).predict(Pr)

`lgbm`, `perapt` and `mlp` are read from the v23 intermediates unchanged, so the
bulk is bit-identical and any score difference is the calibration and nothing else.

WHY, when CLAUDE.md records out-of-fold calibration as CLOSED at +18.5 s:

The old measurement was taken 2026-09-02 through the j3 window, which held 43
validation extremes, and it reported one number. Measured on the rebuilt window
(149 extremes) with the two channels SEPARATED, across independent classifier seed
sets:

    seedset (0,1,2)    d bulk SSE -8.907e8     d ext SSE -4.692e9
    seedset (3,4,5)    d bulk SSE -9.049e8     d ext SSE +1.640e8

The extreme channel is noise: its own BASELINE moves 4.5e9 between seed sets, more
than the effect. The bulk channel is stable to 1.6% and sits on 344,187 rows.

So the closed result was not wrong so much as unresolved: a single total was read
on a window whose extreme half is dominated by seed noise, and the stable bulk-side
signal underneath it was never separated out. This ships only the half that is
measured on rows that certainly exist in the scored set.

Board projection from the bulk channel alone: 273.8, or -4.7 s.
The extreme channel is deliberately NOT counted.
"""
import hashlib, json, os, sys, time, warnings; warnings.filterwarnings('ignore')
os.environ['DC26_DATA']  # env default
os.environ.setdefault('DC26_TAG', '_jul19')
os.environ.setdefault('DC26_RECIPE', 'v19')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import features

SHA = os.environ.get('V25_SHA', '')
OUT = os.environ.get('V25_OUT', 'submissions/quick-boat_v25.parquet')
NFOLD = int(os.environ.get('V25_NFOLD', '3'))
CAP, LOBT_THR, LOBT_P, FLOOR = 200000, 7200, 0.9, 60
W_LGB, W_MLP, W_PA = 0.2, 0.4, 0.4
EHAM_NAIVE_MIN, EHAM_PRED_MIN = 4800, 6000
S = (0, 1, 2)
T0 = time.time()

tr = pd.read_parquet(features.wk('v15_tr')); rk = pd.read_parquet(features.wk('v15_rk'))
c = json.load(open(features.wk('v15_cols', 'json'))); F, CATS = c['F'], c['CATS']
for x in CATS:
    u = pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str)) | set(rk[x].dropna().astype(str))))
    tr[x] = tr[x].astype(str).astype(u); rk[x] = rk[x].astype(str).astype(u)
ext = (tr.y >= LOBT_THR).values
print(f'train {len(tr):,}  ranking {len(rk):,}  F={len(F)}  train extremes {int(ext.sum())}', flush=True)
C = lambda s: HGC(learning_rate=.06, max_iter=400, max_leaf_nodes=63, min_samples_leaf=50,
    l2_regularization=1., categorical_features='from_dtype', early_stopping=True,
    validation_fraction=.1, n_iter_no_change=40, random_state=s)
Pr = np.mean([C(s).fit(tr[F], ext.astype(int)).predict_proba(rk[F])[:, 1] for s in S], axis=0)
print(f'ranking-side classifier [{time.time()-T0:.0f}s]', flush=True)
t1 = time.time(); Pt_oof = np.zeros(len(tr))
for ia, ib in StratifiedKFold(NFOLD, shuffle=True, random_state=0).split(tr[F], ext):
    Pt_oof[ib] = np.mean([C(s).fit(tr.iloc[ia][F], ext[ia].astype(int))
                          .predict_proba(tr.iloc[ib][F])[:, 1] for s in S], axis=0)
print(f'cross-fitted calibration set, {NFOLD} folds [{time.time()-t1:.0f}s]', flush=True)
Pc = IsotonicRegression(out_of_bounds='clip').fit(Pt_oof, ext.astype(float)).predict(Pr)

t = pd.read_parquet(features.wk('v15_trees')); n = pd.read_parquet(features.wk('v15_net'))
tpl = pq.read_table(f"{features.DATA}/submitting.parquet"); tb = tpl.to_pandas()[['MVT_ID_mvt']]
rkm = pq.read_table(features.wk('v15_rk'), columns=['MVT_ID_mvt', 'apt', 'naive']).to_pandas()
m = t.merge(n, on='MVT_ID_mvt', how='inner').merge(rkm, on='MVT_ID_mvt', how='left')
assert len(m) == len(tb), f'row count {len(m)} != submitting.parquet {len(tb)}'
pcn = pd.DataFrame({'MVT_ID_mvt': rk.MVT_ID_mvt.values, 'Pc_new': Pc})
m = m.merge(pcn, on='MVT_ID_mvt', how='left'); assert m.Pc_new.isna().sum() == 0

# Control: the bulk must be bit-identical to v23. Only Pc moves.
print(f'\nPc  shipped vs OOF:  mean {m.Pc.mean():.6f} -> {m.Pc_new.mean():.6f}   '
      f'rows >0: {int((m.Pc>0).sum())} -> {int((m.Pc_new>0).sum())}   '
      f'rows >=0.5: {int((m.Pc>=.5).sum())} -> {int((m.Pc_new>=.5).sum())}', flush=True)

sd = m.sched_delay.values; lb = m.mvt_minus_lobt.values
sdv = np.clip(sd, FLOOR, CAP); lbv = np.clip(np.where(np.isnan(lb), sd, lb), FLOOR, CAP)
hit = (lb >= LOBT_THR) & ~np.isnan(lb)
bulk = W_LGB*m.lgbm.values + W_MLP*m.mlp.values + W_PA*m.perapt.values
P = np.maximum(m.Pc_new.values, hit*LOBT_P)
pred = np.clip(P*np.where(hit, lbv, sdv) + (1-P)*bulk, FLOOR, CAP)
nv = m.naive.values
edit = (m.apt.values == 'EHAM') & (nv >= EHAM_NAIVE_MIN) & (pred > EHAM_PRED_MIN) & ~np.isnan(nv)
k = int(edit.sum())
print(f'EHAM edit fires on {k} rows (v23: 46). Re-derived, not assumed.')
assert 30 <= k <= 70, f'edit count {k} outside the sane band -- inspect before shipping'
pred = np.where(edit, nv, pred)

o = tb.merge(pd.DataFrame({'MVT_ID_mvt': m.MVT_ID_mvt.values, 'TAXITIME_SEC_mvt': np.rint(pred)}),
             on='MVT_ID_mvt', how='left')
o['TAXITIME_SEC_mvt'] = o.TAXITIME_SEC_mvt.astype('int32')
assert len(o) == len(tb) and o.TAXITIME_SEC_mvt.isna().sum() == 0
assert bool((o.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all()), 'row order must match submitting.parquet'
assert o.TAXITIME_SEC_mvt.min() >= FLOOR and o.TAXITIME_SEC_mvt.max() <= CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt', 'TAXITIME_SEC_mvt']], schema=tpl.schema,
                                    preserve_index=False), OUT)
v23 = pd.read_parquet('submissions/quick-boat_v23.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
d = o.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt - v23.reindex(o.MVT_ID_mvt)
print(f'wrote {OUT}')
print(f'  rows {len(o):,}  median {o.TAXITIME_SEC_mvt.median():.0f}  max {o.TAXITIME_SEC_mvt.max():,}  '
      f'>2h {int((o.TAXITIME_SEC_mvt>7200).sum())} (v23: 153)')
print(f'  vs v23: {int((d!=0).sum()):,} rows changed, mean {d.mean():+.1f}s, |max| {d.abs().max():,.0f}s')
print(f'  at CAP {int((o.TAXITIME_SEC_mvt>=CAP-1).sum())}  at FLOOR {int((o.TAXITIME_SEC_mvt<=FLOOR).sum())}')
h = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
print(f'  sha256 {h}   [{time.time()-T0:.0f}s total]')
if SHA:
    assert h == SHA, f'HASH MISMATCH\n  expected {SHA}\n  got      {h}'
    print('  sha256 asserted OK')
else:
    print('  (no SHA recorded yet: re-run with V25_SHA=<above> to assert reproducibility)')
