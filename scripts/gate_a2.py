"""Stage A2 (CPU, LightGBM + sklearn): the SHIPPED blend's three bulk components,
cross-fitted, so the gate can be trained against the real thing.

s103 built the gate against a single cross-fitted LightGBM at 236.21 and flagged
that the shipped bulk is `0.2 LGB + 0.4 MLP + 0.4 per-airport`. Reading the build
scripts finds THREE divergences from shipped, not one:

  1. third expert is one pooled LGB, not the 0.2/0.4/0.4 blend   (s103 flagged)
  2. DC26_RECIPE defaulted to v18; the shipped v23 is v19 (+k_q15)
  3. the pooled model shipped at DC26_POOLED_CAP=2000,255,20, not 600x127x100

The classifier does NOT diverge: v15_trees.py aggregates 3 HGC seeds with an
arithmetic mean and isotonic-calibrates, which is exactly what gate_a.py did. So
the SHIPPED row's extreme column in s103 was already faithful, and the -4,324 s
extreme win was measured against a real baseline.

Emits each component separately so stage B can form any blend without refitting.

Seeds: 3 on the validation side (matching shipped), 1 on the train-side 2-fold
cross-fit, whose only consumer is gate training. The bulk seed floor is 0.144 s
(CLAUDE.md), so the asymmetry is far below anything that could move the routing.
"""
import os
os.environ.setdefault('DC26_RECIPE', 'v19')          # the shipped v23 rung
import pandas as pd, numpy as np, json, sys, time, warnings; warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, os.environ.get('DC26_SCRIPTS', os.path.dirname(os.path.abspath(__file__))))
import features, v17_features

# DC26_WORK / DC26_SCRIPTS added 2026-09-13 so this runs on a second machine; the paths change nothing in the
# build (clean room 2026-09-23: gate_stage_a2.npz rebuilt IDENTICALLY from an empty work folder).
SP = os.environ['DC26_WORK']
THR = 7200; CAP, FLOOR = 200000, 60
# DC26_GATE_OUT added 2026-09-13 (mac-01 Q1). gate_stage_a2.npz is the reference every recorded
# number on BOTH machines was measured against, so a rebuild that has not yet proven itself must
# not be able to overwrite it. Default is unchanged, so the Mac's entry point is byte-identical.
OUT = os.environ.get('DC26_GATE_OUT') or os.path.join(SP, 'gate_stage_a2.npz')
S3 = (0, 1, 2)
POOLED = dict(learning_rate=.06, n_estimators=2000, num_leaves=255, min_child_samples=20,
              reg_lambda=1., n_jobs=-1, verbose=-1)          # DC26_POOLED_CAP of v23
PERAPT = dict(learning_rate=.06, n_estimators=600, num_leaves=127, min_child_samples=100,
              reg_lambda=1., n_jobs=-1, verbose=-1)          # unchanged since v6
T0 = time.time()

tr = pd.read_parquet(f'{SP}/night_tr_all.parquet'); va = pd.read_parquet(f'{SP}/night_va_all.parquet')
c = json.load(open(f'{SP}/night_cols_all.json')); F, CATS = list(c['F']), c['CATS']
X = v17_features.build()
tr = tr.merge(X, on='MVT_ID_mvt', how='left'); va = va.merge(X, on='MVT_ID_mvt', how='left')
FB = F + v17_features.NEW
print(f'recipe {os.environ["DC26_RECIPE"]}  F={len(F)}  F_BULK={len(FB)}  '
      f'tr={len(tr):,} va={len(va):,}  [{time.time()-T0:.0f}s]', flush=True)
for x in CATS:
    u = pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str)) | set(va[x].dropna().astype(str))))
    tr[x] = tr[x].astype(str).astype(u); va[x] = va[x].astype(str).astype(u)

base_of = lambda d: np.nan_to_num(np.where(~np.isnan(d.naive.values.astype(float)),
    d.naive.values.astype(float), d.apt_hr_naive.values.astype(float)).astype(float), nan=900.)
bt_, bv_ = base_of(tr), base_of(va)
yt, yv = tr.y.values.astype(float), va.y.values.astype(float)
nx = yt < THR                                     # bulk rows: what every bulk model trains on
nxv = yv < THR
rmse = lambda p, m: float(np.sqrt(np.mean((np.clip(p, FLOOR, CAP)[m] - yv[m]) ** 2)))
folds = list(KFold(2, shuffle=True, random_state=0).split(tr))     # identical to gate_a.py

# ---------------- pooled LightGBM, the 0.2 component ----------------
t0 = time.time(); lgb_tr = np.zeros(len(tr))
for ia, ib in folds:
    m = ia[nx[ia]]
    lgb_tr[ib] = bt_[ib] + lgb.LGBMRegressor(random_state=0, **POOLED).fit(
        tr.iloc[m][FB], yt[m] - bt_[m], categorical_feature=CATS).predict(tr.iloc[ib][FB])
mt = np.where(nx)[0]
lgb_va = bv_ + np.mean([lgb.LGBMRegressor(random_state=s, **POOLED).fit(
    tr.iloc[mt][FB], yt[mt] - bt_[mt], categorical_feature=CATS).predict(va[FB]) for s in S3], axis=0)
print(f'pooled LGB   val bulk {rmse(lgb_va, nxv):7.2f}   [{time.time()-t0:.0f}s]', flush=True)

# ---------------- per-airport LightGBM, the 0.4 component ----------------
t0 = time.time(); ap_t = tr.apt.astype(str).values; ap_v = va.apt.astype(str).values
pa_tr = np.zeros(len(tr))
for ia, ib in folds:
    fit_rows = ia[nx[ia]]
    for a in np.unique(ap_v):
        it = fit_rows[ap_t[fit_rows] == a]; iv = ib[ap_t[ib] == a]
        if len(iv) == 0: continue
        if len(it) < 500: pa_tr[iv] = lgb_tr[iv]; continue      # shipped fallback
        pa_tr[iv] = bt_[iv] + lgb.LGBMRegressor(random_state=0, **PERAPT).fit(
            tr.iloc[it][FB], yt[it] - bt_[it], categorical_feature=CATS).predict(tr.iloc[iv][FB])
pa_va = np.zeros(len(va))
for s in S3:
    p = np.full(len(va), np.nan)
    for a in np.unique(ap_v):
        it = mt[ap_t[mt] == a]; iv = np.where(ap_v == a)[0]
        if len(it) < 500: p[iv] = lgb_va[iv]; continue
        p[iv] = bv_[iv] + lgb.LGBMRegressor(random_state=s, **PERAPT).fit(
            tr.iloc[it][FB], yt[it] - bt_[it], categorical_feature=CATS).predict(va.iloc[iv][FB])
    assert not np.isnan(p).any(); pa_va += p / len(S3)
print(f'per-airport  val bulk {rmse(pa_va, nxv):7.2f}   [{time.time()-t0:.0f}s]', flush=True)
print(f'lgb vs perapt correlation on val: {np.corrcoef(lgb_va, pa_va)[0,1]:.4f}', flush=True)

# ---------------- classifier, unchanged from v15_trees.py and gate_a.py ----------------
t0 = time.time(); ext = (yt >= THR)
C = lambda s: HGC(learning_rate=.06, max_iter=400, max_leaf_nodes=63, min_samples_leaf=50,
    l2_regularization=1., categorical_features='from_dtype', early_stopping=True,
    validation_fraction=.1, n_iter_no_change=40, random_state=s)
Pv = np.mean([C(s).fit(tr[F], ext.astype(int)).predict_proba(va[F])[:, 1] for s in S3], 0)
Pt = np.mean([C(s).fit(tr[F], ext.astype(int)).predict_proba(tr[F])[:, 1] for s in S3], 0)
Pc = IsotonicRegression(out_of_bounds='clip').fit(Pt, ext.astype(float)).predict(Pv)
print(f'classifier   [{time.time()-t0:.0f}s]', flush=True)

# ---------------- feature matrices for the gate and the net ----------------
NUMF = [x for x in FB if x not in CATS]
def num(d):
    A = d[NUMF].to_numpy(np.float64); return A, np.isnan(A).astype(np.float32)
At, Mt = num(tr); Av, Mv = num(va); med = np.nanmedian(At, 0)
At = np.where(np.isnan(At), med, At); Av = np.where(np.isnan(Av), med, Av)
q1, q9 = np.percentile(At, 1, axis=0), np.percentile(At, 99, axis=0); sc = np.where(q9 - q1 < 1e-9, 1, q9 - q1)
Xt = np.hstack([np.clip((At - q1) / sc, -5, 5), Mt]).astype(np.float32)
Xv = np.hstack([np.clip((Av - q1) / sc, -5, 5), Mv]).astype(np.float32)
cards = []; CT = []; CV = []
for x in CATS:
    u = pd.Index(tr[x].astype(str).unique()); cards.append(len(u) + 1)
    CT.append(pd.Index(u).get_indexer(tr[x].astype(str)) + 1)
    CV.append(pd.Index(u).get_indexer(va[x].astype(str)) + 1)

np.savez_compressed(OUT, Xt=Xt, Xv=Xv, CT=np.clip(np.vstack(CT).T, 0, None),
    CV=np.clip(np.vstack(CV).T, 0, None), cards=np.array(cards),
    lgb_tr=lgb_tr, lgb_va=lgb_va, pa_tr=pa_tr, pa_va=pa_va, base_tr=bt_, base_va=bv_, Pc=Pc,
    sd_t=tr.sched_delay.values.astype(float), lb_t=tr.mvt_minus_lobt.values.astype(float),
    sd_v=va.sched_delay.values.astype(float), lb_v=va.mvt_minus_lobt.values.astype(float),
    yt=yt, yv=yv, folds=np.array([np.isin(np.arange(len(tr)), folds[0][1])]))
print(f'saved {OUT}  [{time.time()-T0:.0f}s total]', flush=True)
