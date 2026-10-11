"""v15 stage 1: bulk LightGBM (+block_dev), per-airport bulk, and the classifier.

Differences from v8s_trees.py:
  * F_BULK now carries block_dev (s77)
  * adds a 10-model per-airport bulk (s61), the only decorrelation candidate
    that beat the pooled model solo AND decorrelated from it
The classifier is UNCHANGED: same F, same seeds, so Pc should reproduce v14's.
"""
import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HGC
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import features, os
tr=pd.read_parquet(features.wk('v15_tr')); rk=pd.read_parquet(features.wk('v15_rk'))
c=json.load(open(features.wk('v15_cols','json'))); F,FB,CATS=c['F'],c['F_BULK'],c['CATS']
for x in CATS:
    u=pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str))|set(rk[x].dropna().astype(str))))
    tr[x]=tr[x].astype(str).astype(u); rk[x]=rk[x].astype(str).astype(u)
ext=(tr.y>=7200).values; S=(0,1,2); t0=time.time()
trn=tr[~ext]; nt=trn.naive.values.astype(float); nr=rk.naive.values.astype(float)
bt=np.where(~np.isnan(nt),nt,trn.apt_hr_naive.values.astype(float)); br=np.where(~np.isnan(nr),nr,rk.apt_hr_naive.values.astype(float))
g=np.nanmedian(bt); bt=np.nan_to_num(bt,nan=g); br=np.nan_to_num(br,nan=g)
rt=trn.y.values-bt
L=lambda s: lgb.LGBMRegressor(learning_rate=.06,n_estimators=600,num_leaves=127,min_child_samples=100,
    reg_lambda=1.,n_jobs=-1,verbose=-1,random_state=s)
# DC26_POOLED_CAP="n,leaves,min_child" raises capacity on the POOLED model only.
# Measured on the all window: 2000x255x20 gives -1.82 on the pooled model but
# +0.82 to +2.17 on the per-airport models, which train on ~1/10 the data each
# and are already at the right capacity. So it must never reach the per-airport
# loop. Default empty reproduces every prior build byte-identically.
_CAP=os.environ.get('DC26_POOLED_CAP','')
def Lp(s):
    if not _CAP: return L(s)
    n,lv,mcs=(int(x) for x in _CAP.split(','))
    return lgb.LGBMRegressor(learning_rate=.06,n_estimators=n,num_leaves=lv,min_child_samples=mcs,
        reg_lambda=1.,n_jobs=-1,verbose=-1,random_state=s)
lgbm=np.mean([br+Lp(s).fit(trn[FB],rt,categorical_feature=CATS).predict(rk[FB]) for s in S],axis=0)
print(f'pooled bulk done [{time.time()-t0:.0f}s]',flush=True)
# --- per-airport bulk (s61) ---
t1=time.time(); ap_t=trn.apt.astype(str).values; ap_r=rk.apt.astype(str).values
perapt=np.zeros(len(rk))
for s in S:
    p=np.full(len(rk),np.nan)
    for a in np.unique(ap_r):
        it=ap_t==a; iv=ap_r==a
        if it.sum()<500: p[iv]=lgbm[iv]; continue
        p[iv]=br[iv]+L(s).fit(trn[FB][it],rt[it],categorical_feature=CATS).predict(rk[FB][iv])
    perapt+=p/len(S)
assert not np.isnan(perapt).any()
print(f'per-airport bulk done ({len(np.unique(ap_r))} airports x {len(S)} seeds) [{time.time()-t1:.0f}s]',flush=True)
# --- classifier, unchanged ---
t2=time.time()
C=lambda s:HGC(learning_rate=.06,max_iter=400,max_leaf_nodes=63,min_samples_leaf=50,l2_regularization=1.,
   categorical_features='from_dtype',early_stopping=True,validation_fraction=.1,n_iter_no_change=40,random_state=s)
Pr=np.mean([C(s).fit(tr[F],ext.astype(int)).predict_proba(rk[F])[:,1] for s in S],axis=0)
Pt=np.mean([C(s).fit(tr[F],ext.astype(int)).predict_proba(tr[F])[:,1] for s in S],axis=0)
Pc=IsotonicRegression(out_of_bounds='clip').fit(Pt,ext.astype(float)).predict(Pr)
print(f'classifier done [{time.time()-t2:.0f}s]',flush=True)
out=pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'lgbm':lgbm,'perapt':perapt,'Pc':Pc,
  'sched_delay':rk.sched_delay.values,'mvt_minus_lobt':rk.mvt_minus_lobt.values})
# The Pc-vs-v14 control is an INNER join. If /tmp/v8s_trees covers a different
# extract than rk (old 215,876 ranking rows against the new 344,841), that join
# silently truncates the output and v15_combine's row assert then passes on a
# short file. Compare only when the id sets match, and never let it touch `out`.
_p=features.wk('v8s_trees')
if os.path.exists(_p):
    old=pd.read_parquet(_p)[['MVT_ID_mvt','Pc']].rename(columns={'Pc':'Pc_v14'})
    if len(old)==len(out) and set(old.MVT_ID_mvt)==set(out.MVT_ID_mvt):
        chk=out[['MVT_ID_mvt','Pc']].merge(old,on='MVT_ID_mvt')
        d=np.abs(chk.Pc.values-chk.Pc_v14.values)
        print(f'\nPc vs v14: max|diff| {d.max():.6f}, rows differing >1e-6: {int((d>1e-6).sum())}  (expect ~0)')
    else:
        print(f'\nPc vs v14 control SKIPPED: {_p} covers {len(old):,} rows, this build has {len(out):,}')
else:
    print(f'\nPc vs v14 control SKIPPED: {_p} absent')
print(f'lgbm vs perapt correlation on ranking: {np.corrcoef(out.lgbm,out.perapt)[0,1]:.4f}')
assert len(out)==len(rk), f'trees output {len(out)} != ranking {len(rk)}'
out.to_parquet(features.wk('v15_trees'),index=False)
print(f"saved {features.wk('v15_trees')}  rows {len(out):,}  [{time.time()-t0:.0f}s total]",flush=True)
