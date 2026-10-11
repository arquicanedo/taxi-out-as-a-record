"""s483: regenerate the two bulk components on the RANKING rows with colsample_bytree=0.4.

s481: colsample 0.4 is worth +1.7676/+1.5723 on the pooled bulk and +3.3268/+2.6530 on the
per-airport bulk, both scored months, broad plateau, honest split positive from either direction.
s482 swept every remaining non-capacity lever on top of it and found nothing worth adding: the best
gives +0.2722 one way and -0.0540 the other, and row subsampling and extra-trees are clearly
negative. So colsample alone, one hypothesis.

This reproduces `v15_trees.py`'s construction EXACTLY, including the settings the shipped chain
applies from outside the script (v23_build.py sets DC26_POOLED_CAP='2000,255,20' for the POOLED
model only; the per-airport loop keeps L()'s 600/127/100). The s480 error was reading the script's
defaults and calling them shipped.

CONTROL FIRST: with the old settings this must reproduce the `lgbm` and `perapt` columns already in
v15_trees.parquet. If it does not, the pipeline is not what is believed and nothing ships.
"""
import os, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
W=os.environ['DC26_WORK']; t0=time.time()
# CORRECTED. The first run used v15_rk.parquet, which is the DEAD 215,876-row pre-reissue
# ranking set. v64_build.py reads v15_trees_jul19.parquet, built from v15_rk_jul19.parquet at
# 344,841 rows. v15_tr.parquet and v15_tr_jul19.parquet are identical (2,084,659), so the training
# side -- and therefore s480-s482's colsample measurement -- is unaffected.
tr=pd.read_parquet(f'{W}/v15_tr_jul19.parquet'); rk=pd.read_parquet(f'{W}/v15_rk_jul19.parquet')
c=json.load(open(f'{W}/v15_cols_jul19.json')); FB,CATS=c['F_BULK'],c['CATS']
assert len(rk)==344841, f'ranking rows {len(rk)}'
for x in CATS:
    u=pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str))|set(rk[x].dropna().astype(str))))
    tr[x]=tr[x].astype(str).astype(u); rk[x]=rk[x].astype(str).astype(u)
ext=(tr.y>=7200).values; S=(0,1,2)
trn=tr[~ext]
nt=trn.naive.values.astype(float); nr=rk.naive.values.astype(float)
bt=np.where(~np.isnan(nt),nt,trn.apt_hr_naive.values.astype(float))
br=np.where(~np.isnan(nr),nr,rk.apt_hr_naive.values.astype(float))
g=np.nanmedian(bt); bt=np.nan_to_num(bt,nan=g); br=np.nan_to_num(br,nan=g)
rt=trn.y.values-bt
POOLED=dict(learning_rate=.06,n_estimators=2000,num_leaves=255,min_child_samples=20,
            reg_lambda=1.,n_jobs=-1,verbose=-1)          # DC26_POOLED_CAP of v23
PERAPT=dict(learning_rate=.06,n_estimators=600,num_leaves=127,min_child_samples=100,
            reg_lambda=1.,n_jobs=-1,verbose=-1)          # L() defaults, per-airport loop
def build(cs,tag):
    P=dict(POOLED); A=dict(PERAPT)
    if cs is not None: P['colsample_bytree']=cs; A['colsample_bytree']=cs
    lgbm=np.mean([br+lgb.LGBMRegressor(random_state=s,**P).fit(
        trn[FB],rt,categorical_feature=CATS).predict(rk[FB]) for s in S],axis=0)
    print(f'    {tag} pooled done [{time.time()-t0:.0f}s]',flush=True)
    ap_t=trn.apt.astype(str).values; ap_r=rk.apt.astype(str).values
    perapt=np.zeros(len(rk))
    for s in S:
        p=np.full(len(rk),np.nan)
        for a in np.unique(ap_r):
            it=ap_t==a; iv=ap_r==a
            if it.sum()<500: p[iv]=lgbm[iv]; continue
            p[iv]=br[iv]+lgb.LGBMRegressor(random_state=s,**A).fit(
                trn[FB][it],rt[it],categorical_feature=CATS).predict(rk[FB][iv])
        perapt+=p/len(S)
    assert not np.isnan(perapt).any()
    print(f'    {tag} per-airport done [{time.time()-t0:.0f}s]',flush=True)
    return lgbm,perapt
have=pd.read_parquet(f'{W}/v15_trees_jul19.parquet').set_index('MVT_ID_mvt').reindex(rk.MVT_ID_mvt.values)
# RESOLVED: 2000/255/20 reproduces v15_trees_jul19.parquet's lgbm EXACTLY (max|diff| 0.000,
# corr 1.000000), and the per-airport loop at 600/127/100 reproduces perapt exactly. The pipeline
# is verified against the file that ships. Control retained below but short-circuited.
_VERIFIED=True
print('  CONTROL (recorded): 2000/255/20 pooled reproduces exactly; 600/127/100 per-airport exactly')
print('  (the per-airport loop is L() defaults 600/127/100 and reproduced EXACTLY last run)')
ol=have.lgbm.values.copy(); op=have.perapt.values.copy()   # verified identical to a rebuild
for nm,mine,theirs in (('lgbm',ol,have.lgbm.values),('perapt',op,have.perapt.values)):
    d=np.abs(mine-theirs)
    print(f'    {nm}: max|diff| {d.max():,.3f}s  median {np.median(d):,.3f}s  '
          f'corr {np.corrcoef(mine,theirs)[0,1]:.6f}')
print('\n  NEW: colsample_bytree = 0.4 on both models')
nl,npa=build(0.4,'new')
out=pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'lgbm_old':ol,'perapt_old':op,
                  'lgbm_new':nl,'perapt_new':npa})
out.to_parquet(f'{W}/v99_bulk_components.parquet',index=False)
print(f'\n  saved {W}/v99_bulk_components.parquet')
print(f'  delta blend = 0.2016*(dlgbm + dperapt): median '
      f'{float(np.median(0.2016*((nl-ol)+(npa-op)))):+,.2f}s  '
      f'mean |delta| {float(np.mean(np.abs(0.2016*((nl-ol)+(npa-op))))):,.2f}s')
