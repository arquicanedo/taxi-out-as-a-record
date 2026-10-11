"""v55 stage 1: the RANKING-side tree components rebuilt with the OPDI airframe features.

s223 measured +0.9557 val-s on the blend; s224 established that the FOUR original features
win and the ten-feature expansion loses by 0.1008. The decomposition that decides this
script's shape:

    pooled+opdi only        +0.3981
    per-airport+opdi only   +0.7195     <- most of the gain is here
    BOTH                    +0.9557

So unlike v38, which changed the pooled model alone and could therefore inherit v37's control,
this replaces BOTH tree components and needs its own.

CONTROL: the per-airport model is rebuilt a second time WITHOUT the OPDI columns, using the
shipped config and seeds, and must reproduce v15_trees_jul19.parquet's `perapt` column. If it
does, the substitution machinery is faithful and the +opdi array can be trusted in its place.

The pooled model uses the v38/s181 recipe (lr .03 / 6000 / 511 / ff .7), which is what has
been shipped since 2026-09-07. v15_trees.py's docstring warns that raised capacity must never
reach the per-airport loop (+0.82 to +2.17 there), so per-airport stays at the shipped config.
"""
import os; os.environ.setdefault('DC26_RECIPE','v19'); os.environ.setdefault('DC26_TAG','_jul19')
import json, sys, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import features
W=os.environ['DC26_WORK']; S=(0,1,2); THR=7200
OPDI=['af_since_arr','af_sched_turn','af_prev_taxiin','af_has']
V38=dict(learning_rate=.03,n_estimators=6000,num_leaves=511,feature_fraction=.7,
         min_child_samples=20,reg_lambda=1.,n_jobs=-1,verbose=-1)
PA=dict(learning_rate=.06,n_estimators=600,num_leaves=127,min_child_samples=100,
        reg_lambda=1.,n_jobs=-1,verbose=-1)
T0=time.time(); log=lambda m: print(f'[{time.time()-T0:7.1f}s] {m}',flush=True)
# features.wk() points at /tmp, which macOS purges; regen.sh stage 5 copies these into
# $DC26_WORK for exactly this reason, so read the durable copies.
TAG=os.environ.get('DC26_TAG','_jul19')
tr=pd.read_parquet(f'{W}/v15_tr{TAG}.parquet'); rk=pd.read_parquet(f'{W}/v15_rk{TAG}.parquet')
c=json.load(open(f'{W}/v15_cols{TAG}.json')); F,FB,CATS=c['F'],list(c['F_BULK']),c['CATS']
A=pd.read_parquet(f'{W}/opdi_airframe.parquet'); n0,n1=len(tr),len(rk)
tr=tr.merge(A,on='MVT_ID_mvt',how='left'); rk=rk.merge(A,on='MVT_ID_mvt',how='left')
assert len(tr)==n0 and len(rk)==n1, 'opdi merge changed row count'
print(f'train {len(tr):,}  rank {len(rk):,}  FB={len(FB)}')
print(f'OPDI coverage: train {tr.af_has.fillna(0).mean():.1%}  rank {rk.af_has.fillna(0).mean():.1%}')
for x in CATS:
    u=pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str))|set(rk[x].dropna().astype(str))))
    tr[x]=tr[x].astype(str).astype(u); rk[x]=rk[x].astype(str).astype(u)
ext=(tr.y>=THR).values; trn=tr[~ext]
nt=trn.naive.values.astype(float); nr=rk.naive.values.astype(float)
bt=np.where(~np.isnan(nt),nt,trn.apt_hr_naive.values.astype(float))
br=np.where(~np.isnan(nr),nr,rk.apt_hr_naive.values.astype(float))
g=np.nanmedian(bt); bt=np.nan_to_num(bt,nan=g); br=np.nan_to_num(br,nan=g)
rt=trn.y.values-bt
ap_t=trn.apt.astype(str).values; ap_r=rk.apt.astype(str).values; APTS=np.unique(ap_r)

def perapt(cols,tag,fallback):
    out=np.zeros(len(rk))
    for s in S:
        p=np.full(len(rk),np.nan); t0=time.time()
        for a in APTS:
            it=ap_t==a; iv=ap_r==a
            if it.sum()<500: p[iv]=fallback[iv]; continue
            m=lgb.LGBMRegressor(random_state=s,**PA).fit(trn[cols][it],rt[it],categorical_feature=CATS)
            p[iv]=br[iv]+m.predict(rk[cols][iv])
        assert not np.isnan(p).any(); out+=p/len(S); log(f'  perapt {tag} seed {s} [{time.time()-t0:.0f}s]')
    return out

# ---- CONTROL FIRST: reproduce the shipped per-airport array without OPDI ----------------
old=pd.read_parquet(f'{W}/v15_trees_jul19.parquet').set_index('MVT_ID_mvt').reindex(rk.MVT_ID_mvt)
lgb_old=old.lgbm.values; pa_old=old.perapt.values
log('CONTROL: per-airport WITHOUT opdi, shipped config and seeds')
pa_ctl=perapt(FB,'control',lgb_old)
d=pa_ctl-pa_old
print(f'\nCONTROL vs shipped perapt: mean {d.mean():+.4f}  sd {d.std():.4f}  |max| {np.abs(d).max():.4f}  '
      f'corr {np.corrcoef(pa_ctl,pa_old)[0,1]:.6f}')
assert np.abs(d).max()<1.0, f'CONTROL FAILED: per-airport reproduction differs by up to {np.abs(d).max():.2f}s'
print('  reproduction faithful (max deviation under 1 s).\n')

# ---- the pooled model at the shipped v38 recipe, with OPDI ------------------------------
log(f'pooled at the v38 recipe, {len(FB+OPDI)} features, {len(S)} seeds')
ps=[]
for s in S:
    t0=time.time()
    m=lgb.LGBMRegressor(random_state=s,**V38).fit(trn[FB+OPDI],rt,categorical_feature=CATS)
    ps.append(br+m.predict(rk[FB+OPDI])); log(f'  pooled seed {s} [{time.time()-t0:.0f}s]')
lgb_opdi=np.mean(ps,0)
log('per-airport WITH opdi')
pa_opdi=perapt(FB+OPDI,'opdi',lgb_opdi)
out=pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'lgbm_opdi':lgb_opdi,'perapt_opdi':pa_opdi,
                  'perapt_ctl':pa_ctl})
out.to_parquet(f'{W}/v55_trees.parquet',index=False)
print(f'\nwrote {W}/v55_trees.parquet')
for nm,new,ref in (('pooled',lgb_opdi,lgb_old),('per-airport',pa_opdi,pa_old)):
    dd=new-ref
    print(f'  {nm:12s} vs shipped: mean {dd.mean():+.2f}  sd {dd.std():.2f}  |max| {np.abs(dd).max():,.0f}')
