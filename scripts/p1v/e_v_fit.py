"""Verifier copy of trackE/p1e_fit.py with ONE switch: TE=leakfree recomputes the eight te_* target
encodings from the ten TRAINING months only (months 2-6, 8-12, all rows y>0, the night_prep.py / v8s
formula), so a window row's encoding no longer contains its own label or any Jan/Jul label.
TE=stored (default) reproduces p1e_fit.py exactly. Everything else is p1e_fit.py verbatim.
Usage: v_fit.py ARM SEED MODEL [CS]"""
import os, sys, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
TE_DIR=(os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackE')
sys.path.insert(0,TE_DIR)
W=os.environ['DC26_WORK']
OUT=TE_DIR+'/verify'
ARM=sys.argv[1]; SEED=int(sys.argv[2]); MODEL=sys.argv[3]; CS=float(sys.argv[4]) if len(sys.argv)>4 else 0.4
NJ=int(os.environ.get('NJ','4')); t0=time.time(); TE=os.environ.get('TE','stored')
FEATF=os.environ.get('FEAT','v15_cols_jul19.json')
tr=pd.read_parquet(f'{W}/v15_tr_jul19.parquet')
c=json.load(open(f'{W}/{FEATF}')); FB,CATS=list(c['F_BULK']),c['CATS']
mon=pd.to_datetime(tr.MVT_TIME_UTC_mvt).dt.month.values
y=tr.y.values.astype(float); ext=y>=7200
TRN=~np.isin(mon,[1,7])&~ext; WIN=np.isin(mon,[1,7])
if TE=='leakfree':
    SPEC=[(['apt','STAND_mvt','RUNWAY_mvt'],'te_stand_rwy',50),(['apt','STAND_mvt'],'te_stand',50),(['apt','RUNWAY_mvt'],'te_rwy',50),
          (['apt','AIRCRAFT_OPERATOR_flt'],'te_op',50),(['apt'],'te_apt',50),(['ADES_mvt'],'te_des',100),
          (['apt','ADES_mvt'],'te_apt_des',100),(['apt','aobt_null'],'te_apt_null',20)]
    kd=tr[['apt','STAND_mvt','RUNWAY_mvt','AIRCRAFT_OPERATOR_flt','ADES_mvt','aobt_null','y']].copy()
    for cc in ['apt','STAND_mvt','RUNWAY_mvt','AIRCRAFT_OPERATOR_flt','ADES_mvt']: kd[cc]=kd[cc].astype(object)
    src=kd[~np.isin(mon,[1,7])]     # ten training months, all rows (extremes included, as night_prep/v8s)
    for k,n,q in SPEC:
        gm=src.y.mean(); g=src.groupby(k,observed=True).y.agg(['mean','size']); sm=(g['mean']*g['size']+gm*q)/(g['size']+q)
        new=kd.set_index(k).index.map(sm).astype(float).values
        old=tr[n].values
        print(f'    te {n:<13} window mean|old-new| {np.nanmean(np.abs(old[WIN]-new[WIN])):7.3f}  train mean|old-new| {np.nanmean(np.abs(old[TRN]-new[TRN])):7.3f}  win NaN {np.isnan(new[WIN]).mean():.4f}',flush=True)
        tr[n]=new
extra=None
if ARM!='base':
    sys.path.insert(0,OUT); import v_arms
    tr,FB,extra=v_arms.apply(ARM,tr,FB,TRN,WIN,y)
    CATS=list(CATS)+list(extra.get('cats',[]))
for x in CATS:
    u=pd.api.types.CategoricalDtype(sorted(set(tr[x].dropna().astype(str))))
    tr[x]=tr[x].astype(str).astype(u)
nv=tr.naive.values.astype(float); ah=tr.apt_hr_naive.values.astype(float)
b=np.where(~np.isnan(nv),nv,ah); g=np.nanmedian(b[TRN]); b=np.nan_to_num(b,nan=g)
rt=y-b
wkw={} if extra is None or 'weight' not in extra else {'sample_weight':extra['weight'][TRN]}
X=tr[FB]
if MODEL=='pooled':
    P=dict(objective='regression',learning_rate=.06,n_estimators=2000,num_leaves=255,
           min_child_samples=20,reg_lambda=1.,n_jobs=NJ,verbose=-1,random_state=SEED)
else:
    P=dict(objective='regression',learning_rate=.06,n_estimators=600,num_leaves=127,
           min_child_samples=100,reg_lambda=1.,n_jobs=NJ,verbose=-1,random_state=SEED)
if CS<1: P['colsample_bytree']=CS
if extra is not None and 'params' in extra: P.update(extra['params'])
print(f'  arm {ARM} seed {SEED} model {MODEL} cs {CS} TE {TE} features {len(FB)} ({FEATF}) '
      f'train rows {int(TRN.sum()):,} window {int(WIN.sum()):,} [{time.time()-t0:.0f}s]',flush=True)
if MODEL=='pooled':
    m=lgb.LGBMRegressor(**P).fit(X[TRN],rt[TRN],categorical_feature=CATS,**wkw)
    pw=b[WIN]+m.predict(X[WIN])
else:
    apt=tr.apt.astype(str).values; pw=np.full(int(WIN.sum()),np.nan); aw=apt[WIN]
    Xw=X[WIN]; bw=b[WIN]
    for a in np.unique(aw):
        it=TRN&(apt==a); iv=aw==a
        if it.sum()<500: pw[iv]=bw[iv]; continue
        kw={} if not wkw else {'sample_weight':extra['weight'][it]}
        pw[iv]=bw[iv]+lgb.LGBMRegressor(**P).fit(X[it],rt[it],categorical_feature=CATS,**kw).predict(Xw[iv])
    assert not np.isnan(pw).any()
pw=np.clip(pw,60,200000)
yw=y[WIN]; ew=ext[WIN]; mw=mon[WIN]
r=lambda k: float(np.sqrt(np.mean((yw[k]-pw[k])**2)))
tag=f'{ARM}_{MODEL}_cs{CS}_s{SEED}_te{TE}'+('' if FEATF=='v15_cols_jul19.json' else '_f62')
np.save(f'{OUT}/win_{tag}.npy',pw.astype(np.float64))
print(f'  {tag}: ordinary Jan {r((mw==1)&~ew):.4f}  Jul {r((mw==7)&~ew):.4f}  both {r(~ew):.4f}'
      f'   [{time.time()-t0:.0f}s]',flush=True)
