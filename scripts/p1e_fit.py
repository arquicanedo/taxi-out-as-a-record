"""Track E re-test harness: the SHIPPED bulk construction, trained on the ten non-window months.

Construction reproduced from scripts/v99_components.py (the ship since v99):
  pooled      LightGBM lr .06, 2000 trees, 255 leaves, min_child 20, lambda 1, colsample 0.4
  per-airport LightGBM lr .06,  600 trees, 127 leaves, min_child 100, lambda 1, colsample 0.4
  target      y - base, base = naive, else apt_hr_naive, else the training nanmedian
  training    non-extreme rows (y < 7200) of months 2-6 and 8-12
  features    v15_cols_jul19.json F_BULK (71 columns, 7 categorical)  <- the SHIP's list
Scores the 344,336 window rows (months 1 and 7, v15_tr_jul19 file order) and saves them.

Usage: p1e_fit.py ARM SEED MODEL [CS]
  ARM    base | any arm defined in ARMS below
  MODEL  pooled | perapt
Env:   FEAT=v15_cols.json swaps in the 62-feature list the s447/s480-s482 sweeps used.
"""
import os, sys, json, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
W=os.environ['DC26_WORK']
OUT=(os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackE')
ARM=sys.argv[1]; SEED=int(sys.argv[2]); MODEL=sys.argv[3]; CS=float(sys.argv[4]) if len(sys.argv)>4 else 0.4
NJ=int(os.environ.get('NJ','4')); t0=time.time()
FEATF=os.environ.get('FEAT','v15_cols_jul19.json')
tr=pd.read_parquet(f'{W}/v15_tr_jul19.parquet')
c=json.load(open(f'{W}/{FEATF}')); FB,CATS=list(c['F_BULK']),c['CATS']
mon=pd.to_datetime(tr.MVT_TIME_UTC_mvt).dt.month.values
y=tr.y.values.astype(float); ext=y>=7200
TRN=~np.isin(mon,[1,7])&~ext; WIN=np.isin(mon,[1,7])
extra=None
if ARM!='base':
    import p1e_arms
    tr,FB,extra=p1e_arms.apply(ARM,tr,FB,TRN,WIN,y)
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
print(f'  arm {ARM} seed {SEED} model {MODEL} cs {CS} features {len(FB)} ({FEATF}) '
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
tag=f'{ARM}_{MODEL}_cs{CS}_s{SEED}'+('' if FEATF=='v15_cols_jul19.json' else '_f62')
np.save(f'{OUT}/win_{tag}.npy',pw.astype(np.float64))
print(f'  {tag}: ordinary Jan {r((mw==1)&~ew):.4f}  Jul {r((mw==7)&~ew):.4f}  both {r(~ew):.4f}'
      f'   [{time.time()-t0:.0f}s]',flush=True)
