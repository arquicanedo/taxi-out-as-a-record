"""v84 = v79 plus a two-stage model for the OPERATIONAL extremes (s346's NEITHER class).

s346 established that 138 of 584 labelled 2025 extremes are neither schedule-copies nor date
rollovers: BLOCK sits on the minute grid like a real off-block, values cluster just over two hours,
they spread over five airports, and the flight-plan join is absent on only half. They are long
taxis that happened, they are the only extremes a model can reach, and a classifier separates them
at AUC 0.9989 against a shuffled null of 0.5693.

This applies that to the scored set. Nothing here touches the v34 cell, the copy gate, or any
board-derived constant.

  stage 1   classify "operational extreme" on all labelled 2025, predict on the scored rows
  stage 2   regress y on the labelled operational-extreme rows only
  mix       p*stage2 + (1-p)*v79        soft, no threshold, no free scalar

VALIDATION IS OUT OF SAMPLE BY MONTH. Trained on 2025 minus January and July, scored on January and
July, which mirrors the scored set's composition. The gain reported is that holdout's, and the
shipped model then refits on all twelve months, which is the same discipline every other component
here uses.

CONTROL: with MIX=0 it reproduces v79 byte-identically.
"""
import hashlib, json, os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgbm
from sklearn.metrics import roc_auc_score
D=os.environ['DC26_DATA']
W=os.environ['DC26_WORK']; TR=f'{W}/v15_tr_jul19.parquet'; RK=f'{W}/v15_rk_jul19.parquet'
OUT=os.environ.get('V84_OUT','submissions/quick-boat_v84.parquet')
MIX=os.environ.get('MIX','1')=='1'
THR,FLOOR,CAP,DAY=7200,60.,200000.,86400.; SHA79='452aab544906b2e1'
c=json.load(open(f'{W}/night_cols_all.json')); F,CATS=c['F'],c['CATS']
def dedup(xs):
    seen=set(); out=[]
    for x in xs:
        if x not in seen: seen.add(x); out.append(x)
    return out
trn=set(pq.read_schema(TR).names); rkn=set(pq.read_schema(RK).names)
tr=pq.read_table(TR,columns=[x for x in dedup(F+CATS+['MVT_ID_mvt','TAXITIME_SEC_mvt','sched_delay','month']) if x in trn]).to_pandas()
rk=pq.read_table(RK,columns=[x for x in dedup(F+CATS+['MVT_ID_mvt','sched_delay']) if x in rkn]).to_pandas()
assert not tr.columns.duplicated().any() and not rk.columns.duplicated().any()
print(f'labelled {len(tr):,}   scored {len(rk):,}   features {len(F)}+{len(CATS)}')
y=tr.TAXITIME_SEC_mvt.astype(float).values; ms=tr.sched_delay.astype(float).values
ext=y>=THR; fall=ext&(np.abs(y-ms)<=60); roll=ext&(y>=DAY*.9)&(y<=DAY*1.25)&~fall
op=(ext&~fall&~roll)                                     # the OPERATIONAL extremes
assert 'month' in tr.columns, 'no month column; cannot hold out Jan+Jul'
mon=tr.month.astype(int).values
print(f'  operational extremes {int(op.sum())}   fallback {int(fall.sum())}   rollover {int(roll.sum())}')
for c_ in CATS:
    u=pd.Categorical(tr[c_].astype(str)).categories
    tr[c_]=pd.Categorical(tr[c_].astype(str),categories=u)
    rk[c_]=pd.Categorical(rk[c_].astype(str),categories=u)
FEATS=dedup([x for x in F if x not in CATS]+CATS)      # a name in both would duplicate the column
assert len(FEATS)==len(set(FEATS))
X=tr[FEATS]; Xr=rk[FEATS]
assert not X.columns.duplicated().any() and list(X.columns)==list(Xr.columns)
print(f'  model features {len(FEATS)}')
PC=dict(objective='binary',learning_rate=.06,n_estimators=400,num_leaves=63,min_child_samples=40,
        reg_lambda=1.,n_jobs=-1,verbose=-1)
PR=dict(objective='regression',learning_rate=.05,n_estimators=600,num_leaves=63,min_child_samples=40,
        reg_lambda=1.,n_jobs=-1,verbose=-1)
# ---- honest holdout: train outside Jan+Jul, score on Jan+Jul ----
ho=np.isin(mon,[1,7]); tri=~ho
mc=lgbm.LGBMClassifier(random_state=0,**PC).fit(X[tri],op[tri].astype(int))
pv=mc.predict_proba(X[ho])[:,1]
auc=roc_auc_score(op[ho].astype(int),pv)
mr=lgbm.LGBMRegressor(random_state=0,**PR).fit(X[tri&op],y[tri&op])
rv=np.clip(mr.predict(X[ho]),FLOOR,CAP)
print(f'\n  HOLDOUT Jan+Jul: n {int(ho.sum()):,}   operational {int(op[ho].sum())}'
      f'   stage-1 AUC {auc:.4f}   mean p {pv.mean():.6f}   rate {op[ho].mean():.6f}')
sh=op[tri].copy(); np.random.default_rng(0).shuffle(sh)
mcn=lgbm.LGBMClassifier(random_state=0,**PC).fit(X[tri],sh.astype(int))
print(f'  shuffled-label null AUC {roc_auc_score(op[ho].astype(int),mcn.predict_proba(X[ho])[:,1]):.4f}')
assert auc>0.90, f'stage-1 AUC {auc:.4f} too low to ship'
# ---- ship: refit on all twelve months, apply to the scored set ----
mcf=lgbm.LGBMClassifier(random_state=0,**PC).fit(X,op.astype(int))
mrf=lgbm.LGBMRegressor(random_state=0,**PR).fit(X[op],y[op])
pr_=mcf.predict_proba(Xr)[:,1]; rr=np.clip(mrf.predict(Xr),FLOOR,CAP)
print(f'\n  SCORED set: mean p {pr_.mean():.6f}   p>=0.5 on {int((pr_>=.5).sum())} rows'
      f'   stage-2 median {np.median(rr):,.0f} s')
tpl=pq.read_table(f'{D}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
v79=pq.read_table('submissions/quick-boat_v79.parquet').to_pandas()
assert hashlib.sha256(open('submissions/quick-boat_v79.parquet','rb').read()).hexdigest()[:16]==SHA79
assert bool((v79.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
o=pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'p':pr_,'r':rr}).set_index('MVT_ID_mvt').reindex(tb.MVT_ID_mvt)
assert o.p.notna().all(), 'scored rows did not align'
base=v79.TAXITIME_SEC_mvt.values.astype(float)
p=base.copy()
if MIX: p=o.p.values*o.r.values+(1-o.p.values)*base
q=np.clip(np.rint(p),FLOOR,CAP)
nd=int((q.astype('int32')!=base.astype('int32')).sum())
print(f'  rows changed against v79: {nd:,}')
assert (nd>0)==MIX
oo=pd.DataFrame({'MVT_ID_mvt':tb.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':q.astype('int32')})
t=pa.Table.from_pandas(oo,schema=tpl.schema,preserve_index=False); assert t.schema.equals(tpl.schema)
pq.write_table(t,OUT)
print(f'  wrote {OUT}  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')

# ---------------------------------------------------------------------------------------------
print('\n=== WHAT THE MIX DOES ON THE HOLDOUT, aligned on MVT_ID_mvt ===')
# An earlier version compared positionally and scored 874.575 where the shipped blend is known to
# score 397.352 on this window. Same row COUNT, different ORDER. Align explicitly.
z=np.load(f'{W}/gate_stage_a2.npz')
yv=z['yv'].astype(float); nv=len(yv)
ld=lambda f: np.load(f'{W}/{f}').ravel()[:nv]
bulk=(.2230*z['lgb_va']+.2030*z['pa_va']+.0880*ld('arch_d2_wide_va_K32.npy')+.4860*ld('arch_d2_wide_va_K128.npy'))
Pc=z['Pc'].astype(float); sd_=z['sd_v'].astype(float); lb=z['lb_v'].astype(float)
hit=np.isfinite(lb)&(lb>=THR)
cp=np.where(hit,np.clip(np.where(np.isfinite(lb),lb,sd_),FLOOR,CAP),np.clip(sd_,FLOOR,CAP))
blend=np.clip(np.rint(Pc*cp+(1-Pc)*bulk),FLOOR,CAP)
vid=pq.read_table(f'{W}/night_va_all.parquet',columns=['MVT_ID_mvt']).to_pandas().MVT_ID_mvt.values
assert len(vid)==nv
B=pd.DataFrame({'MVT_ID_mvt':vid,'blend':blend,'yv':yv}).set_index('MVT_ID_mvt')
H=pd.DataFrame({'MVT_ID_mvt':tr.MVT_ID_mvt.values[ho],'p':pv,'r':rv,'y':y[ho],
                'op':op[ho]}).set_index('MVT_ID_mvt')
J=H.join(B,how='inner')
print(f'  joined {len(J):,} of {len(H):,} holdout rows and {len(B):,} blend rows')
assert len(J)>0.98*len(H), 'join lost too many rows'
assert float(np.abs(J.y-J.yv).max())<1.0, 'y disagrees between sources after the join'
yh=J.y.values; bl=J.blend.values
mixed=np.clip(np.rint(J.p.values*J.r.values+(1-J.p.values)*bl),FLOOR,CAP)
eh=yh>=THR; opj=J.op.values.astype(bool)
rm=lambda q,m: float(np.sqrt(np.mean((q[m]-yh[m])**2)))
print(f'  {"":26} {"FULL":>9} {"ordinary":>9} {"operational":>12}')
print(f'  {"shipped blend + gate":26} {rm(bl,np.ones(len(yh),bool)):9.3f} {rm(bl,~eh):9.3f} {rm(bl,opj):12,.0f}')
print(f'  {"+ operational two-stage":26} {rm(mixed,np.ones(len(yh),bool)):9.3f} {rm(mixed,~eh):9.3f} {rm(mixed,opj):12,.0f}')
rng=np.random.default_rng(0)
for nm,m in (('FULL window',np.ones(len(yh),bool)),('ORDINARY rows',~eh)):
    nn=int(m.sum()); e0=(bl[m]-yh[m])**2; e1=(mixed[m]-yh[m])**2; g=np.empty(2000)
    for b in range(2000):
        i2=rng.integers(0,nn,nn); g[b]=np.sqrt(e0[i2].mean())-np.sqrt(e1[i2].mean())
    print(f'    PAIRED {nm:16} {g.mean():+8.4f}  SE {g.std():7.4f}  P(>0) {float((g>0).mean()):.3f}')
