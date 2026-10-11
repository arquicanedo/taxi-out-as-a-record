"""v38/v39: v34 with the RANKING pooled LightGBM rebuilt at the s180/s181 config.

s181 measured the new component at 231.637 against the shipped 232.930 (UNCLIPPED) -- 1.29 s better,
out of time, 9x the seed floor. Unlike v37's fitted scalar (s174), what transfers here is a
RECIPE, re-applied to the ranking build by scripts/v38_pooled.py (pooled model only; the
per-airport loop and the classifier are untouched and read from the shipped artifact).

V38_W selects the blend:
   0.2,0.4,0.4  = v38, weights UNCHANGED. Isolates the component change. RECOMMENDED (s182).
   0.4,0.4,0.2  = v39, round weights favouring the strongest component. s182 records why this
                  is the direction the board already refused once: v37 scored BETTER on
                  validation than these weights do and lost 0.32 s.

Reconstruction and CONTROL are v37's, which is why they are trusted: with the OLD component
AND old weights the rebuild must reproduce quick-boat_v34.parquet byte-identically.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
SHA=os.environ.get('V38_SHA',''); OUT=os.environ.get('V38_OUT','submissions/quick-boat_v38.parquet')
W=np.array([float(x) for x in os.environ.get('V38_W','0.2,0.4,0.4').split(',')]); W=W/W.sum()
DATA=os.environ['DC26_DATA']; WORK=os.environ['DC26_WORK']
CAP,FLOOR,DAY=200000,60,86400.; SD_MIN,APT=50000.,'LIRF'
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
v29=pq.read_table('submissions/quick-boat_v29.parquet').to_pandas()
v34=pq.read_table('submissions/quick-boat_v34.parquet').to_pandas()
assert bool((v29.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
t=pd.read_parquet(f'{WORK}/v15_trees_jul19.parquet'); n=pd.read_parquet(f'{WORK}/v15_net_jul19.parquet')
m=t.merge(n,on='MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(v29.MVT_ID_mvt)
lgb_old,mlp_,pa_=m.lgbm.values,m.mlp.values,m.perapt.values
nw=pd.read_parquet(f'{WORK}/v38_lgbm_new.parquet').set_index('MVT_ID_mvt').reindex(v29.MVT_ID_mvt)
lgb_new=nw.lgbm_new.values
assert np.isfinite(lgb_new).all(), 'new component has NaNs after reindex'
print(f'new vs old component: mean {(lgb_new-lgb_old).mean():+.2f}  sd {(lgb_new-lgb_old).std():.2f}')
sd=m.sched_delay.values.astype(float); lb=m.mvt_minus_lobt.values.astype(float)
hit=np.isfinite(lb)&(lb>=7200)
copy=np.where(hit,np.clip(np.where(np.isfinite(lb),lb,sd),FLOOR,CAP),np.clip(sd,FLOOR,CAP))
p29=v29.TAXITIME_SEC_mvt.values.astype(float)
bo_=0.2*lgb_old+0.4*mlp_+0.4*pa_
den_=copy-bo_; solid_=np.abs(den_)>1.0
P_=np.clip(np.where(solid_,(p29-bo_)/np.where(solid_,den_,1.0),0.0),0.0,1.0)
_rec=np.clip(np.rint(np.where(solid_,P_*copy+(1-P_)*bo_,p29)),FLOOR,CAP)
FAITH=_rec.astype('int32')==p29.astype('int32')
def build(w,lgbsrc):
    bulk=w[0]*lgbsrc+w[1]*mlp_+w[2]*pa_
    out=np.where(FAITH&solid_, P_*copy+(1-P_)*bulk,
        np.where(FAITH, p29+(bulk-bo_), p29))
    return np.clip(np.rint(out),FLOOR,CAP),bulk
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt',
    'MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']).to_pandas()
for c in ('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt'): rk[c]=pd.to_datetime(rk[c],errors='coerce')
rd=rk[rk.PHASE_mvt=='DEP'].copy()
rd['sd']=(rd.MVT_TIME_UTC_mvt-rd.SCHED_TIME_UTC_mvt).dt.total_seconds()
cell=set(rd.MVT_ID_mvt[(rd.LOBT_flt.isna())&(rd.sd>=SD_MIN)&(rd.ADEP_mvt.astype(str)==APT)].values)
inC=v29.MVT_ID_mvt.isin(cell).values
def apply_roll(pred,bulk):
    return np.where(inC,np.clip(np.rint(0.40*(bulk+DAY)+0.60*p29),FLOOR,CAP),pred)
# ---- CONTROL: OLD component + OLD weights must reproduce v34 byte-identically ----
chk,bc=build(np.array([0.2,0.4,0.4]),lgb_old); chk=apply_roll(chk,bc)
d=int((chk.astype('int32')!=v34.TAXITIME_SEC_mvt.values.astype('int32')).sum())
print(f'CONTROL: old component + old weights differs from v34 on {d} of {len(chk):,} rows')
assert d==0, 'CONTROL FAILED -- reconstruction not faithful, do not ship'
print('  byte-identical. reconstruction verified.\n')
print(f'building with weights lgb/mlp/perapt = {W.round(4)} and the NEW component')
pred,bulk=build(W,lgb_new); pred=apply_roll(pred,bulk)
ch=int((pred.astype('int32')!=v34.TAXITIME_SEC_mvt.values.astype('int32')).sum())
dd=pred-v34.TAXITIME_SEC_mvt.values.astype(float)
print(f'vs v34: {ch:,} rows changed ({ch/len(pred):.1%}), mean {dd.mean():+.3f}s, sd {dd.std():.2f}s, |max| {np.abs(dd).max():,.0f}s')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':v29.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert bool((o.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False),OUT)
h=hashlib.sha256(open(OUT,'rb').read()).hexdigest()
print(f'\nwrote {OUT}\n  sha256 {h}')
if SHA: assert h==SHA, f'HASH MISMATCH expected {SHA} got {h}'; print('  sha asserted OK')
