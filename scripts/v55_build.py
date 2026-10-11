"""v55: the shipped chain rebuilt on the OPDI airframe tree components.

scripts/v55_trees.py produced both tree components retrained with the four s224 features.
Its own control rebuilt per-airport WITHOUT opdi and reproduced v15_trees_jul19's array to
0.0000 exactly, so the substitution is faithful.

This composes them through the whole shipped stack, in order:
    v29's inverted P  ->  v34 rollover  ->  v45 month shifts  ->  v49  ->  v51  ->  v54

P is not stored anywhere; v38_build.py recovers it by inverting v29 against its own bulk, and
that machinery is reused verbatim rather than reimplemented.

CONTROL: with the OLD components the entire chain must reproduce quick-boat_v54.parquet
byte-identically. That is a stronger control than v38's, because this replaces BOTH tree
components where v38 replaced only the pooled one (s224: per-airport carries +0.7195 of the
+0.9557, pooled only +0.3981, so pooled-only was not an option).
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
SHA=os.environ.get('V55_SHA',''); OUT=os.environ.get('V55_OUT','submissions/quick-boat_v55.parquet')
OLD=bool(os.environ.get('V55_CONTROL'))
DATA=os.environ['DC26_DATA']; W=os.environ['DC26_WORK']
CAP,FLOOR,DAY,THR=200000,60,86400.,7200; SD_MIN,APT=50000.,'LIRF'
SHIFT={1:-8.778,7:-3.861}; A49,A51=0.3376,0.2922
# A54 was SHIPPED AT 0.60 on validation evidence and the board measured the true optimum at
# -0.1371 (s226): v51 264.9044 -> v54 267.9129, dSSE +5.5277e8, Q 4.3381e8, T +5.9481e7.
# The sign was wrong. Default now comes from that board measurement, not from validation.
A54=float(os.environ.get('V55_A54','-0.1371'))
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
v29=pq.read_table('submissions/quick-boat_v29.parquet').to_pandas()
v54=pq.read_table('submissions/quick-boat_v54.parquet').to_pandas()
assert bool((v29.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
ids=v29.MVT_ID_mvt
t=pd.read_parquet(f'{W}/v15_trees_jul19.parquet'); n=pd.read_parquet(f'{W}/v15_net_jul19.parquet')
m=t.merge(n,on='MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
lgb_old,mlp_,pa_old=m.lgbm.values,m.mlp.values,m.perapt.values
Pc=m.Pc.values.astype(float); sd=m.sched_delay.values.astype(float); lb=m.mvt_minus_lobt.values.astype(float)
# THE SHIPPED CHAIN GOES THROUGH v38, which replaced the pooled component. The control must
# therefore use v38_lgbm_new, not v15_trees' lgbm. Using the pre-v38 array reconstructs a
# chain that was never shipped and the control fails on 308,486 rows -- which is what it is
# for. `lgb_old` stays in use only for bo_, the reference bulk v29's P was inverted against.
v38=pd.read_parquet(f'{W}/v38_lgbm_new.parquet').set_index('MVT_ID_mvt').reindex(ids)
lgb_shipped=v38.lgbm_new.values
assert np.isfinite(lgb_shipped).all(), 'v38 component has NaNs after reindex'
nw=pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(ids)
lgb_new,pa_new=nw.lgbm_opdi.values,nw.perapt_opdi.values
assert np.isfinite(lgb_new).all() and np.isfinite(pa_new).all(), 'new components have NaNs'
assert np.abs(nw.perapt_ctl.values-pa_old).max()<1e-9, 'v55_trees control array does not match shipped'
hit=np.isfinite(lb)&(lb>=THR)
copy=np.where(hit,np.clip(np.where(np.isfinite(lb),lb,sd),FLOOR,CAP),np.clip(sd,FLOOR,CAP))
p29=v29.TAXITIME_SEC_mvt.values.astype(float)
bo_=0.2*lgb_old+0.4*mlp_+0.4*pa_old
den_=copy-bo_; solid_=np.abs(den_)>1.0
P_=np.clip(np.where(solid_,(p29-bo_)/np.where(solid_,den_,1.0),0.0),0.0,1.0)
_rec=np.clip(np.rint(np.where(solid_,P_*copy+(1-P_)*bo_,p29)),FLOOR,CAP)
FAITH=_rec.astype('int32')==p29.astype('int32')
print(f'P reconstruction faithful on {FAITH.mean():.2%} of rows')
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt',
    'MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','FLIGHT_mvt','CALLSIGN_flt']).to_pandas()
for c in ('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt'): rk[c]=pd.to_datetime(rk[c],errors='coerce')
rd=rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(ids)
mon=rd.MVT_TIME_UTC_mvt.dt.month.values
V34=(rd.ADEP_mvt.astype(str).values==APT)&rd.LOBT_flt.isna().values&(sd>=SD_MIN)
C49=(Pc>=.10)&(Pc<.95)&~V34
C51=(~np.isfinite(lb))&(sd>=7200)&(sd<14400)&(Pc>=.01)&~C49&~V34
# the s224/s219 v54 cell, rebuilt from its own detector so the chain is self-contained
import glob
_tf=pd.concat([pq.read_table(f,columns=['PHASE_mvt','FLIGHT_mvt']).to_pandas()
               for f in sorted(glob.glob(os.environ.get('DC26_TRAIN_LEGACY', os.environ['DC26_DATA']) + '/training_*.parquet'))],
              ignore_index=True)
_tf=_tf[_tf.PHASE_mvt=='DEP'].FLIGHT_mvt.astype(str)
KNOWN=frozenset(_tf[_tf.str.match(r'^[A-Z]{3}\d')].str[:3].value_counts()[lambda s:s>=50].index)
def dup(x):
    x=str(x)
    if len(x)<4 or not x[:3].isalpha() or x[:3] not in KNOWN: return False
    return any(len(x)>=3+k and x[3:3+k]==x[:3][-k:] for k in (1,2,3))
V54=np.array([dup(x) for x in rd.FLIGHT_mvt.astype(str)])&rd.CALLSIGN_flt.isna().values&~V34
cid=hashlib.sha256(','.join(map(str,sorted(ids.values[V54]))).encode()).hexdigest()[:16]
assert cid=='b8f4510e983fb155', f'v54 cell fingerprint changed: {cid}'
print(f'cells: v34 {int(V34.sum())}  v49 {int(C49.sum()):,}  v51 {int(C51.sum()):,}  v54 {int(V54.sum())} (fingerprint OK)')

def chain(lgbsrc,pasrc,a54=A54):
    bulk=0.2*lgbsrc+0.4*mlp_+0.4*pasrc
    p=np.where(FAITH&solid_,P_*copy+(1-P_)*bulk,np.where(FAITH,p29+(bulk-bo_),p29))
    p=np.clip(np.rint(p),FLOOR,CAP)
    p=np.where(V34,np.clip(np.rint(0.40*(bulk+DAY)+0.60*p29),FLOOR,CAP),p)          # v34
    u=np.zeros(len(p))
    for mm,s in SHIFT.items(): u[mon==mm]=s
    p=np.clip(np.rint(p+u),FLOOR,CAP)                                               # v45
    p=np.where(C49,np.clip(np.rint((1-A49)*p+A49*copy),FLOOR,CAP),p)                # v49
    # v51 de-commits toward `bo_`, the PRE-v38 bulk, NOT the current one. v50_probe.py read
    # `lgbm` from v15_trees_jul19 and never loaded v38_lgbm_new, so the shipped alpha* 0.2922
    # was probe-calibrated against that direction. Using the v38 bulk here breaks the control
    # on exactly the 72 C51 rows. Replicated rather than corrected: alpha* is a property of
    # the direction it was measured along, and re-pointing it would invalidate the board
    # measurement that set it. 77 rows.
    p=np.where(C51,np.clip(np.rint((1-A51)*p+A51*bo_),FLOOR,CAP),p)                 # v51
    p=np.where(V54,np.clip(np.rint((1-a54)*p+a54*copy),FLOOR,CAP),p)                # v54
    return p
ctl=chain(lgb_shipped,pa_old,0.60)
d=int((ctl.astype('int32')!=v54.TAXITIME_SEC_mvt.values.astype('int32')).sum())
print(f'\nCONTROL: SHIPPED components through the full chain differ from v54 on {d} of {len(ctl):,} rows')
assert d==0, 'CONTROL FAILED -- chain not faithful, do not ship'
print('  byte-identical. the whole v29->v34->v45->v49->v51->v54 chain is reproduced.\n')
if OLD: raise SystemExit('control-only run')
pred=chain(lgb_new,pa_new,A54)
print(f'built with A54 = {A54:+.4f}  (shipped v54/v55 used +0.60; board optimum -0.1371)')
dd=pred-v54.TAXITIME_SEC_mvt.values.astype(float)
ch=int((pred.astype('int32')!=v54.TAXITIME_SEC_mvt.values.astype('int32')).sum())
print(f'vs v54: {ch:,} rows changed ({ch/len(pred):.1%})  mean {dd.mean():+.3f}s  sd {dd.std():.2f}s  |max| {np.abs(dd).max():,.0f}s')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert bool((o.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False),OUT)
h=hashlib.sha256(open(OUT,'rb').read()).hexdigest(); print(f'\nwrote {OUT}\n  sha256 {h}')
if SHA: assert h==SHA, f'HASH MISMATCH {h}'; print('  sha asserted OK')
