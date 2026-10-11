"""v90 = v84 with the LIRF cell shrunk 30% toward s369's rarity discriminator.

s369: flight-number rarity (appearances per 30 days, scale-free within pool) splits the cell where
the carrier cannot -- 89.47% copy at the rare end against 37.24% at the frequent end, WITHIN the
four coin-flip carriers that are 48% of the scored cell.

s370 priced it: trained on the 433 cell rows of the ten non-scored months, predicting the 174 of
Jan+Jul, cell AUC 0.8256, and the gain/Q table clears s367's bar of 1.92 once shrunk.

ALPHA = 0.30, chosen because it clears the bar by 2x, not because it maximises anything. At full
strength the ratio is 0.48 and it would lose exactly as v87 (1.09) and v89 (1.15) did.

    target = P*copy + (1-P)*blend        blend = the shipped 0.2/0.4/0.4 bulk (s365: correct here,
    v90    = v84 + ALPHA*(target - v84)  median 1,423 s against a truth of 1,091 s)

CONTROL: ALPHA=0 reproduces v84's values exactly.
"""
import hashlib, os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgbm
D=os.environ['DC26_DATA']
W=os.environ['DC26_WORK']
ALPHA=float(os.environ.get('ALPHA','0.30')); OUT=os.environ.get('V90_OUT','submissions/quick-boat_v90.parquet')
THR,FLOOR,CAP=7200.,60.,200000.
C=['MVT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt',
   'SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','LOBT_flt','STAND_mvt']
def prep(df):
    d=df[df.PHASE_mvt.eq('DEP')&df.ADEP_mvt.eq('LIRF')].copy()
    for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']: d[c]=pd.to_datetime(d[c],errors='coerce',utc=True)
    d['y']=d.TAXITIME_SEC_mvt.astype(float) if 'TAXITIME_SEC_mvt' in d else np.nan
    d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['ln']=d.LOBT_flt.isna(); d['flt']=d.FLIGHT_mvt.astype(str); d['car']=d.flt.str.slice(0,3)
    d['dbl']=d.flt.str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d').fillna(False).astype(float)
    d['hour']=d.MVT_TIME_UTC_mvt.dt.hour; d['dow']=d.MVT_TIME_UTC_mvt.dt.dayofweek
    d['sp']=d.STAND_mvt.astype(str).str.slice(0,1); d['des']=d.ADES_mvt.astype(str).str.slice(0,2)
    cov=float(d.MVT_TIME_UTC_mvt.dt.normalize().nunique())     # scale-free within pool
    g=d.groupby('flt').ln.agg(['sum','size']); d=d.join(g.rename(columns={'sum':'fs','size':'fn'}),on='flt')
    d['flt_rate']=d.fn/cov*30.0; d['flt_jfail']=(d.fs-d.ln)/(d.fn-1).clip(lower=1)
    s=d.groupby('STAND_mvt').ln.agg(['sum','size']); d=d.join(s.rename(columns={'sum':'ss','size':'sn'}),on='STAND_mvt')
    d['stand_jfail']=(d.ss-d.ln)/(d.sn-1).clip(lower=1)
    print(f'    pool: {len(d):,} LIRF departures over {cov:.0f} covered days')
    return d
print('train pool (all 2025):'); tr=prep(pd.concat([pq.read_table(f,columns=C).to_pandas() for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True))
print('score pool (2026 file):'); rk=prep(pq.read_table(f'{D}/ranking.parquet',columns=C).to_pandas())
cell=lambda p: p[p.ln&(p.ms>=7200)&(p.ms<50000)].copy()
CT,CR=cell(tr),cell(rk)
CT['cp']=(np.abs(CT.y-CT.ms)<=60).astype(int)
print(f'  train cell {len(CT)} rows, copy {CT.cp.mean():.2%}   score cell {len(CR)} rows')
NUM=['flt_rate','flt_jfail','stand_jfail','ms','hour','dow','dbl']; CATS=['car','sp','des']
for c in CATS:
    u=pd.Categorical(pd.concat([CT[c],CR[c]]).astype(str)).categories
    CT[c]=pd.Categorical(CT[c].astype(str),categories=u); CR[c]=pd.Categorical(CR[c].astype(str),categories=u)
F=NUM+CATS
P=np.mean([lgbm.LGBMClassifier(n_estimators=250,learning_rate=.05,num_leaves=4,min_child_samples=20,
    subsample=.8,colsample_bytree=.8,reg_lambda=2.,verbose=-1,random_state=s)
    .fit(CT[F],CT.cp,categorical_feature=CATS).predict_proba(CR[F])[:,1] for s in range(5)],axis=0)
t=pd.read_parquet(f'{W}/v15_trees_jul19.parquet').merge(
  pd.read_parquet(f'{W}/v15_net_jul19.parquet'),on='MVT_ID_mvt').set_index('MVT_ID_mvt')
bl=np.clip((0.2*t.lgbm+0.4*t.mlp+0.4*t.perapt).reindex(CR.MVT_ID_mvt.values).values,FLOOR,CAP)
cop=np.clip(CR.ms.values,FLOOR,CAP)
tgt=np.clip(P*cop+(1-P)*bl,FLOOR,CAP)
v84=pq.read_table('submissions/quick-boat_v84.parquet').to_pandas()
cur=pd.Series(v84.TAXITIME_SEC_mvt.values.astype(float),index=v84.MVT_ID_mvt.values)
a=cur.reindex(CR.MVT_ID_mvt.values).values
delta=pd.Series(ALPHA*(tgt-a),index=CR.MVT_ID_mvt.values).reindex(v84.MVT_ID_mvt.values).fillna(0.).values
p=np.clip(np.rint(v84.TAXITIME_SEC_mvt.values.astype(float)+delta),FLOOR,CAP)
print(f'\nALPHA {ALPHA}   rows moved {int((p!=v84.TAXITIME_SEC_mvt.values).sum())}   '
      f'Q {np.sum(delta**2):.4e}   max|delta| {np.abs(delta).max():,.0f} s')
print(f'  P on the scored cell: mean {P.mean():.3f}  >=0.8 on {int((P>=.8).sum())}  <=0.2 on {int((P<=.2).sum())}')
tpl=pq.read_table(f'{D}/submitting.parquet')
o=pd.DataFrame({'MVT_ID_mvt':v84.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':p.astype('int32')})
tb=pa.Table.from_pandas(o,schema=tpl.schema,preserve_index=False); assert tb.schema.equals(tpl.schema)
pq.write_table(tb,OUT); print(f'wrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
if ALPHA==0.0:
    ref=pq.read_table('submissions/quick-boat_v84.parquet').to_pandas()
    print(f'  CONTROL ALPHA=0: values {"IDENTICAL" if bool((ref.values==o.values).all()) else "DIFFER -- FAIL"}')
    assert bool((ref.values==o.values).all())
