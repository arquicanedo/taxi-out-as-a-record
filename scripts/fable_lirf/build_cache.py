import os
import pandas as pd, numpy as np, glob, time
D=(os.environ['DC26_DATA'] + '/')
t0=time.time()
fs=sorted(glob.glob(D+'training_2025-*.parquet'))
dfs=[]
for f in fs:
    d=pd.read_parquet(f); d['src']=f.split('/')[-1][9:16]; dfs.append(d)
tr=pd.concat(dfs,ignore_index=True); tr['set']='train'
rk=pd.read_parquet(D+'ranking.parquet'); rk['src']='2026'; rk['set']='rank'
df=pd.concat([tr,rk],ignore_index=True)
print('rows',len(df), 'train',len(tr),'rank',len(rk), time.time()-t0)
# units check
for c in ['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','AOBT_3_flt','ARVT_1_flt','ARVT_3_flt']:
    df[c]=pd.to_datetime(df[c],utc=True)
mv=df.loc[df.set=='train','MVT_TIME_UTC_mvt'].astype('int64')/1e9
# timestamp[us] -> int64 is microseconds; convert properly
mv_s=df.loc[df.set=='train','MVT_TIME_UTC_mvt'].astype('int64')/1e6
print('median epoch s (train):', np.median(mv_s), ' (bad unit test /1e9 ->', np.median(mv), ')')
assert 1.735e9<np.median(mv_s)<1.77e9
df['y']=(df.MVT_TIME_UTC_mvt-df.BLOCK_TIME_UTC_mvt).dt.total_seconds()
chk=(df.y-df.TAXITIME_SEC_mvt).abs()
print('y vs TAXITIME mismatch (labelled rows):', (chk[df.TAXITIME_SEC_mvt.notna()]>1).sum())
df.to_parquet('all.parquet',index=False)
L=df[(df.ADEP_mvt=='LIRF')|(df.ADES_mvt=='LIRF')]
L.to_parquet('lirf.parquet',index=False)
print('LIRF rows',len(L), L.groupby(['set','PHASE_mvt']).size())
print('scored LIRF deps', ((rk.ADEP_mvt=='LIRF')&(rk.PHASE_mvt=='DEP')).sum())
print(time.time()-t0)
