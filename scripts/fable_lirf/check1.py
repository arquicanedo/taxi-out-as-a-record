import pandas as pd, numpy as np
df=pd.read_parquet('lirf.parquet')
dep=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')]
arr=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')]
print('LIRF dep', dep.groupby('set').size().to_dict(), 'arr', arr.groupby('set').size().to_dict())
lab=dep[dep.set=='train']
print('dep y vs TAXITIME mismatch', ((lab.y-lab.TAXITIME_SEC_mvt).abs()>1).sum(), 'null TAXI', lab.TAXITIME_SEC_mvt.isna().sum())
# arrivals: taxi-in = BLOCK - MVT ?
a=arr[arr.TAXITIME_SEC_mvt.notna()]
yin=(a.BLOCK_TIME_UTC_mvt-a.MVT_TIME_UTC_mvt).dt.total_seconds()
print('arr taxi-in mismatch', ((yin-a.TAXITIME_SEC_mvt).abs()>1).sum(), 'of', len(a))
print('rank dep TAXI null', dep[dep.set=='rank'].TAXITIME_SEC_mvt.isna().sum(), 'BLOCK null', dep[dep.set=='rank'].BLOCK_TIME_UTC_mvt.isna().sum())
print('rank arr TAXI null', arr[arr.set=='rank'].TAXITIME_SEC_mvt.isna().sum(), 'of', (arr.set=='rank').sum())
print(lab.y.describe())
print('copies (|BLOCK-SCHED|<=60):', ((lab.BLOCK_TIME_UTC_mvt-lab.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60).mean())
print('extremes y>=7200:', (lab.y>=7200).sum(), 'y>=80000', (lab.y>=80000).sum(), 'y<0', (lab.y<0).sum())
print(lab.columns.tolist())
print(lab.iloc[0])
