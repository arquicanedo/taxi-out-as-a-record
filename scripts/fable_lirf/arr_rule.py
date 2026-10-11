import pandas as pd, numpy as np
df=pd.read_parquet('all.parquet')
a=df[(df.PHASE_mvt=='ARR')].copy()
q=[0,.0001,.001,.5,.999,.9999,1]
for s in ['train','rank']:
    x=a[(a.set==s)&a.LOBT_flt.notna()]
    print(s,'arrivals join-present',len(x),'of',(a.set==s).sum())
    for b in ['ARVT_1_flt','ARVT_3_flt','LOBT_flt','AOBT_3_flt']:
        v=(x.BLOCK_TIME_UTC_mvt-x[b]).dt.total_seconds()
        print('  BLOCK -',b, v.quantile(q).round(0).astype(int).to_dict())
    for b in ['ARVT_1_flt','ARVT_3_flt']:
        v=(x.MVT_TIME_UTC_mvt-x[b]).dt.total_seconds()
        print('  MVT -',b, v.quantile(q).round(0).astype(int).to_dict())
# 2026 departures: MVT-LOBT distribution join-present vs 2025
for s in ['train','rank']:
    x=df[(df.PHASE_mvt=='DEP')&(df.set==s)&df.LOBT_flt.notna()]
    v=(x.MVT_TIME_UTC_mvt-x.LOBT_flt).dt.total_seconds()
    print(s,'DEP MVT-LOBT', v.quantile(q).round(0).astype(int).to_dict(), 'n',len(x), 'jn rate', 1-len(x)/((df.PHASE_mvt=='DEP')&(df.set==s)).sum())
