import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
df=pd.read_parquet('all.parquet')
d=df[(df.PHASE_mvt=='DEP')&(df.set=='train')].copy()
jp=d[d.LOBT_flt.notna()]
q=[0,.0001,.001,.01,.5,.99,.999,.9999,1]
for a,b in [('BLOCK_TIME_UTC_mvt','LOBT_flt'),('BLOCK_TIME_UTC_mvt','EOBT_1_flt'),('BLOCK_TIME_UTC_mvt','IOBT_flt'),('BLOCK_TIME_UTC_mvt','AOBT_3_flt'),('MVT_TIME_UTC_mvt','LOBT_flt'),('SCHED_TIME_UTC_mvt','LOBT_flt'),('SCHED_TIME_UTC_mvt','IOBT_flt'),('MVT_TIME_UTC_mvt','AOBT_3_flt')]:
    x=(jp[a]-jp[b]).dt.total_seconds()
    print(f'{a[:5]}-{b[:6]}: n={x.notna().sum()} ', x.quantile(q).round(0).astype(int).to_dict())
# per airport max |BLOCK-LOBT|
x=(jp.BLOCK_TIME_UTC_mvt-jp.LOBT_flt).dt.total_seconds()
print(x.abs().groupby(jp.ADEP_mvt).agg(['max',lambda s: s.quantile(.999)]).to_string())
x=(jp.MVT_TIME_UTC_mvt-jp.LOBT_flt).dt.total_seconds()
print('MVT-LOBT per airport'); print(x.groupby(jp.ADEP_mvt).agg(['min','max']).to_string())
# join-null rate by airport and by ms band
d['jn']=d.LOBT_flt.isna(); d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['msb']=pd.cut(d.ms,[-1e9,-600,600,1800,3600,7200,14400,1e9])
print(d.groupby(['ADEP_mvt']).jn.mean().round(4).to_dict())
print(d[d.ADEP_mvt=='LIRF'].groupby('msb',observed=True).agg(n=('jn','size'),jn=('jn','mean')).to_string())
# callsign equality for join-present: FLIGHT_mvt == CALLSIGN_flt ?
print('LIRF join-present FLIGHT_mvt==CALLSIGN_flt:', (jp[jp.ADEP_mvt=='LIRF'].FLIGHT_mvt==jp[jp.ADEP_mvt=='LIRF'].CALLSIGN_flt).mean())
print('estate join-present FLIGHT_mvt==CALLSIGN_flt by airport:', (jp.FLIGHT_mvt==jp.CALLSIGN_flt).groupby(jp.ADEP_mvt).mean().round(3).to_dict())
