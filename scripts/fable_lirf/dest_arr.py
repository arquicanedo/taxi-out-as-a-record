import pandas as pd, numpy as np, re
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
df=pd.read_parquet('all.parquet')
TEN=sorted(df.ADEP_mvt[df.PHASE_mvt=='DEP'].unique()); print('airports',TEN)
dep=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')].copy()
arr=df[(df.PHASE_mvt=='ARR')&(df.ADEP_mvt=='LIRF')&(df.ADES_mvt.isin(TEN))].copy()
print('LIRF deps', len(dep), 'to ten', dep.ADES_mvt.isin(TEN).sum(), '; arrival rows at ten airports from LIRF', len(arr))
# normalize callsign: ICAO3 + last5 pattern -> cannot invert exactly; match on suffix: digits part
def norm(s):
    if not isinstance(s,str): return None
    m=re.match(r'^([A-Z]{3})([A-Z]{2})([A-Z0-9]{3})$',s)
    if m and m.group(1)[1:]==m.group(2): return m.group(1)+'*'+m.group(3)   # ambiguous
    return s
dep['key']=dep.FLIGHT_mvt
arr['key']=arr.FLIGHT_mvt
# match on exact FLIGHT_mvt + ADES + arrival MVT within [dep MVT+15min, dep MVT+6h]
a=arr[['MVT_ID_mvt','FLIGHT_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','AOBT_3_flt','IOBT_flt','EOBT_1_flt','CALLSIGN_flt','AIRCRAFT_TYPE_flt','AIRCRAFT_OPERATOR_flt','set']].rename(columns=lambda c: 'a_'+c if c not in ('FLIGHT_mvt','ADES_mvt') else c)
m=dep.merge(a,on=['FLIGHT_mvt','ADES_mvt'],how='inner')
dt=(m.a_MVT_TIME_UTC_mvt-m.MVT_TIME_UTC_mvt).dt.total_seconds()
m=m[(dt>900)&(dt<6*3600)]
print('matched pairs', len(m), 'unique deps', m.MVT_ID_mvt.nunique(), 'dup deps', m.MVT_ID_mvt.duplicated().sum())
m=m.sort_values(['MVT_ID_mvt']).drop_duplicates('MVT_ID_mvt')
dep10=dep[dep.ADES_mvt.isin(TEN)]
print('match rate of LIRF->ten deps:', len(m)/len(dep10))
m['jn']=m.LOBT_flt.isna(); m['a_jn']=m.a_LOBT_flt.isna()
print(pd.crosstab(m.jn,m.a_jn))
# for train rows: copy status vs recovered AOBT_3
t=m[m.set=='train'].copy()
t['copy']=(t.BLOCK_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
t['ms']=(t.MVT_TIME_UTC_mvt-t.SCHED_TIME_UTC_mvt).dt.total_seconds()
t['a_aobt_null']=t.a_AOBT_3_flt.isna()
t['own_aobt_null']=t.AOBT_3_flt.isna()
t['a_ma']=(t.MVT_TIME_UTC_mvt-t.a_AOBT_3_flt).dt.total_seconds()
t['a_ml']=(t.MVT_TIME_UTC_mvt-t.a_LOBT_flt).dt.total_seconds()
print('join-null deps with recovered arrival flt join:', (t.jn&~t.a_jn).sum())
x=t[t.jn&~t.a_jn]
print(x.groupby(['a_aobt_null']).agg(n=('copy','size'),copy=('copy','mean'),ms_med=('ms','median')).to_string())
x2=x[x.ms>=3600]
print('coin-flip cell with recovered arrival join:', len(x2)); print(x2.groupby(['a_aobt_null']).agg(n=('copy','size'),copy=('copy','mean')).to_string())
print(x2[['src','FLIGHT_mvt','ADES_mvt','MVT_TIME_UTC_mvt','y','ms','copy','a_ml','a_ma','a_aobt_null','a_CALLSIGN_flt']].head(40).to_string())
# consistency check on join-present: own AOBT_3 vs arrival AOBT_3
jp=t[~t.jn&~t.a_jn]
print('join-present both: AOBT_3 equal', ((jp.AOBT_3_flt==jp.a_AOBT_3_flt)|(jp.AOBT_3_flt.isna()&jp.a_AOBT_3_flt.isna())).mean(), 'LOBT equal', (jp.LOBT_flt==jp.a_LOBT_flt).mean())
m.to_parquet('dep_destarr.parquet',index=False)
