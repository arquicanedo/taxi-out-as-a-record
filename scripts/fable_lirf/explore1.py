import pandas as pd, numpy as np, re
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
L=df[((df.ADEP_mvt=='LIRF')&(df.PHASE_mvt=='DEP'))|((df.ADES_mvt=='LIRF')&(df.PHASE_mvt=='ARR'))].copy()
dep=L[L.PHASE_mvt=='DEP']; arr=L[L.PHASE_mvt=='ARR']
print('FLIGHT_ID null', L.FLIGHT_ID_mvt.isna().mean(), 'unique', L.FLIGHT_ID_mvt.nunique(), 'rows', len(L))
vc=L.FLIGHT_ID_mvt.value_counts()
print('FLIGHT_ID multiplicity', vc.value_counts().head())
# shared between ARR and DEP?
sh=set(dep.FLIGHT_ID_mvt.dropna())&set(arr.FLIGHT_ID_mvt.dropna())
print('FLIGHT_ID shared ARR/DEP', len(sh))
# relation MVT_ID vs FLIGHT_ID
d=dep[dep.set=='train'].sort_values('MVT_TIME_UTC_mvt')
print(d[['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','SCHED_TIME_UTC_mvt','MVT_TIME_UTC_mvt','y']].head(15).to_string())
print('corr rank MVT_ID vs FLIGHT_ID', d.MVT_ID_mvt.corr(d.FLIGHT_ID_mvt,method='pearson'))
print('corr rank FLIGHT_ID vs SCHED', d.FLIGHT_ID_mvt.corr(d.SCHED_TIME_UTC_mvt.astype('int64'),method='pearson'))
print('corr rank MVT_ID vs SCHED', d.MVT_ID_mvt.corr(d.SCHED_TIME_UTC_mvt.astype('int64'),method='pearson'))
# join-null
d['jn']=d.LOBT_flt.isna()
d['copy']=(d.BLOCK_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
print('join-null rate', d.jn.mean(), 'copy rate by jn', d.groupby('jn')['copy'].mean().to_dict())
# callsign pattern: doubled prefix like ITYTY680
def pat(s):
    if not isinstance(s,str): return 'null'
    m=re.match(r'^([A-Z]{3})([A-Z]{2})(.*)$',s)
    if m and m.group(1)[1:]==m.group(2): return 'dbl'
    if re.match(r'^[A-Z]{3}\d',s): return 'icao'
    if re.match(r'^[A-Z0-9]{2}\d',s): return 'iata'
    return 'other'
d['pat']=d.FLIGHT_mvt.map(pat)
print(d.groupby(['jn','pat']).agg(n=('y','size'),copy=('copy','mean'),ext=('y',lambda s:(s>=7200).mean())).to_string())
# null patterns in mvt columns
for c in ['AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','FLIGHT_RULE_mvt','FLIGHT_ID_mvt','ADES_mvt']:
    d['n_'+c]=d[c].isna()
print(d.groupby(['jn','n_AIRCRAFT_TYPE_mvt','n_STAND_mvt','n_RUNWAY_mvt','n_FLIGHT_RULE_mvt']).agg(n=('y','size'),copy=('copy','mean')).to_string())
# the coin-flip cell: jn & ms>=3600
cf=d[d.jn&(d.ms>=3600)]
print('coin-flip cell', len(cf), 'copy', cf['copy'].mean())
print(cf.groupby('pat').agg(n=('y','size'),copy=('copy','mean')).to_string())
print(cf.groupby('src').agg(n=('y','size'),copy=('copy','mean')).to_string())
