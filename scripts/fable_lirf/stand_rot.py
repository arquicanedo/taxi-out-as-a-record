import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
df=pd.read_parquet('lirf.parquet')
a=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')].copy()
a['acopy']=(a.BLOCK_TIME_UTC_mvt-a.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
a['acar']=a.FLIGHT_mvt.str[:3]
a=a[['STAND_mvt','BLOCK_TIME_UTC_mvt','MVT_TIME_UTC_mvt','acopy','acar','AIRCRAFT_TYPE_mvt','FLIGHT_mvt']].rename(columns={'BLOCK_TIME_UTC_mvt':'a_inblk','MVT_TIME_UTC_mvt':'a_land','AIRCRAFT_TYPE_mvt':'a_type','FLIGHT_mvt':'a_flt'})
a=a.dropna(subset=['STAND_mvt']).sort_values('a_land')
d=load_lirf_dep()
dep=d.dropna(subset=['STAND_mvt']).sort_values('MVT_TIME_UTC_mvt').copy()
# previous arrival at the same stand: landing before dep MVT (use landing, not inblock, since inblock may be copied)
m=pd.merge_asof(dep, a, left_on='MVT_TIME_UTC_mvt', right_on='a_land', by='STAND_mvt', direction='backward', tolerance=pd.Timedelta('12h'))
m['gap_h']=(m.MVT_TIME_UTC_mvt-m.a_land).dt.total_seconds()/3600
m['same_car']=m.acar==m.car
print('deps', len(dep), 'with prev arrival at stand within 12h', m.a_land.notna().sum(), 'same carrier', m.same_car.sum())
print('gap hours quantiles', m.gap_h.quantile([.05,.25,.5,.75,.95]).round(2).to_dict())
t=m[m.set=='train']
cf=t[t.jn&(t.ms>=3600)&t.a_land.notna()]
print('coin-flip rows with prev arrival', len(cf)); print(pd.crosstab([cf.same_car,cf.acopy],cf['copy'],margins=True))
cf2=cf[cf.same_car&(cf.gap_h<6)]; print('same carrier, gap<6h:'); print(pd.crosstab(cf2.acopy,cf2['copy'],margins=True))
# all delayed departures (join-present too)
dl=t[(t.ms>=3600)&t.a_land.notna()&t.same_car]
print('all delayed deps same-carrier prev arrival:'); print(pd.crosstab(dl.acopy,dl['copy'],margins=True))
# all departures: lift
al=t[t.a_land.notna()&t.same_car&(t.gap_h<6)]
print('all deps same-carrier prev arrival gap<6h:'); print(pd.crosstab(al.acopy,al['copy'],margins=True, normalize='index'))
# conditional on carrier-month rate: logistic-ish check via stratified rates
al=al.copy(); al['cm']=al.car+'_'+al.mon.astype(str)
rate=al.groupby('cm')['copy'].transform('mean')
al['rb']=pd.cut(rate,[-.01,.05,.15,.3,.6,1.01])
print(al.groupby(['rb','acopy'],observed=True)['copy'].agg(['mean','size']).to_string())
m.to_parquet('dep_prevarr.parquet',index=False)

# (b) carrier-day arrival copy fraction
a2=df[(df.PHASE_mvt=='ARR')&(df.ADES_mvt=='LIRF')].copy()
a2['acopy']=(a2.BLOCK_TIME_UTC_mvt-a2.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60
a2['car']=a2.FLIGHT_mvt.str[:3]; a2['day']=a2.MVT_TIME_UTC_mvt.dt.floor('D')
cd=a2.groupby(['car','day']).acopy.agg(['mean','size']).rename(columns={'mean':'arr_day_rate','size':'arr_day_n'}).reset_index()
t2=t.merge(cd,on=['car','day'],how='left')
t2['cm']=t2.car+'_'+t2.mon.astype(str)
t2['cm_rate']=t2.groupby('cm')['copy'].transform('mean')
t2['rb']=pd.cut(t2.cm_rate,[-.01,.05,.15,.3,.6,1.01]); t2['ab']=pd.cut(t2.arr_day_rate,[-.01,.001,.2,.5,1.01])
for name,sel in [('join-present delayed ms>=1800',(~t2.jn)&(t2.ms>=1800)),('coin-flip',t2.jn&(t2.ms>=3600)),('all',t2.ms>-1e9)]:
    x=t2[sel&(t2.arr_day_n>=3)]
    print(name,'n',len(x)); print(x.groupby(['rb','ab'],observed=True)['copy'].agg(['mean','size']).unstack('ab').to_string())
