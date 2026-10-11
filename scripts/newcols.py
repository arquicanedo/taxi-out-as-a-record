import pandas as pd, numpy as np, glob, pyarrow.parquet as pq, warnings; warnings.filterwarnings('ignore')
import features
D=features.DATA
f=sorted(glob.glob(f'{D}/training_*.parquet'))
raw=pd.concat([pq.read_table(x).to_pandas() for x in f],ignore_index=True).drop_duplicates('MVT_ID_mvt')
dep=features.build(raw)
extra=raw[['MVT_ID_mvt','FLIGHT_mvt','CALLSIGN_flt','ADES_flt','ADES_FILED_flt','AIRCRAFT_TYPE_flt','FLIGHT_RULE_mvt','FLIGHT_RULE_flt','ADES_mvt','AIRCRAFT_TYPE_mvt']]
d=dep.merge(extra,on='MVT_ID_mvt',how='left',suffixes=('','_x'))
for c in ['ADES_mvt','AIRCRAFT_TYPE_mvt']:
    if c+'_x' in d: d[c]=d[c].fillna(d[c+'_x'])
# --- cross-system disagreement: the LIRF insight, generalized ---
d['dis_ades_filed']=(d.ADES_flt.astype(str)!=d.ADES_FILED_flt.astype(str)).astype(float)
d['dis_ades']=(d.ADES_mvt.astype(str)!=d.ADES_flt.astype(str)).astype(float)
d['dis_type']=(d.AIRCRAFT_TYPE_mvt.astype(str)!=d.AIRCRAFT_TYPE_flt.astype(str)).astype(float)
d['dis_rule']=(d.FLIGHT_RULE_mvt.astype(str)!=d.FLIGHT_RULE_flt.astype(str)).astype(float)
d['dis_any']=d[['dis_ades_filed','dis_ades','dis_type']].max(axis=1)
# --- runway configuration and its switches ---
d=d.sort_values(['apt','MVT_TIME_UTC_mvt']).reset_index(drop=True)
d['bucket']=d.MVT_TIME_UTC_mvt.dt.floor('30min')
mode=d.groupby(['apt','bucket'],observed=True).RUNWAY_mvt.agg(lambda s:s.mode().iat[0] if len(s.mode()) else None).rename('cfg').reset_index()
mode['cfg_prev']=mode.groupby('apt',observed=True).cfg.shift(1)
mode['cfg_switch']=(mode.cfg.astype(str)!=mode.cfg_prev.astype(str)).astype(float)
mode['switch_run']=mode.groupby('apt',observed=True).cfg_switch.cumsum()
mode['since_switch']=mode.groupby(['apt','switch_run'],observed=True).cumcount().astype(float)
d=d.merge(mode[['apt','bucket','cfg','cfg_switch','since_switch']],on=['apt','bucket'],how='left')
d['rwy_is_cfg']=(d.RUNWAY_mvt.astype(str)==d.cfg.astype(str)).astype(float)   # off-config departure
# --- scheduled pressure: slot demand, independent of realised traffic ---
d['sbucket']=d.SCHED_TIME_UTC_mvt.dt.floor('15min')
d['sched_dens']=d.groupby(['apt','sbucket'],observed=True).MVT_ID_mvt.transform('size').astype(float)
d.to_parquet(f'{features.TMP}/dc26_newcols.parquet',index=False)
print('rows',len(d))
print(d[['dis_ades_filed','dis_ades','dis_type','dis_any','cfg_switch','since_switch','rwy_is_cfg','sched_dens']].describe().T.to_string())
print('\nflight-number cardinality:',d.FLIGHT_mvt.nunique(),' rows/flight',len(d)/d.FLIGHT_mvt.nunique())
