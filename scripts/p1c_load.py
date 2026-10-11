# Load all raw 2025 training files + ranking file into one frame with epoch-second clocks.
import pandas as pd, numpy as np, glob, time, os
t0=time.time()
D=os.environ['DC26_DATA'].rstrip('/')+'/'
OUT=os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/trackC/'
os.makedirs(OUT,exist_ok=True)   # a fresh DC26_SESSION has no trackC/ (clean-room find, 2026-09-23)
TS=['MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt','IOBT_flt','EOBT_1_flt','ARVT_1_flt','AOBT_3_flt','ARVT_3_flt']
SHORT={'MVT_TIME_UTC_mvt':'t_mvt','BLOCK_TIME_UTC_mvt':'t_blk','SCHED_TIME_UTC_mvt':'t_sch','LOBT_flt':'t_lobt','IOBT_flt':'t_iobt','EOBT_1_flt':'t_eobt','ARVT_1_flt':'t_arvt1','AOBT_3_flt':'t_aobt','ARVT_3_flt':'t_arvt3'}
frames=[]
files=sorted(glob.glob(D+'training_*.parquet'))+[D+'ranking.parquet']
for i,f in enumerate(files):
    d=pd.read_parquet(f)
    d['src']= i  # 0..11 = training months 1..12, 12 = ranking
    for c in TS:
        s=pd.to_datetime(d[c],utc=True)
        v=(s - pd.Timestamp('1970-01-01',tz='UTC')).dt.total_seconds()
        d[SHORT[c]]=v.astype('float64')
        d.drop(columns=[c],inplace=True)
    d['rowpos']=np.arange(len(d))
    frames.append(d)
    print(f.split('/')[-1],len(d),round(time.time()-t0,1),flush=True)
df=pd.concat(frames,ignore_index=True)
del frames
med=np.nanmedian(df.t_mvt.values)
print('median t_mvt',med); assert 1.735e9<med<1.80e9
for c in ['FLIGHT_mvt','FLIGHT_RULE_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','CALLSIGN_flt','ADEP_flt','ADES_flt','ADES_FILED_flt','MARKET_SEGMENT_flt','FLIGHT_RULE_flt','FLIGHT_TYPE_flt','AIRCRAFT_TYPE_flt','WK_TBL_CAT_flt','AIRCRAFT_OPERATOR_flt']:
    df[c]=df[c].astype('string')
df['y']=df['TAXITIME_SEC_mvt'].astype('float64')
df.drop(columns=['TAXITIME_SEC_mvt'],inplace=True)
df['dep']=(df.PHASE_mvt=='DEP')
df['apt']=np.where(df.dep,df.ADEP_mvt,df.ADES_mvt)
print(df.groupby(['src','PHASE_mvt']).size().unstack())
print('y null by src/phase'); print(df.assign(yn=df.y.isna()).groupby(['src','PHASE_mvt']).yn.mean().unstack())
# check y consistency on labelled rows
lab=df.y.notna()&df.t_blk.notna()
dd=df[lab]
calc=np.where(dd.dep,dd.t_mvt-dd.t_blk,dd.t_blk-dd.t_mvt)
print('y == computed (dep/arr) frac', np.mean(np.abs(calc-dd.y.values)<0.5))
df.to_parquet(OUT+'p1c_raw.parquet',index=False)
print('saved',df.shape,round(time.time()-t0,1))
