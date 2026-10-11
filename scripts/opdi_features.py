"""Per-airframe turnaround features from OPDI registrations, for 2025 training and the 2026
scored set. Cached to $DC26_WORK/opdi_airframe.parquet keyed by MVT_ID_mvt.

Not a veto (s221 measured that at -68.9: the target is a bookkeeping difference and is not
bound by physics). These are FEATURES, so the model can learn where the record is physical
and where it is not.

`since_inblk` already exists in the feature set and approximates this from STAND_mvt, a proxy
s221 measured as wrong 99.45% of the time against the true registration's 0.097%.

Built with merge_asof on (airport, registration), which is the same backward search the veto
script did in a Python loop, vectorised."""
import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
W=os.environ['DC26_WORK']; OUT=f'{W}/opdi_airframe.parquet'
# TRAIN pointed at 09-01-2026-09-35-06_files_list, the PRE-REISSUE extract whose ranking.parquet
# holds a scored set that no longer exists. The twelve training_*.parquet are identical between
# the two extracts so nothing ever failed loudly, but a published script must not name a
# directory only one machine has. Both now default to the current extract. (2026-09-14)
DATA=os.environ['DC26_DATA']
TRAIN=os.environ.get('DC26_TRAIN', DATA)
EPOCH=pd.Timestamp('1970-01-01',tz='UTC')

# repo-relative, so the OPDI pull lands where scripts/pull_opdi.py puts it
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fls=sorted(glob.glob(os.path.join(os.environ.get('DC26_OPDI', f'{_ROOT}/data/opdi'),
                                  'flight_list_*.parquet')))
fl=pd.concat([pq.read_table(f,columns=['flt_id','dof','registration']).to_pandas() for f in fls],ignore_index=True)
fl['dof']=fl.dof.astype(str).str[:10]; fl['fid']=fl.flt_id.astype(str).str.strip().str.upper()
fl=fl.dropna(subset=['registration']).drop_duplicates(['fid','dof'])
lut=fl.set_index(['fid','dof']).registration
print(f'OPDI: {len(fls)} months, {len(lut):,} keyed flights, {fl.registration.nunique():,} airframes')

C=['MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt',
   'BLOCK_TIME_UTC_mvt','TAXITIME_SEC_mvt','FLIGHT_mvt','CALLSIGN_flt']
parts=[pq.read_table(f,columns=C).to_pandas() for f in sorted(glob.glob(f'{TRAIN}/training_*.parquet'))]
parts.append(pq.read_table(f'{DATA}/ranking.parquet',columns=C).to_pandas())
raw=pd.concat(parts,ignore_index=True).drop_duplicates('MVT_ID_mvt')
raw['t']=(pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True)-EPOCH).dt.total_seconds()
raw['bt']=(pd.to_datetime(raw.BLOCK_TIME_UTC_mvt,utc=True)-EPOCH).dt.total_seconds()
raw['st']=(pd.to_datetime(raw.SCHED_TIME_UTC_mvt,utc=True)-EPOCH).dt.total_seconds()
assert 1.73e9<raw.t.median()<1.80e9, f'epoch wrong {raw.t.median():.4g}'
raw['dof']=pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m-%d')
raw['apt']=raw.ADEP_mvt.where(raw.PHASE_mvt=='DEP',raw.ADES_mvt).astype(str)
reg=pd.Series(index=raw.index,dtype=object)
for col in (raw.CALLSIGN_flt, raw.FLIGHT_mvt):
    k=pd.MultiIndex.from_arrays([col.astype(str).str.strip().str.upper(), raw.dof])
    reg=reg.where(reg.notna(), pd.Series(lut.reindex(k).values,index=raw.index))
raw['reg']=reg
print(f'movements {len(raw):,}   with registration {raw.reg.notna().mean():.1%}')

dep=raw[(raw.PHASE_mvt=='DEP')&raw.reg.notna()].sort_values('t')
arr=(raw[(raw.PHASE_mvt=='ARR')&raw.reg.notna()&raw.bt.notna()]
     .rename(columns={'bt':'a_inblk','t':'a_land','TAXITIME_SEC_mvt':'a_taxiin'})
     .sort_values('a_inblk'))
m=pd.merge_asof(dep[['MVT_ID_mvt','apt','reg','t','st']],
                arr[['apt','reg','a_inblk','a_land','a_taxiin']],
                left_on='t', right_on='a_inblk', by=['apt','reg'], direction='backward', allow_exact_matches=False)
assert len(m)==len(dep), 'merge_asof changed row count'
m['af_since_arr']=m.t-m.a_inblk                    # take-off minus the airframe's on-block
m['af_sched_turn']=m.st-m.a_inblk                  # scheduled turnaround
m['af_prev_taxiin']=m.a_taxiin.astype(float)
m['af_has']=m.a_inblk.notna().astype('int8')
bad=(m.af_since_arr<0).sum(); assert bad==0, f'{bad} negative turnarounds: chain broken'
o=m[['MVT_ID_mvt','af_since_arr','af_sched_turn','af_prev_taxiin','af_has']]
o.to_parquet(OUT,index=False)
print(f'\nwrote {OUT}: {len(o):,} departures')
print(f'  chained {o.af_has.mean():.1%}   af_since_arr (h): ' +
      ' '.join(f'{q:.0%}={np.nanquantile(o.af_since_arr,q)/3600:,.1f}' for q in (.05,.25,.5,.75,.95)))
