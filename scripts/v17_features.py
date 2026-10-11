"""Shared feature builder for v17. ONE definition, used by the ranking pipeline
and the validation measurement alike, so the two cannot drift.

  block_dev     s77, -1.08
  sched_vs_hr   s78, -1.80   (the CTOT/ATFM slot proxy)
  eight more    s82, -1.43 jointly
"""
import pandas as pd, numpy as np, glob, sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import features
import pyarrow.parquet as pq

COL={'MVT':'MVT_TIME_UTC_mvt','SCHED':'SCHED_TIME_UTC_mvt','LOBT':'LOBT_flt','IOBT':'IOBT_flt',
     'EOBT':'EOBT_1_flt','ARVT1':'ARVT_1_flt','AOBT3':'AOBT_3_flt','ARVT3':'ARVT_3_flt'}
PAIRS=[('LOBT','AOBT3'),('LOBT','EOBT'),('IOBT','AOBT3'),('IOBT','EOBT'),
       ('SCHED','LOBT'),('SCHED','IOBT'),('ARVT1','AOBT3')]        # s82, delta < -0.15
ATFM=['atfm_reg_share','atfm_late_share','atfm_early_share','atfm_in_share',
      'atfm_cause_S','atfm_cause_W','atfm_cause_G','atfm_dly_per_arr']   # s86, -1.00 jointly
_LADDER={
    'v15': ['block_dev'],                                                  # s77
    'v16': ['block_dev','sched_vs_hr'],                                    # +s78
    'v17': ['block_dev','sched_vs_hr']+[f'p_{a}_{b}' for a,b in PAIRS]+['k_rwy'],   # +s82
    'v18': ['block_dev','sched_vs_hr']+[f'p_{a}_{b}' for a,b in PAIRS]+['k_rwy']+ATFM,  # +s86
    'v19': ['block_dev','sched_vs_hr']+[f'p_{a}_{b}' for a,b in PAIRS]+['k_rwy']+ATFM+['k_q15'],
}
# DC26_RECIPE picks a rung of the version ladder, so a fixed model can be rebuilt
# against a different extract. Needed 2026-09-04: the July re-extract changed the
# scored set from 215,876 to 344,841 rows, and measuring that shift requires
# holding the model constant while the data moves.
_R=os.environ.get('DC26_RECIPE','v18')
assert _R in _LADDER, f'DC26_RECIPE={_R!r} not in {sorted(_LADDER)}'
NEW=_LADDER[_R]
ALL=_LADDER['v19']

def build():
    U=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','RUNWAY_mvt']+list(COL.values())
    src=sorted(glob.glob(f'{features.DATA}/training_*.parquet'))+[f'{features.DATA}/ranking.parquet']
    d=pd.concat([pq.read_table(f,columns=U).to_pandas() for f in src],ignore_index=True).drop_duplicates('MVT_ID_mvt')
    d=d[d.PHASE_mvt=='DEP'].copy()
    S=lambda a,b:(pd.to_datetime(d[COL[a]])-pd.to_datetime(d[COL[b]])).dt.total_seconds()
    d['block_dev']=(S('ARVT3','AOBT3')-S('ARVT1','EOBT')).values
    d['dep_delay']=S('AOBT3','EOBT').values
    t=pd.to_datetime(d[COL['MVT']])
    d['_per']=(t.dt.year*100+t.dt.month); d['_apt']=d.ADEP_mvt.astype(str); d['_hk']=t.dt.floor('h')
    g=d.groupby(['_apt','_hk','_per'],observed=True).dep_delay
    d['sched_vs_hr']=d.dep_delay.values-np.where(g.transform('size')>=5,g.transform('median'),np.nan)
    gr=d.groupby(['_apt','RUNWAY_mvt','_hk','_per'],observed=True).dep_delay
    d['k_rwy']=d.dep_delay.values-np.where(gr.transform('size')>=5,gr.transform('median'),np.nan)
    # s82 measured k_q15 at +0.11 on the j3 window and rejected it as "too fine".
    # The adversarial run re-measured it on the all window at -0.52, confirmed
    # -0.38 on disjoint seeds. The flip is mechanism-consistent: the ATFM slot
    # compliance window is 15 minutes, and the airports July added are the
    # regulation-heavy ones. Same construction as sched_vs_hr, the one block
    # whose board transfer measured 0.869.
    d['_q15']=t.dt.floor('15min')
    gq=d.groupby(['_apt','_q15','_per'],observed=True).dep_delay
    d['k_q15']=d.dep_delay.values-np.where(gq.transform('size')>=5,gq.transform('median'),np.nan)
    for a,b in PAIRS: d[f'p_{a}_{b}']=S(a,b).values
    # --- s86: EUROCONTROL PRU daily ATFM regulation data, open licence ---
    R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','reference','pru')
    sl=pd.read_parquet(os.path.join(R,'ATFM_Slot_Adherence.parquet'))
    sl['_d']=pd.to_datetime(sl.FLT_DATE,errors='coerce').dt.normalize()
    reg=sl.FLT_DEP_REG_1.replace(0,np.nan)
    sl['atfm_reg_share']=sl.FLT_DEP_REG_1/sl.FLT_DEP_1.replace(0,np.nan)
    sl['atfm_late_share']=sl.FLT_DEP_OUT_LATE_1/reg
    sl['atfm_early_share']=sl.FLT_DEP_OUT_EARLY_1/reg
    sl['atfm_in_share']=sl.FLT_DEP_IN_1/reg
    sl=sl[['_d','APT_ICAO','atfm_reg_share','atfm_late_share','atfm_early_share','atfm_in_share']]
    ar=pd.read_parquet(os.path.join(R,'Airport_Arrival_ATFM_Delay.parquet'))
    ar['_d']=pd.to_datetime(ar.FLT_DATE,errors='coerce').dt.normalize()
    tot=ar.DLY_APT_ARR_1.replace(0,np.nan)
    for cc in ('S','W','G'): ar[f'atfm_cause_{cc}']=ar[f'DLY_APT_ARR_{cc}_1']/tot
    ar['atfm_dly_per_arr']=ar.DLY_APT_ARR_1/ar.FLT_ARR_1.replace(0,np.nan)
    ar=ar[['_d','APT_ICAO','atfm_cause_S','atfm_cause_W','atfm_cause_G','atfm_dly_per_arr']]
    P=sl.merge(ar,on=['_d','APT_ICAO'],how='outer')
    d['_d']=pd.to_datetime(d[COL['MVT']]).dt.normalize().dt.tz_localize(None)
    d['APT_ICAO']=d.ADEP_mvt.astype(str)
    n0=len(d); d=d.merge(P,on=['_d','APT_ICAO'],how='left')
    assert len(d)==n0, 'PRU join fanned out'
    return d[['MVT_ID_mvt']+NEW]   # every column above is computed; NEW selects the rung
