import os
import pandas as pd, numpy as np, re
TOL=3606
def load_lirf_dep():
    df=pd.read_parquet('lirf.parquet')
    d=df[(df.PHASE_mvt=='DEP')&(df.ADEP_mvt=='LIRF')].copy()
    d['bs']=(d.BLOCK_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['ml']=(d.MVT_TIME_UTC_mvt-d.LOBT_flt).dt.total_seconds()
    d['ma']=(d.MVT_TIME_UTC_mvt-d.AOBT_3_flt).dt.total_seconds()
    d['ls']=(d.LOBT_flt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['jn']=d.LOBT_flt.isna()
    loc=d.MVT_TIME_UTC_mvt.dt.tz_convert('Europe/Rome'); sl=d.SCHED_TIME_UTC_mvt.dt.tz_convert('Europe/Rome')
    d['off']=loc.map(lambda t: t.utcoffset().total_seconds())
    d['cross']=(loc.dt.normalize()-sl.dt.normalize()).dt.days
    d['hour']=loc.dt.hour; d['dow']=loc.dt.dayofweek
    d['mon']=d.MVT_TIME_UTC_mvt.dt.month; d['day']=d.MVT_TIME_UTC_mvt.dt.floor('D')
    d['car']=d.FLIGHT_mvt.str[:3]
    def pat(s):
        if not isinstance(s,str): return 'null'
        m=re.match(r'^([A-Z]{3})([A-Z]{2})(.*)$',s)
        if m and m.group(1)[1:]==m.group(2): return 'dbl'
        if re.match(r'^[A-Z]{3}\d',s): return 'icao'
        return 'other'
    d['pat']=d.FLIGHT_mvt.map(pat)
    d['copy']=d.bs.abs()<=60
    d['roll']=d.y>=80000
    d['tz']=((d.y-d.off)>=200)&((d.y-d.off)<=3000)&(d.bs<-1500)
    d['cls']=np.where(d['copy'],'copy',np.where(d.roll,'roll',np.where(d.tz,'tz',np.where((d.y>=200)&(d.y<=3600),'real','other'))))
    d.loc[d.set=='rank','cls']=None
    return d
def load_window(d):
    z=np.load(os.path.join(os.environ['DC26_WORK'], 'adsblol', 'cand', 'v121_window.npz'))
    w=pd.DataFrame({'MVT_ID_mvt':z['win_ids'],'newW':z['newW']})
    win=d[d.set=='train'].merge(w,on='MVT_ID_mvt',how='inner').copy()
    assert len(win)==26528
    win['se']=(win.y-win.newW)**2
    return win
def boot_days(gain, days, B=2000, seed=0):
    rng=np.random.default_rng(seed); per=gain.groupby(days).sum(); v=per.values; n=len(v)
    bs=np.array([v[rng.integers(0,n,n)].sum() for _ in range(B)])
    return np.percentile(bs,[2.5,97.5])
def report(name, win, pred):
    se2=(win.y-pred)**2; gain=win.se-se2; ext=win.y>=7200
    ci=boot_days(gain,win.day)
    print(f'{name}: SSE removed {gain.sum():.4e} [{ci[0]:.3e}, {ci[1]:.3e}]  ordinary {gain[~ext].sum():.4e}  extreme {gain[ext].sum():.4e}  rows changed {(pred!=win.newW).sum()}')
    return gain
