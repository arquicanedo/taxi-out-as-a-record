import os
import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
TOL=3606
df=pd.read_parquet('all.parquet')
d=df[(df.PHASE_mvt=='DEP')].copy()
z=np.load(os.path.join(os.environ['DC26_WORK'], 'adsblol', 'cand', 'v121_window.npz'))
w=pd.DataFrame({'MVT_ID_mvt':z['win_ids'],'newW':z['newW']})
win=d[d.set=='train'].merge(w,on='MVT_ID_mvt',how='inner')
print('window rows', len(win))
win['ml']=(win.MVT_TIME_UTC_mvt-win.LOBT_flt).dt.total_seconds()
win['lo']=win.ml-TOL; win['hi']=win.ml+TOL
jp=win.LOBT_flt.notna()
print('window join-present', jp.sum(), ' y outside bound:', ((win.y<win.lo)|(win.y>win.hi))[jp].sum())
win['cl']=np.where(jp, win.newW.clip(lower=win.lo,upper=win.hi), win.newW)
win['se']=(win.y-win.newW)**2; win['se2']=(win.y-win.cl)**2
win['day']=win.MVT_TIME_UTC_mvt.dt.floor('D')
win['ext']=win.y>=7200
def boot(x, days, B=2000, seed=0):
    rng=np.random.default_rng(seed); ud=days.unique(); per=x.groupby(days).sum().reindex(ud).values
    bs=np.array([per[rng.integers(0,len(ud),len(ud))].sum() for _ in range(B)])
    return np.percentile(bs,[2.5,97.5])
for apt,g in [('ALL',win)]+[(a,win[win.ADEP_mvt==a]) for a in sorted(win.ADEP_mvt.unique())]:
    gain=g.se-g.se2; nch=(g.clip!=g.newW).sum()
    ci=boot(gain,g.day)
    print(f'{apt}: changed {nch} rows, SSE removed {gain.sum():.4e} [{ci[0]:.3e},{ci[1]:.3e}]  ordinary {gain[~g.ext].sum():.4e} extreme {gain[g.ext].sum():.4e}')
L=win[win.ADEP_mvt=='LIRF']
ch=L[L.cl!=L.newW].copy(); ch['gain']=ch.se-ch.se2
print(ch.sort_values('gain',ascending=False)[['src','FLIGHT_mvt','ADES_mvt','MVT_TIME_UTC_mvt','y','newW','cl','ml','lo','hi','gain']].head(30).to_string())
print('LIRF changed rows: gain>0', (ch.gain>0).sum(), 'gain<0', (ch.gain<0).sum())
# 2026
r=d[d.set=='rank'].copy()
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(sub,on='MVT_ID_mvt',how='left'); assert r.pred.notna().all()
r['ml']=(r.MVT_TIME_UTC_mvt-r.LOBT_flt).dt.total_seconds(); r['lo']=r.ml-TOL; r['hi']=r.ml+TOL
jp=r.LOBT_flt.notna()
r['cl']=np.where(jp, r.pred.clip(lower=r.lo,upper=r.hi), r.pred)
r['floor']=(r.pred-r.cl)**2
print('2026: join-present', jp.sum(), 'of', len(r), '; v121 preds outside bound:', (r.cl!=r.pred).sum(), '; guaranteed SSE floor removed:', r.floor.sum())
print(r.groupby('ADEP_mvt').agg(n=('pred','size'),out=('floor',lambda s:(s>0).sum()),floor=('floor','sum')).to_string())
print(r[r.floor>0].sort_values('floor',ascending=False)[['ADEP_mvt','FLIGHT_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','pred','cl','ml','floor']].head(30).to_string())
r[['MVT_ID_mvt','ADEP_mvt','pred','cl','ml','floor']].to_parquet('rank_clip.parquet',index=False)
