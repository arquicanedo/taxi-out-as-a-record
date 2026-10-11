import os
import pandas as pd, numpy as np
from base import *; from feats import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=build(); t=d[d.set=='train'].copy(); win=load_window(d)
t['hb']=t.hour//3; win['hb']=win.hour//3
m=pd.read_parquet('dep_destarr.parquet')[['MVT_ID_mvt','a_LOBT_flt','a_AOBT_3_flt']]
win=win.merge(m,on='MVT_ID_mvt',how='left'); win['a_ml']=(win.MVT_TIME_UTC_mvt-win.a_LOBT_flt).dt.total_seconds(); win['a_ma']=(win.MVT_TIME_UTC_mvt-win.a_AOBT_3_flt).dt.total_seconds()
cfp=pd.read_parquet('win_cf_pred.parquet'); win=win.merge(cfp,on='MVT_ID_mvt',how='left')
grp=lambda c: 'tzcar' if c in ('BEL','AEA','AEZ','AEE','LZB','ETH') else 'rest'
win['grp']=win.car.map(grp); t['grp']=t.car.map(grp)
ext=win.y>=7200
def step(name, prev, new):
    g=(win.y-prev)**2-(win.y-new)**2; ci=boot_days(g,win.day)
    print(f'{name}: marginal SSE removed {g.sum():.4e} [{ci[0]:.3e}, {ci[1]:.3e}]  ordinary {g[~ext].sum():.4e}  extreme {g[ext].sum():.4e}  rows changed {(new!=prev).sum()}')
    return new
p0=win.newW.copy()
# L1: join bound clip
lo=win.ml-TOL; hi=win.ml+TOL
p1=p0.clip(lower=lo,upper=hi).where(~win.jn,p0); p1=step('L1 join-bound clip',p0,p1)
# L2: destination-arrival recovery (join-null rows with recovered LOBT)
sel=win.jn&win.a_ml.notna()&win.FLIGHT_mvt.notna(); far=sel&(win.a_ml>TOL); near=sel&(win.a_ml<=TOL)
p2=p1.copy(); p2[far]=np.where(win.a_ma[far].notna(),win.a_ma[far],win.taxi_hat[far].fillna(1000)); 
# near: y > a_ml+TOL; copy if ms > a_ml+TOL else tz-like (taxi+off)
p2[near]=np.where(win.ms[near]>win.a_ml[near]+TOL, win.ms[near], 1000+win.off[near])
p2=step('L2 dest-arrival recovery',p1,p2)
# L3: TZ cell A (jn & ms<1800), excluding rows already fixed by L2
cellA=win.jn&(win.ms<1800)&~sel
p3=p2.copy()
for mon in [1,7]:
    tr=t[(t.mon!=mon)&t.jn&(t.ms<1800)]
    for g in ['tzcar','rest']:
        s_=tr[tr.grp==g]; p_tz=(s_.cls=='tz').mean(); p_copy=(s_.cls=='copy').mean(); p_real=1-p_tz-p_copy
        idx=cellA&(win.mon==mon)&(win.grp==g)
        p3[idx]=p_real*1000+p_tz*(1000+win.off[idx])+p_copy*win.ms[idx].clip(lower=60)
p3=step('L3 TZ cell mixture',p2,p3)
# L4: roll impossible for jn & 14400<=ms<43200 -> cap at ms
idx=win.jn&(win.ms>=14400)&(win.ms<43200)&(p3>win.ms); p4=p3.copy(); p4[idx]=win.ms[idx]; p4=step('L4 roll-impossible cap',p3,p4)
# L5: coin-flip classifier blend, beta=0.5, ms<14400, excluding L2 rows
cf=win.p_clf.notna()&(win.ms<14400)&~sel
mix=win.taxi_hat+win.p_clf*(win.ms-win.taxi_hat)
p5=p4.copy(); p5[cf]=0.5*mix[cf]+0.5*p4[cf]; p5=step('L5 coin-flip clf blend (beta 0.5, ms<14400)',p4,p5)
# cross-fitted beta variant for honesty
best={}
for mon,oth in [(1,7),(7,1)]:
    idx=cf&(win.mon==oth); bs=np.arange(0,1.01,0.05); sse=[((win.y[idx]-(b*mix[idx]+(1-b)*p4[idx]))**2).sum() for b in bs]; best[mon]=bs[int(np.argmin(sse))]
p5x=p4.copy()
for mon in [1,7]:
    idx=cf&(win.mon==mon); p5x[idx]=best[mon]*mix[idx]+(1-best[mon])*p4[idx]
step(f'L5x cross-fitted beta {best}',p4,p5x)
g=(win.y-p0)**2-(win.y-p5)**2; ci=boot_days(g,win.day)
print(f'TOTAL L1-L5 vs newW: {g.sum():.4e} [{ci[0]:.3e}, {ci[1]:.3e}] ordinary {g[~ext].sum():.4e} extreme {g[ext].sum():.4e}; Jan {g[win.mon==1].sum():.4e} Jul {g[win.mon==7].sum():.4e}')
g=(win.y-p0)**2-(win.y-p5x)**2; ci=boot_days(g,win.day)
print(f'TOTAL L1-L5x vs newW: {g.sum():.4e} [{ci[0]:.3e}, {ci[1]:.3e}] ordinary {g[~ext].sum():.4e} extreme {g[ext].sum():.4e}')
print('LIRF window RMSE newW', np.sqrt(win.se.mean()), '-> L1-L5', np.sqrt(((win.y-p5)**2).mean()))
print('moved distance window: L5 rows', ((p5-p4)**2)[cf].sum(), ' L3 rows', ((p3-p2)**2).sum(), ' L2 rows', ((p2-p1)**2).sum())
win.assign(p_final=p5)[['MVT_ID_mvt','y','newW','p_final']].to_parquet('win_levers.parquet',index=False)
# ---------------- 2026 application ----------------
r=d[d.set=='rank'].copy(); r['hb']=r.hour//3
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'}); r=r.merge(sub,on='MVT_ID_mvt')
r=r.merge(m,on='MVT_ID_mvt',how='left'); r['a_ml']=(r.MVT_TIME_UTC_mvt-r.a_LOBT_flt).dt.total_seconds(); r['a_ma']=(r.MVT_TIME_UTC_mvt-r.a_AOBT_3_flt).dt.total_seconds()
c26=pd.read_parquet('cf_clf_2026.parquet')[['MVT_ID_mvt','p_clf']]; r=r.merge(c26,on='MVT_ID_mvt',how='left')
med=t[t.cls=='real'].groupby('hb').y.median(); r['taxi_hat']=r.hb.map(med); r['grp']=r.car.map(grp)
q0=r.pred.astype(float).copy()
lo=r.ml-TOL; hi=r.ml+TOL; q1=q0.clip(lower=lo,upper=hi).where(~r.jn,q0); print('2026 L1 rows', (q1!=q0).sum(), 'floor', ((q1-q0)**2).sum())
sel=r.jn&r.a_ml.notna()&r.FLIGHT_mvt.notna(); far=sel&(r.a_ml>TOL); near=sel&(r.a_ml<=TOL)
q2=q1.copy(); q2[far]=np.where(r.a_ma[far].notna(),r.a_ma[far],r.taxi_hat[far]); q2[near]=np.where(r.ms[near]>r.a_ml[near]+TOL, r.ms[near], 1000+r.off[near]); print('2026 L2 rows', (q2!=q1).sum(), 'moved', ((q2-q1)**2).sum())
cellA=r.jn&(r.ms<1800)&~sel; q3=q2.copy()
for g_ in ['tzcar','rest']:
    s_=t[t.jn&(t.ms<1800)&(t.grp==g_)]; p_tz=(s_.cls=='tz').mean(); p_copy=(s_.cls=='copy').mean(); p_real=1-p_tz-p_copy
    idx=cellA&(r.grp==g_); q3[idx]=p_real*1000+p_tz*(1000+r.off[idx])+p_copy*r.ms[idx].clip(lower=60)
print('2026 L3 rows', (q3!=q2).sum(), 'moved', ((q3-q2)**2).sum(), ' tzcar rows', (cellA&(r.grp=='tzcar')).sum())
idx=r.jn&(r.ms>=14400)&(r.ms<43200)&(q3>r.ms); q4=q3.copy(); q4[idx]=r.ms[idx]; print('2026 L4 rows', (q4!=q3).sum(), 'moved', ((q4-q3)**2).sum())
cf=r.p_clf.notna()&(r.ms<14400)&~sel; mix=r.taxi_hat+r.p_clf*(r.ms-r.taxi_hat); q5=q4.copy(); q5[cf]=0.5*mix[cf]+0.5*q4[cf]; print('2026 L5 rows', (q5!=q4).sum(), 'moved', ((q5-q4)**2).sum())
print('2026 total moved distance', ((q5-q0)**2).sum())
r.assign(p_final=q5)[['MVT_ID_mvt','pred','p_final']].to_parquet('rank_levers_lirf.parquet',index=False)
