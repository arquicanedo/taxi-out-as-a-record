import os
import pandas as pd, numpy as np, lightgbm as lgb
from base import *; from feats import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=build(); t=d[d.set=='train'].copy(); win=load_window(d)
t['hb']=t.hour//3
sel=lambda x: (~x.jn)&(x.ms>=1800)
FE=['ms','ls','ma','hour','dow','cross','pat_bad','n_cf','n_del','frac_del','car_day_dla','car_day_dla_n','car_day_arr','car_day_arr_n','car_mon_arr','car_day_cf','car_day_del','dest2_cat','stand1_cat','RUNWAY_mvt_cat','aobt_null',
    'car_del_copy','car_all_copy']
t['aobt_null']=t.AOBT_3_flt.isna().astype(int); d['aobt_null']=d.AOBT_3_flt.isna().astype(int)
params=dict(objective='binary',learning_rate=0.05,num_leaves=15,min_data_in_leaf=50,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=6,seed=0)
def fit_predict(tr, te, seeds=3, rounds=300):
    trs=tr[sel(tr)]; Xtr=trs[[f for f in FE if not f.startswith('car_') or f in ('car_day_dla','car_day_dla_n','car_day_arr','car_day_arr_n','car_mon_arr','car_day_cf','car_day_del')]].copy()
    for c in ['car_del_copy','car_all_copy']: Xtr[c]=np.nan
    for mon in tr.mon.unique():
        sub=tr[tr.mon!=mon]; idx=trs[trs.mon==mon].index
        cr=carrier_rates(sub, trs.loc[idx]); Xtr.loc[idx,['car_del_copy','car_all_copy']]=cr[['car_del_copy','car_all_copy']].values
    Xtr=Xtr[FE]; ytr=trs['copy'].astype(int)
    cr=carrier_rates(tr, te); Xte=pd.concat([te[[f for f in FE if f not in ('car_del_copy','car_all_copy')]],cr[['car_del_copy','car_all_copy']]],axis=1)[FE]
    p=np.zeros(len(te))
    for s in range(seeds):
        pr=dict(params); pr['seed']=s; m=lgb.train(pr, lgb.Dataset(Xtr,ytr), num_boost_round=rounds); p+=m.predict(Xte)/seeds
    return p,m
t['p_jp']=np.nan
for mon in range(1,13):
    tr=t[t.mon!=mon]; te=t[(t.mon==mon)&sel(t)]
    p,m=fit_predict(tr,te); t.loc[te.index,'p_jp']=p
jp=t[sel(t)]; yb=jp['copy'].astype(float)
print('LOMO Brier jp clf', ((jp.p_jp-yb)**2).mean(), 'base rate Brier', (yb.var()))
imp=pd.Series(m.feature_importance('gain'),index=FE).sort_values(ascending=False); print(imp.round(0).head(15).to_string())
w=win.merge(jp[['MVT_ID_mvt','p_jp']],on='MVT_ID_mvt',how='left'); w['hb']=w.hour//3
s=w.p_jp.notna(); w['wimp']=((w.newW-900)/(w.ms-900)).clip(0,1)
yb=w['copy'].astype(float)
for lo in [1800,3600,7200]:
    m_=s&(w.ms>=lo); print(f'window jp ms>={lo}: n {m_.sum()} Brier wimp {((w.wimp-yb)**2)[m_].mean():.4f} clf {((w.p_jp-yb)**2)[m_].mean():.4f}')
# taxi_hat
taxi=pd.Series(np.nan,index=w.index)
for mon in [1,7]:
    med=t[(t.mon!=mon)&(t.cls=='real')].groupby('hb').y.median(); idx=(w.mon==mon); taxi[idx]=w.hb[idx].map(med)
w['taxi_hat']=taxi
for lo in [1800,3600]:
    m_=s&(w.ms>=lo)
    pred=w.newW.copy(); pred[m_]=w.taxi_hat[m_]+w.p_jp[m_]*(w.ms[m_]-w.taxi_hat[m_])
    g=report(f'jp direct mixture ms>={lo}', w, pred)
    mix=pred.copy(); best={}
    for mon,oth in [(1,7),(7,1)]:
        idx=m_&(w.mon==oth); bs=np.arange(0,1.01,0.05); sse=[((w.y[idx]-(b*mix[idx]+(1-b)*w.newW[idx]))**2).sum() for b in bs]; best[mon]=bs[int(np.argmin(sse))]
    pred3=w.newW.copy()
    for mon in [1,7]:
        idx=m_&(w.mon==mon); pred3[idx]=best[mon]*mix[idx]+(1-best[mon])*w.newW[idx]
    print('   betas',best); report(f'jp cross-fitted blend ms>={lo}', w, pred3)
    # only act where clf is confident: P>=0.5
    hi=m_&(w.p_jp>=0.5); pred4=w.newW.copy(); pred4[hi]=w.taxi_hat[hi]+w.p_jp[hi]*(w.ms[hi]-w.taxi_hat[hi]); report(f'jp mixture only P>=0.5 ms>={lo} (n={hi.sum()})', w, pred4)
w[['MVT_ID_mvt','p_jp','taxi_hat']].to_parquet('win_jp_pred.parquet',index=False)
# 2026
r=d[d.set=='rank'].copy(); rs=r[sel(r)]; p26,_=fit_predict(t, rs)
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'})
rs=rs.merge(sub,on='MVT_ID_mvt'); rs['p_jp']=p26; rs['wimp']=((rs.pred-900)/(rs.ms-900)).clip(0,1)
print('2026 jp delayed rows', len(rs)); print(rs.groupby('mon').agg(n=('ms','size'),wimp=('wimp','mean'),p_jp=('p_jp','mean'),hiP=('p_jp',lambda s:(s>=0.5).sum())).to_string())
rs[['MVT_ID_mvt','mon','ms','pred','wimp','p_jp']].to_parquet('jp_clf_2026.parquet',index=False)
