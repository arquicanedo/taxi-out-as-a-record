import os
import pandas as pd, numpy as np, lightgbm as lgb, sys
from base import *; from feats import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=build(); t=d[d.set=='train'].copy(); win=load_window(d)
FE=['ms','hour','dow','cross','pat_bad','n_cf','n_del','frac_del','car_day_dla','car_day_dla_n','car_day_arr','car_day_arr_n','car_mon_arr','car_day_cf','car_day_del','dest2_cat','stand1_cat','RUNWAY_mvt_cat',
    'car_del_copy','car_cf_copy','car_q','car_r','car_bayes','car_all_copy']
params=dict(objective='binary',learning_rate=0.03,num_leaves=7,min_data_in_leaf=20,feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=6,seed=0)
def fit_predict(tr, te, seeds=5, rounds=300):
    cr=carrier_rates(tr, tr[tr.cfc]); Xtr=pd.concat([tr[tr.cfc][[f for f in FE if not f.startswith('car_') or f in ('car_day_dla','car_day_dla_n','car_day_arr','car_day_arr_n','car_mon_arr','car_day_cf','car_day_del')]],cr],axis=1)[FE]
    # NOTE: carrier rates for the training rows are in-sample (leak); use nested month-out for training rows
    Xtr=Xtr.copy()
    for mon in tr.mon.unique():
        sub=tr[tr.mon!=mon]; idx=tr[tr.cfc&(tr.mon==mon)].index
        cr2=carrier_rates(sub, tr.loc[idx]); Xtr.loc[idx,cr2.columns]=cr2.values
    ytr=tr[tr.cfc]['copy'].astype(int)
    cr_te=carrier_rates(tr, te); Xte=pd.concat([te[[f for f in FE if f not in cr_te.columns]],cr_te],axis=1)[FE]
    p=np.zeros(len(te))
    for s in range(seeds):
        pr=dict(params); pr['seed']=s
        m=lgb.train(pr, lgb.Dataset(Xtr,ytr), num_boost_round=rounds)
        p+=m.predict(Xte)/seeds
    return p, m
# leave-one-month-out over 2025
t['p_clf']=np.nan
for mon in range(1,13):
    tr=t[t.mon!=mon]; te=t[(t.mon==mon)&t.cfc]
    p,m=fit_predict(tr,te); t.loc[te.index,'p_clf']=p
cf=t[t.cfc]; yb=cf['copy'].astype(float)
print('LOMO Brier clf', ((cf.p_clf-yb)**2).mean())
# baseline: carrier cf rate cross-fitted
print('by month Brier:'); print(cf.groupby('mon').apply(lambda g: pd.Series({'n':len(g),'copy':g['copy'].mean(),'brier':((g.p_clf-g['copy'])**2).mean()})).to_string())
imp=pd.Series(m.feature_importance('gain'),index=FE).sort_values(ascending=False); print(imp.round(0).to_string())
# window comparison vs model wimp
w=win.merge(cf[['MVT_ID_mvt','p_clf']],on='MVT_ID_mvt'); w=w[w.cfc]
w['wimp']=((w.newW-900)/(w.ms-900)).clip(0,1)
yb=w['copy'].astype(float)
print('window n',len(w),'Brier wimp',((w.wimp-yb)**2).mean(),'Brier clf',((w.p_clf-yb)**2).mean(), 'Brier avg', (((w.wimp+w.p_clf)/2-yb)**2).mean())
# SSE via weight shift on non-roll rows (ms<45000)
for alpha in [0.25,0.5,0.75,1.0]:
    pred=win.newW.copy(); idx=w.index
    shift=alpha*(w.p_clf-w.wimp)*(w.ms-900); shift[w.ms>=45000]=0
    pred.loc[idx]=w.newW+shift
    report(f'coin-flip clf weight shift alpha={alpha}', win, pred)
# 2026 application: v121 implied weights vs clf
r=d[d.set=='rank'].copy(); rc=r[r.cfc]
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet'))).rename(columns={'TAXITIME_SEC_mvt':'pred'})
p26,_=fit_predict(t, rc)
rc=rc.merge(sub,on='MVT_ID_mvt'); rc['p_clf']=p26; rc['wimp']=((rc.pred-900)/(rc.ms-900)).clip(0,1)
print('2026 coin-flip rows', len(rc)); print(rc.groupby('mon').agg(n=('ms','size'),wimp=('wimp','mean'),p_clf=('p_clf','mean')).to_string())
rc['shift']=(rc.p_clf-rc.wimp)*(rc.ms-900); rc.loc[rc.ms>=45000,'shift']=0
print('2026 sum shift^2 (distance moved at alpha=1):', (rc['shift']**2).sum())
rc[['MVT_ID_mvt','mon','ms','pred','wimp','p_clf','shift']].to_parquet('cf_clf_2026.parquet',index=False)
w[['MVT_ID_mvt','mon','y','newW','ms','copy','wimp','p_clf']].to_parquet('cf_clf_win.parquet',index=False)
