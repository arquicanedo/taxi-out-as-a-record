import pandas as pd, numpy as np
from base import *; from feats import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=build(); t=d[d.set=='train'].copy(); win=load_window(d)
cw=pd.read_parquet('cf_clf_win.parquet')[['MVT_ID_mvt','p_clf']]
win=win.merge(cw,on='MVT_ID_mvt',how='left')   # keeps row order; p_clf NaN outside coin-flip
# taxi_hat: median real-class y by (hour band) from other months, LIRF
t['hb']=(t.hour//3); win['hb']=(win.hour//3)
cf=win.p_clf.notna()
print('coin-flip window rows', cf.sum())
win['wimp']=((win.newW-900)/(win.ms-900)).clip(0,1)
taxi=pd.Series(np.nan,index=win.index)
for mon in [1,7]:
    tr=t[(t.mon!=mon)&(t.cls=='real')]
    med=tr.groupby('hb').y.median()
    idx=(win.mon==mon); taxi[idx]=win.hb[idx].map(med)
win['taxi_hat']=taxi
print('taxi_hat by hb', taxi[cf].describe().round(0).to_dict())
noroll=cf&(win.ms<45000)
print('rows changed (ms<45000):', noroll.sum(), 'SSE newW on them', win.se[noroll].sum())
for alpha in [0.5,1.0]:
    pred=win.newW.copy(); pred[noroll]=win.newW[noroll]+alpha*(win.p_clf[noroll]-win.wimp[noroll])*(win.ms[noroll]-win.taxi_hat[noroll])
    report(f'(i) weight shift alpha={alpha}', win, pred)
pred=win.newW.copy(); pred[noroll]=win.taxi_hat[noroll]+win.p_clf[noroll]*(win.ms[noroll]-win.taxi_hat[noroll])
g=report('(ii) direct mixture', win, pred)
print('   Jan', g[win.mon==1].sum(), 'Jul', g[win.mon==7].sum())
# (iii) blend direct mixture with newW, beta cross-fitted Jan<->Jul
mix=pred.copy()
best={}
for mon,oth in [(1,7),(7,1)]:
    idx=noroll&(win.mon==oth)
    bs=np.arange(0,1.01,0.05); sse=[((win.y[idx]-(b*mix[idx]+(1-b)*win.newW[idx]))**2).sum() for b in bs]
    best[mon]=bs[int(np.argmin(sse))]; print('fit on',oth,'best beta',best[mon])
pred3=win.newW.copy()
for mon in [1,7]:
    idx=noroll&(win.mon==mon); pred3[idx]=best[mon]*mix[idx]+(1-best[mon])*win.newW[idx]
g=report('(iii) cross-fitted blend', win, pred3)
# per-row view of the biggest changes
ch=win[noroll].assign(pred=pred[noroll]); ch['gain']=ch.se-(ch.y-ch.pred)**2
print(ch.sort_values('gain')[['src','FLIGHT_mvt','car','y','newW','pred','ms','wimp','p_clf','cls']].head(12).to_string())
print(ch.sort_values('gain',ascending=False)[['src','FLIGHT_mvt','car','y','newW','pred','ms','wimp','p_clf','cls']].head(12).to_string())
print('gain>0 rows', (ch.gain>0).sum(), 'gain<0', (ch.gain<0).sum())
win[['MVT_ID_mvt','taxi_hat','wimp','p_clf']].to_parquet('win_cf_pred.parquet',index=False)
