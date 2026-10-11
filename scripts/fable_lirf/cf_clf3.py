import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); win=load_window(d)
p=pd.read_parquet('win_cf_pred.parquet'); win=win.merge(p,on='MVT_ID_mvt',how='left')
cf=win.p_clf.notna()
win['mix']=win.taxi_hat+win.p_clf*(win.ms-win.taxi_hat)
def blend_cf(mask, label):
    best={}
    for mon,oth in [(1,7),(7,1)]:
        idx=mask&(win.mon==oth); bs=np.arange(0,1.01,0.05); sse=[((win.y[idx]-(b*win.mix[idx]+(1-b)*win.newW[idx]))**2).sum() for b in bs]; best[mon]=bs[int(np.argmin(sse))]
    pred=win.newW.copy()
    for mon in [1,7]:
        idx=mask&(win.mon==mon); pred[idx]=best[mon]*win.mix[idx]+(1-best[mon])*win.newW[idx]
    g=report(f'{label} betas={best}', win, pred)
    for lo,hi in [(3600,7200),(7200,14400),(14400,45000)]:
        b=mask&(win.ms>=lo)&(win.ms<hi); print(f'    band {lo}-{hi}: n {b.sum()} gain {g[b].sum():.3e}  Jan {g[b&(win.mon==1)].sum():.3e} Jul {g[b&(win.mon==7)].sum():.3e}')
    return pred
blend_cf(cf&(win.ms<45000),'all <45000')
blend_cf(cf&(win.ms<14400),'ms<14400')
blend_cf(cf&(win.ms<7200),'ms<7200')
# fixed beta 0.5 for all bands (no fitting): 
pred=win.newW.copy(); m=cf&(win.ms<14400); pred[m]=0.5*win.mix[m]+0.5*win.newW[m]; report('fixed beta 0.5 ms<14400', win, pred)
pred=win.newW.copy(); m=cf&(win.ms<45000); pred[m]=0.5*win.mix[m]+0.5*win.newW[m]; report('fixed beta 0.5 ms<45000', win, pred)
# noise: bootstrap over days for the ms<14400 fixed-0.5 variant is in report; also per-day sign count
g=win.se-(win.y-pred)**2; pdv=g[m].groupby(win.day[m]).sum(); print('days with coin-flip rows', len(pdv), 'positive', (pdv>0).sum(), 'negative', (pdv<0).sum())
