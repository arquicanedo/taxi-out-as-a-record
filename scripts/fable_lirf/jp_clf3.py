import pandas as pd, numpy as np
from base import *
d=load_lirf_dep(); win=load_window(d)
p=pd.read_parquet('win_jp_pred.parquet'); win=win.merge(p,on='MVT_ID_mvt',how='left')
s=win.p_jp.notna()&(win.ms<45000)
lo=win.ml-TOL; hi=win.ml+TOL
win['mix']=(win.taxi_hat+win.p_jp*(win.ms-win.taxi_hat)).clip(lower=lo,upper=hi)
base=win.newW.clip(lower=lo,upper=hi).where(~win.jn, win.newW)   # L1-clipped baseline
print('L1 baseline SSE', ((win.y-base)**2).sum(), 'newW SSE', win.se.sum())
for beta in [0.25,0.5,1.0]:
    pred=base.copy(); pred[s]=beta*win.mix[s]+(1-beta)*base[s]
    g=(win.y-base)**2-(win.y-pred)**2; ext=win.y>=7200; ci=boot_days(g,win.day)
    print(f'jp mixture beta={beta} vs L1 baseline: removed {g.sum():.4e} [{ci[0]:.3e},{ci[1]:.3e}] ordinary {g[~ext].sum():.4e} extreme {g[ext].sum():.4e}')
hi_=s&(win.p_jp>=0.5); pred=base.copy(); pred[hi_]=win.mix[hi_]; g=(win.y-base)**2-(win.y-pred)**2; print(f'jp P>=0.5 only (n={hi_.sum()}): removed {g.sum():.4e}')
