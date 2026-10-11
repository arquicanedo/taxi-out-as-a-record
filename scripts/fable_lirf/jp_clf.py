import pandas as pd, numpy as np
from base import *; from feats import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=build(); t=d[d.set=='train'].copy(); win=load_window(d)
# join-present delayed rows: copy structure
jp=t[~t.jn&(t.ms>=1800)].copy()
jp['lsb']=pd.cut(jp.ls,[-1e9,-1,60,900,1800,3600,1e9]); jp['b_arr']=pd.cut(jp.car_day_arr,[-.01,.001,.2,.5,1.01]); jp['b_cf']=pd.cut(jp.n_cf,[-1,1,3,9,100])
print(jp.groupby('lsb',observed=True)['copy'].agg(['mean','size']).to_string())
print(jp.groupby(['lsb','b_arr'],observed=True)['copy'].agg(['mean','size']).unstack('b_arr').to_string())
print(jp.groupby(['lsb','b_cf'],observed=True)['copy'].agg(['mean','size']).unstack('b_cf').to_string())
# window: SSE on join-present delayed copies and the model's implied weight
w=win[~win.jn&(win.ms>=1800)].copy(); w['wimp']=((w.newW-900)/(w.ms-900)).clip(0,1)
print('window jp delayed rows', len(w), 'SSE', w.se.sum(), 'copies', w['copy'].sum(), 'SSE copies', w.se[w['copy']].sum())
w['lsb']=pd.cut(w.ls,[-1e9,-1,60,900,1800,3600,1e9])
print(w.groupby('lsb',observed=True).agg(n=('se','size'),copy=('copy','mean'),wimp=('wimp','mean'),SSE=('se','sum')).to_string())
# oracle: perfect copy knowledge on jp delayed rows (copy->ms; real->newW if newW<2500 else 1000)
pred=w.newW.copy(); pred[w['copy']]=w.ms[w['copy']]; pred[~w['copy']]=np.where(w.newW[~w['copy']]<2500,w.newW[~w['copy']],1000)
print('oracle removed on jp delayed:', (w.se-(w.y-pred)**2).sum())
