import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); t=d[d.set=='train'].copy(); win=load_window(d)
# cell A: join-null & ms<1800
cellA=t[t.jn&(t.ms<1800)]
print('cell A (jn & ms<1800):', len(cellA)); print(cellA.cls.value_counts()); print(cellA.groupby('car').cls.value_counts().unstack(fill_value=0).sort_values('tz',ascending=False).head(15).to_string())
# cross-fitted class probs per month (global, since tiny), by carrier group BEL/AEA/AEZ/AEE vs rest
grp=lambda c: 'tzcar' if c in ('BEL','AEA','AEZ','AEE','LZB','ETH') else 'rest'
t['grp']=t.car.map(grp); win['grp']=win.car.map(grp)
w=win[win.jn&(win.ms<1800)].copy(); print('window cell A rows', len(w), 'SSE', w.se.sum()); print(w[['src','FLIGHT_mvt','y','newW','ms','off','cls','grp']].to_string())
pred=win.newW.copy()
for mon in [1,7]:
    tr=t[(t.mon!=mon)&t.jn&(t.ms<1800)]
    for g in ['tzcar','rest']:
        s=tr[tr.grp==g]; p_tz=(s.cls=='tz').mean(); p_copy=(s.cls=='copy').mean(); p_real=1-p_tz-p_copy
        idx=(win.mon==mon)&win.jn&(win.ms<1800)&(win.grp==g)
        taxi=900.0
        pred[idx]=p_real*taxi+p_tz*(taxi+win.off[idx])+p_copy*win.ms[idx].clip(lower=60)
        print(mon,g,'n_train',len(s),'p_tz',round(p_tz,3),'p_copy',round(p_copy,3),'rows',idx.sum())
report('TZ cell A mixture', win, pred)
# cell B: join-null & date-crossing & ms>=30000 : copy vs roll
cellB=t[t.jn&(t.cross>0)&(t.ms>=30000)]
print('cell B rows', len(cellB)); print(pd.crosstab(cellB.pat,cellB.cls)); print(pd.crosstab(cellB.car,cellB.cls).to_string())
wb=win[win.jn&(win.cross>0)&(win.ms>=30000)]
print('window cell B', len(wb)); print(wb[['src','FLIGHT_mvt','pat','y','newW','ms','cls']].to_string())
