import pandas as pd, numpy as np
from base import *
pd.set_option('display.width',250); pd.set_option('display.max_rows',300)
d=load_lirf_dep(); t=d[d.set=='train'].copy()
win=load_window(d)
t['del']=t.ms>=3600
# taxi_hat: median y of real rows by hour-bucket, computed per held-out month from other months
def shrink(k,n,prior,m): return (k+prior*m)/(n+m)
rows=[]
for mon in range(1,13):
    tr=t[t.mon!=mon]; te=t[(t.mon==mon)&t.jn&t['del']].copy()
    g_cell=tr[tr.jn&tr['del']]['copy'].mean()
    cell=tr[tr.jn&tr['del']].groupby('car')['copy'].agg(['sum','size'])
    dl=tr[tr['del']]
    pc=dl.groupby('car')['copy'].agg(['sum','size'])          # P(copy|delayed)
    q=dl[dl['copy']].groupby('car')['jn'].agg(['sum','size'])  # P(jn|copy,delayed)
    r=dl[~dl['copy']].groupby('car')['jn'].agg(['sum','size']) # P(jn|real,delayed)
    gp=dl['copy'].mean(); gq=dl[dl['copy']].jn.mean(); gr=dl[~dl['copy']].jn.mean()
    for M in [2,5,10]:
        te['pcell_%d'%M]=[shrink(*(cell.loc[c] if c in cell.index else (0,0)),g_cell,M) for c in te.car]
        P=np.array([shrink(*(pc.loc[c] if c in pc.index else (0,0)),gp,M) for c in te.car])
        Q=np.array([shrink(*(q.loc[c] if c in q.index else (0,0)),gq,M) for c in te.car])
        R=np.array([shrink(*(r.loc[c] if c in r.index else (0,0)),gr,M) for c in te.car])
        te['pbay_%d'%M]=P*Q/(P*Q+(1-P)*R)
    te['pglob']=g_cell
    rows.append(te)
cf=pd.concat(rows)
print('coin-flip rows', len(cf), 'copy rate', cf['copy'].mean())
for c in ['pglob']+[f'pcell_{M}' for M in [2,5,10]]+[f'pbay_{M}' for M in [2,5,10]]:
    p=cf[c].clip(1e-3,1-1e-3); yb=cf['copy'].astype(float)
    print(f'{c}: Brier {((p-yb)**2).mean():.4f}  logloss {-(yb*np.log(p)+(1-yb)*np.log(1-p)).mean():.4f}')
# also split by pattern
print(cf.groupby('pat').agg(n=('copy','size'),copy=('copy','mean'),pbay=('pbay_5','mean'),pcell=('pcell_5','mean')).to_string())
# SSE on window: pred = taxi_hat + P*(ms - taxi_hat)
w=win[win.jn&(win.ms>=3600)].merge(cf[['MVT_ID_mvt','pglob','pcell_5','pbay_5','pbay_2']],on='MVT_ID_mvt')
print('window coin-flip rows', len(w), 'SSE under newW', w.se.sum())
w['wimp']=((w.newW-900)/(w.ms-900)).clip(0,1)
yb=w['copy'].astype(float)
for c in ['wimp','pglob','pcell_5','pbay_5','pbay_2']:
    p=w[c].clip(1e-3,1-1e-3); print(f'window {c}: Brier {((p-yb)**2).mean():.4f}')
taxi=900.0
for c in ['pglob','pcell_5','pbay_5','pbay_2']:
    pred=taxi+w[c]*(w.ms-taxi); se2=(w.y-pred)**2
    print(f'{c}: SSE {se2.sum():.4e}  removed vs newW {(w.se-se2).sum():.4e}  Jan {(w.se-se2)[w.mon==1].sum():.4e} Jul {(w.se-se2)[w.mon==7].sum():.4e}')
pred=taxi+w['wimp']*(w.ms-taxi); print('wimp reconstruction SSE', ((w.y-pred)**2).sum())
# exclude rolls (y>=80000) to see ordinary-only
m=w.y<80000
for c in ['pcell_5','pbay_5']:
    pred=taxi+w[c]*(w.ms-taxi); se2=(w.y-pred)**2
    print(f'{c} non-roll rows: removed {(w.se-se2)[m].sum():.4e}')
cf.to_parquet('coinflip_cf.parquet',index=False)
