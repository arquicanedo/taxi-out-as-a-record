import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',40); pd.set_option('display.max_rows',300)
w=pd.read_parquet('win_lirf.parquet')
w['jn']=w.LOBT_flt.isna()
w['roll']=w.y>=80000
w['cls']=np.where(w['copy'],'copy',np.where(w.roll,'roll','real'))
# taxi estimate for real rows: median by (runway, hour) from other months? crude: use newW if newW<2500 else 1000
taxi=np.where(w.newW<2500,w.newW,1000.0)
pred=np.where(w.cls=='copy',w.ms,np.where(w.cls=='roll',86400+taxi,taxi))
w['se_or']=(w.y-pred)**2
tot=w.se.sum()
print('TOTAL', tot, ' oracle3 SSE', w.se_or.sum(), ' removed', tot-w.se_or.sum())
w['msb']=pd.cut(w.ms,[-1e9,1800,3600,7200,14400,1e9],labels=['<30m','30-60m','1-2h','2-4h','>4h'])
g=w.groupby(['jn','msb','cls'],observed=True).agg(n=('se','size'),SSE=('se','sum'),SSE_or=('se_or','sum'))
g['removed']=g.SSE-g.SSE_or
print(g.to_string())
# oracle where only the class is known for ms>=1800 rows, all other rows keep newW
m=w.ms>=1800
p2=np.where(m,pred,w.newW); print('class-oracle restricted to ms>=1800 rows: removed', tot-((w.y-p2)**2).sum(), 'n',m.sum())
m=w.ms>=3600
p2=np.where(m,pred,w.newW); print('class-oracle restricted to ms>=3600 rows: removed', tot-((w.y-p2)**2).sum(), 'n',m.sum())
m=w.jn
p2=np.where(m,pred,w.newW); print('class-oracle restricted to join-null rows: removed', tot-((w.y-p2)**2).sum(), 'n',m.sum())
m=(~w.jn)&(w.ms>=1800)
p2=np.where(m,pred,w.newW); print('class-oracle restricted to join-present ms>=1800: removed', tot-((w.y-p2)**2).sum(), 'n',m.sum())
# ordinary rows where the model is off by >1500 but class=real and ms<1800 : what are they
o=w[(w.cls=='real')&(w.ms<1800)&(w.res.abs()>1500)]
print('real, ms<1800, |res|>1500: n',len(o),'SSE',o.se.sum())
print(o.sort_values('se',ascending=False)[['src','FLIGHT_mvt','ADES_mvt','AIRCRAFT_TYPE_mvt','STAND_mvt','RUNWAY_mvt','MVT_TIME_UTC_mvt','y','newW','res','bs','ms','ml','ma']].head(25).to_string())
