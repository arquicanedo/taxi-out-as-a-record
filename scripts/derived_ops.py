import os, pandas as pd, numpy as np
TMP=os.environ.get('DC26_TMP','/tmp')   # scripts/newcols.py writes the input here
d=pd.read_parquet(f'{TMP}/dc26_newcols.parquet')
d=d.sort_values(['apt','MVT_TIME_UTC_mvt']).reset_index(drop=True)
d['t_off']=d.MVT_TIME_UTC_mvt.values.astype('datetime64[s]').astype('int64').astype(float)
ob=d.AOBT_3_flt.values.astype('datetime64[s]').astype('int64').astype(float)
ob[pd.isna(d.AOBT_3_flt.values)]=np.nan; d['t_obt']=ob
out=[]
for a,g in d.groupby('apt',sort=False):
    g=g.copy(); n=len(g)
    # prevailing configuration = modal departure runway over the trailing 20 departures
    c=g.RUNWAY_mvt.astype('category'); codes=c.cat.codes.values; K=len(c.cat.categories)
    oh=np.zeros((n,K)); oh[np.arange(n),np.clip(codes,0,K-1)]=1.0; oh[codes<0]=0.0
    cs=np.vstack([np.zeros(K),np.cumsum(oh,axis=0)])
    W=20; lo=np.clip(np.arange(n)-W,0,None)
    win=cs[np.arange(n)]-cs[lo]                      # counts in trailing window, excluding self
    prev=np.where(win.sum(1)>0,win.argmax(1),-1)
    sw=np.zeros(n,bool); sw[1:]=(prev[1:]!=prev[:-1])&(prev[1:]>=0)&(prev[:-1]>=0)
    t=g.t_off.values
    sw_t=np.where(sw,t,np.nan)
    last_sw=pd.Series(sw_t).ffill().values                  # most recent switch time at/ before takeoff
    g['min_since_switch']=(t-last_sw)/60.0
    # THE EVENT: pushed back before the switch, took off after it
    g['caught_by_switch']=((g.t_obt.values<last_sw)&(last_sw<=t)).astype(float)
    g['push_to_switch_min']=np.where(g.caught_by_switch>0,(last_sw-g.t_obt.values)/60.0,np.nan)
    # the queue that forms: aircraft on the surface at the moment of the switch
    both=(~np.isnan(g.t_obt.values))&(~np.isnan(t))
    o=np.sort(g.t_obt.values[both]); k=np.sort(t[both])
    surf=lambda x: np.searchsorted(o,x,'right')-np.searchsorted(k,x,'right')
    g['backlog_at_switch']=np.where(np.isnan(last_sw),np.nan,surf(np.nan_to_num(last_sw)))
    out.append(g)
d=pd.concat(out).sort_index()
# storm / ground-delay proxy: realised departure rate vs this airport-hour-dow's norm.
# MVT_TIME is given for every row including ranking, so this leaks nothing.
d['b15']=d.MVT_TIME_UTC_mvt.dt.floor('15min')
cnt=d.groupby(['apt','b15'],observed=True).size().rename('n').reset_index()
cnt['hour']=cnt.b15.dt.hour; cnt['dow']=cnt.b15.dt.dayofweek
norm=cnt.groupby(['apt','hour','dow'],observed=True).n.median().rename('n_exp').reset_index()
cnt=cnt.merge(norm,on=['apt','hour','dow'],how='left')
cnt['rate_ratio']=cnt.n/cnt.n_exp.replace(0,np.nan)
cnt=cnt.sort_values(['apt','b15'])
cnt['deficit_2h']=cnt.groupby('apt',observed=True).apply(
    lambda g:(g.n_exp-g.n).rolling(8,min_periods=1).sum()).reset_index(level=0,drop=True)
d=d.merge(cnt[['apt','b15','rate_ratio','deficit_2h']],on=['apt','b15'],how='left')
y=d.TAXITIME_SEC_mvt.values.astype(float); ok=~np.isnan(y)&(y>0); ext=(y>=7200)&ok
print(f'{ok.sum():,} labelled departures   extremes {int(ext.sum())}')
for c in ['min_since_switch','caught_by_switch','push_to_switch_min','backlog_at_switch','rate_ratio','deficit_2h']:
    v=d[c].values
    print(f'  {c:22s} null {100*np.isnan(v).mean():5.1f}%  median {np.nanmedian(v):8.2f}  p95 {np.nanpercentile(v,95):9.2f}')
print('\n=== does being CAUGHT BY A SWITCH lengthen the taxi? ===')
cb=(d.caught_by_switch.values==1)&ok; nb=(d.caught_by_switch.values==0)&ok
print(f'  caught n={cb.sum():,}  median y {np.median(y[cb]):6.0f}s  mean {y[cb].mean():7.0f}s  ext rate {100*ext[cb].mean():.4f}%')
print(f'  not     n={nb.sum():,}  median y {np.median(y[nb]):6.0f}s  mean {y[nb].mean():7.0f}s  ext rate {100*ext[nb].mean():.4f}%')
print(f'  median difference: {np.median(y[cb])-np.median(y[nb]):+.0f}s')
print('\n=== by minutes since the switch (the ~30 min disruption window) ===')
for lo,hi in [(0,10),(10,20),(20,30),(30,60),(60,120),(120,1e9)]:
    m=ok&(d.min_since_switch.values>=lo)&(d.min_since_switch.values<hi)
    if m.sum()<200: continue
    print(f'  {lo:4.0f}-{hi if hi<1e9 else 999:4.0f} min  n={m.sum():7,d}  median y {np.median(y[m]):6.0f}s  ext {100*ext[m].mean():.4f}%')
print('\n=== storm proxy: suppressed departure rate then surge ===')
for lo,hi,lab in [(-1e9,-10,'surge (rate>norm)'),(-10,10,'normal'),(10,30,'deficit 10-30'),(30,1e9,'deficit >30 (GDP-like)')]:
    m=ok&(d.deficit_2h.values>=lo)&(d.deficit_2h.values<hi)
    if m.sum()<200: continue
    print(f'  {lab:24s} n={m.sum():8,d}  median y {np.median(y[m]):6.0f}s  mean {y[m].mean():7.0f}s  ext {100*ext[m].mean():.4f}%')
d.to_parquet(f'{TMP}/dc26_derived.parquet',index=False); print(f'\nsaved {TMP}/dc26_derived.parquet')
