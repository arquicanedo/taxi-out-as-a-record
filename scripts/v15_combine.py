"""v15 combine: v14's mixture with a three-model bulk.

Changes from v14 (scripts/v14_combine.py), both bulk-side only:
  1. block_dev in F_BULK (s77): -1.07 on the LightGBM bulk, -0.56 in the blend
     when only the trees carried it. Here the net carries it too.
  2. a 10-model per-airport bulk (s61) at weights .2/.4/.4, worth -1.83 on the
     validation bulk, cross-fitted and seed-stable.

The extreme side is UNCHANGED: same classifier features, same LOBT_THR=7200 and
LOBT_P=0.9 (s75 confirmed 7200 is the two-sided optimum), same CAP/FLOOR.

The EHAM edit's row set is RE-DERIVED, not assumed. Changing the bulk changes
`pred`, hence the `pred>6000` predicate, so the count may differ from v14's 46.

Usage: uv run scripts/v15_combine.py [trees] [net] [out]
"""
import pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq, sys, hashlib
import features
D=features.DATA
CAP=200000; LOBT_THR=7200; LOBT_P=0.9; FLOOR=60
W_LGB, W_MLP, W_PA = 0.2, 0.4, 0.4          # s61 min-variance was .15/.459/.392
EHAM_NAIVE_MIN=4800; EHAM_PRED_MIN=6000
trees=sys.argv[1] if len(sys.argv)>1 else features.wk('v15_trees')
net  =sys.argv[2] if len(sys.argv)>2 else features.wk('v15_net')
out  =sys.argv[3] if len(sys.argv)>3 else 'submissions/quick-boat_v18.parquet'
# The expected row count is whatever THIS extract's submitting.parquet asks for.
# It was hardcoded at 215,876; the 2026-09-04 July re-extract moved it to 344,841
# and a literal here would have shipped a short file or blocked a correct one.
tpl=pq.read_table(f'{D}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
NROWS=len(tb)
rk=pq.read_table(features.wk('v15_rk'),columns=['MVT_ID_mvt','apt','naive']).to_pandas()
t=pd.read_parquet(trees); n=pd.read_parquet(net)
m=t.merge(n,on='MVT_ID_mvt',how='inner').merge(rk,on='MVT_ID_mvt',how='left')
assert len(m)==NROWS, f'row count {len(m)} != submitting.parquet {NROWS}'
assert abs(W_LGB+W_MLP+W_PA-1)<1e-9

sd=m.sched_delay.values; lb=m.mvt_minus_lobt.values
sdv=np.clip(sd,FLOOR,CAP)
lbv=np.clip(np.where(np.isnan(lb),sd,lb),FLOOR,CAP)
hit=(lb>=LOBT_THR)&~np.isnan(lb)
bulk=W_LGB*m.lgbm.values + W_MLP*m.mlp.values + W_PA*m.perapt.values
P=np.maximum(m.Pc.values,hit*LOBT_P)
pred=np.clip(P*np.where(hit,lbv,sdv)+(1-P)*bulk,FLOOR,CAP)

nv=m.naive.values
edit=(m.apt.values=='EHAM')&(nv>=EHAM_NAIVE_MIN)&(pred>EHAM_PRED_MIN)&~np.isnan(nv)
k=int(edit.sum()); n_eham=int((m.apt.values=='EHAM').sum())
# The 30-70 band is a sanity guard around v14's 46 rows, sized for EHAM's scored row count. Unchanged by the July
# re-extract (38,182 rows either side: EHAM was already one of the three July
# airports), so the band still applies. Re-derive it if that count moves.
print(f'EHAM edit fires on {k} rows (v14: 46) over {n_eham:,} EHAM rows. Re-derived, not assumed.')
assert 30<=k<=70, f'edit count {k} outside the sane band -- inspect before shipping'
pred=np.where(edit,nv,pred)

o=tb.merge(pd.DataFrame({'MVT_ID_mvt':m.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':np.rint(pred)}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert bool((o.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all()), 'row order must match submitting.parquet'
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False),out)
v14=pd.read_parquet('submissions/quick-boat_v14.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
d=o.set_index('MVT_ID_mvt').TAXITIME_SEC_mvt-v14.reindex(o.MVT_ID_mvt) if len(v14)==len(o) else None
print(f'wrote {out}')
print(f'  rows {len(o):,}  median {o.TAXITIME_SEC_mvt.median():.0f}  max {o.TAXITIME_SEC_mvt.max():,}  >2h: {(o.TAXITIME_SEC_mvt>7200).sum()} (v14: 69)')
print(f'  vs v14: {int((d!=0).sum()):,} rows changed, mean {d.mean():+.1f}s, |max| {d.abs().max():,.0f}s'
      if d is not None else f'  vs v14: SKIPPED, v14 covers {len(v14):,} rows and this build has {len(o):,}')
print(f'  rows at CAP {int((o.TAXITIME_SEC_mvt>=CAP-1).sum())} (must be 0), at FLOOR {int((o.TAXITIME_SEC_mvt<=FLOOR).sum())}')
print(f'  sha256 {hashlib.sha256(open(out,"rb").read()).hexdigest()}')
