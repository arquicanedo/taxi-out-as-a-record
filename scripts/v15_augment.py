"""v17 stage 0: add block_dev (s77) and sched_vs_hr (s78) to the feature tables.

block_dev = flown_block - filed_block, the en-route deviation between NM's M1
(filed) and M3 (flown) trajectories. Worth -1.08 on the LightGBM bulk alone.

Added to F_BULK ONLY, not to F. The classifier drives Pc and the entire
extreme-side result; the measured gain is on the bulk. One change at a time.

Writes /tmp/v15_{tr,rk}.parquet and /tmp/v15_cols.json, leaving the v8s files
untouched so scripts/v14_combine.py keeps reproducing the shipped submission.
"""
import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
t0=time.time()
import features
tr=pd.read_parquet(features.wk('v8s_tr')); rk=pd.read_parquet(features.wk('v8s_rk'))
c=json.load(open(features.wk('v8s_cols','json')))
import v17_features
X=v17_features.build(); NEWF=v17_features.NEW
n0,r0=len(tr),len(rk)
tr=tr.merge(X,on='MVT_ID_mvt',how='left'); rk=rk.merge(X,on='MVT_ID_mvt',how='left')
assert len(tr)==n0 and len(rk)==r0, 'merge fanned out'
for k in NEWF: assert k not in c['F_BULK']
c['F_BULK']=list(c['F_BULK'])+NEWF
tr.to_parquet(features.wk('v15_tr'),index=False); rk.to_parquet(features.wk('v15_rk'),index=False)
json.dump(c,open(features.wk('v15_cols','json'),'w'))
for k in NEWF: print(f'  {k:14s} null: train {tr[k].isna().mean():6.2%}  rank {rk[k].isna().mean():6.2%}')
# The PRU cause features are LEGITIMATELY sparse: arrival delay by cause is
# recorded only on days that had delay, so ~50-60% null is expected and correct.
# Everything else must be dense; a high null rate there means the wrong source
# (dc26_derived holds only 2025 rows and once produced a 100%-null feature).
SPARSE_OK={'atfm_cause_S','atfm_cause_W','atfm_cause_G','atfm_dly_per_arr'}
for k in NEWF:
    lim=0.75 if k in SPARSE_OK else 0.5
    assert rk[k].isna().mean()<lim, f'{k} {rk[k].isna().mean():.1%} null on ranking -- wrong source'
print(f'F_BULK {len(c["F_BULK"])} features, F {len(c["F"])} (classifier unchanged)  [{time.time()-t0:.0f}s]')
