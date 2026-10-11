"""v99 = v96 with the bulk's colsample_bytree changed from 1.0 to 0.4.

THE CHANGE. The shipped bulk considers ALL 65 features at every split. s481 measured
`colsample_bytree=0.4` against the correctly-identified shipped configuration (pooled
2000/255/20 via v23's DC26_POOLED_CAP, per-airport 600/127/100 via L()'s defaults):

    pooled bulk        Jan +1.7676   Jul +1.5723
    per-airport bulk   Jan +3.3268   Jul +2.6530

Positive on BOTH scored months, on a plateau (every value 0.8 down to 0.3 beats 1.0), and the
honest split selects it from either month independently. s482 swept every remaining non-capacity
lever on top of it -- row subsampling, per-node sampling, extra-trees, min_split_gain, max_bin,
cat_smooth -- and found nothing worth adding, so this ships alone: ONE hypothesis.

WHY A DELTA AND NOT A REBUILD. v96's lineage is a ten-node DAG (v96<-v95<-v91<-v90<-v84<-v79<-v74
<-{v64,v67,v73}), each node asserting its inputs' sha256. The bulk enters the final prediction only
through mixture components 0, 3 and 4, all bulk-derived, so the effect is additive:

    delta = 0.2016*(d_lgbm + d_perapt)        the two components' blend weights in v64

applied where the mixture is bulk-dominated. Copy-fired rows are excluded because there the
prediction is `ms`, not the bulk, and v96's own output identifies them: it equals ms to the second.

PIPELINE VERIFIED FIRST (s483). Rebuilding both models with the shipped settings reproduces
v15_trees_jul19.parquet EXACTLY -- lgbm and perapt both max|diff| 0.000, corr 1.000000. Three
baseline errors this session were caught by that control before anything shipped.

EVIDENCE: validation, bulk-side. CLAUDE.md measures bulk validation as trustworthy for DIRECTION
and poor for MAGNITUDE, so the direction is the claim and the board sets the size.
PREDICTION, stated before the measurement: +0.4 to +0.8 board-s. Ninth costs +1.64.
CONTROL: FIRE=0 reproduces v96 byte-identically.
"""
import hashlib, os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
D=os.environ['DC26_DATA']; W=os.environ['DC26_WORK']
BASE='submissions/quick-boat_v96.parquet'; OUT=os.environ.get('V99_OUT','submissions/quick-boat_v99.parquet')
FIRE=os.environ.get('FIRE','1')=='1'
FLOOR,CAP,PER=60.,200000.,1.842e8; WLG=0.2016; WPA=0.2016
EPOCH=pd.Timestamp('1970-01-01',tz='UTC')
def secs(s): return (pd.to_datetime(s,errors='coerce',utc=True)-EPOCH).dt.total_seconds()
b=pq.read_table(BASE).to_pandas()
tpl=pq.read_table(f'{D}/submitting.parquet').to_pandas()[['MVT_ID_mvt']]
assert bool((b.MVT_ID_mvt.values==tpl.MVT_ID_mvt.values).all()), 'base row order'
comp=pd.read_parquet(f'{W}/v99_bulk_components.parquet').set_index('MVT_ID_mvt').reindex(b.MVT_ID_mvt.values)
assert comp.lgbm_new.notna().all(), 'component join incomplete'
dl=comp.lgbm_new.values-comp.lgbm_old.values
dp=comp.perapt_new.values-comp.perapt_old.values
delta=WLG*dl+WPA*dp
R=pq.read_table(f'{D}/ranking.parquet',columns=['MVT_ID_mvt','PHASE_mvt','MVT_TIME_UTC_mvt',
                                               'SCHED_TIME_UTC_mvt']).to_pandas()
R=R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').reindex(b.MVT_ID_mvt.values)
ms=np.clip(secs(R.MVT_TIME_UTC_mvt)-secs(R.SCHED_TIME_UTC_mvt),FLOOR,CAP)
inc=b.TAXITIME_SEC_mvt.values.astype(float)
copy_fired=np.abs(inc-ms)<=60                     # v96 predicts ms there: not a bulk row
apply=~copy_fired
print(f'  {len(b):,} scored rows   copy-fired {int(copy_fired.sum()):,} '
      f'({float(copy_fired.mean()):.2%})   delta applied to {int(apply.sum()):,}')
print(f'  delta on applied rows: median {float(np.median(delta[apply])):+,.2f}s   '
      f'mean |d| {float(np.mean(np.abs(delta[apply]))):,.2f}s   '
      f'q01 {float(np.quantile(delta[apply],.01)):+,.0f}  q99 {float(np.quantile(delta[apply],.99)):+,.0f}')
p=inc.copy()
if FIRE: p[apply]=np.clip(np.rint(inc[apply]+delta[apply]),FLOOR,CAP)
nd=int((p!=inc).sum())
print(f'  rows changed {nd:,}   |change| median {float(np.median(np.abs(p-inc)[p!=inc])) if nd else 0:,.1f}s')
assert (nd>0)==FIRE
out=pd.DataFrame({'MVT_ID_mvt':b.MVT_ID_mvt.values,'TAXITIME_SEC_mvt':p.astype('int32')})
pq.write_table(pa.Table.from_pandas(out,preserve_index=False),OUT)
sha=hashlib.sha256(open(OUT,'rb').read()).hexdigest()
print(f'  wrote {OUT}  sha256 {sha[:16]}')
if not FIRE:
    assert sha==hashlib.sha256(open(BASE,'rb').read()).hexdigest(), 'FIRE=0 must reproduce v96'
    print('  CONTROL: FIRE=0 reproduces v96 byte-identically')
