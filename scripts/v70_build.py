"""v70: v62's LEARNED PROVENANCE GATE on v64's bulk. The honest one-hypothesis test.

v62 (sha 7be204fb51f125ae) replaced the hand-built extreme chain with a learned 5-class gate,
but it was built on the v55-era bulk `0.2 LGB / 0.4 MLP / 0.4 perApt`. v64 then refitted the
bulk to `0.2473 / 0.0000 / 0.2217 / 0.0849 D / 0.4462 D2` and paid +1.28 board-s. Firing v62 now
would bundle the gate change with a bulk REGRESSION -- exactly the v55 mistake v62's own
docstring was written against.

v70 holds the bulk at v64's and changes ONLY the gate:

    v64   bulk_new -> v29 -> v34 -> v45 -> v49 -> v51 -> v54(A54) -> v57(A57)   seven board cells
    v70   bulk_new -> one 5-class gate, every constant from labelled 2025 data

CONTROL. Run with the v62-era bulk it must reproduce quick-boat_v62.parquet byte-identically.
That is asserted below and the script refuses to write if it fails.

The gate is a MODEL SPECIFICATION fitted out of fold on labelled data (s234), not a board
measurement, so `scripts/presubmit_guard.py` will flag it as extreme-band-with-validation-
evidence. That is the v54 rule working as intended: this needs an explicit decision.
"""
import glob, hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']; W=os.environ['DC26_WORK']
OUT=os.environ.get('V70_OUT','submissions/quick-boat_v70.parquet')
TEMP=float(os.environ.get('V70_TEMP','0.90'))      # s234, jointly OOF-fitted with the floor
GFLOOR=float(os.environ.get('V70_FLOOR','0.10'))
FLOOR,CAP,DAY,THR=60.,200000.,86400.,7200
W_LGB,W_MLP,W_PA,W_D,W_D2=0.2473,0.0000,0.2217,0.0849,0.4462   # v64, as shipped
SHA62='7be204fb51f125ae'
BASE=os.environ.get('V70_BASE','submissions/quick-boat_v64.parquet')
BSCORE=float(os.environ.get('V70_BSCORE','263.0551'))          # v64's ACTUAL board score

tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
ids=tb.MVT_ID_mvt
t=pd.read_parquet(f'{W}/v15_trees_jul19.parquet'); n=pd.read_parquet(f'{W}/v15_net_jul19.parquet')
m=t.merge(n,on='MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(ids)
mlp_=m.mlp.values.astype(float)
sd=m.sched_delay.values.astype(float); lb=m.mvt_minus_lobt.values.astype(float)
nw=pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(ids)
lgb_new,pa_new=nw.lgbm_opdi.values.astype(float),nw.perapt_opdi.values.astype(float)
dw=pd.read_parquet(f'{W}/d_window_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
d2=pd.read_parquet(f'{W}/d2_window_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
d_win,d2_win=dw.d_window.values.astype(float),d2.d2_window.values.astype(float)
assert np.isfinite(d_win).all() and np.isfinite(d2_win).all(), 'D/D2 NaN after reindex'

bulk_v62=np.clip(0.2*lgb_new+0.4*mlp_+0.4*pa_new,FLOOR,CAP)          # what v62 used
bulk_v64=W_LGB*lgb_new+W_MLP*mlp_+W_PA*pa_new+W_D*d_win+W_D2*d2_win  # what v64 ships
assert abs(W_LGB+W_MLP+W_PA+W_D+W_D2-1.0)<1e-3
print(f'bulk_v62 median {np.median(bulk_v62):,.0f}s   bulk_v64 median {np.median(bulk_v64):,.0f}s   '
      f'corr {np.corrcoef(bulk_v62,bulk_v64)[0,1]:.4f}')

g=pd.read_parquet(f'{W}/b4_gate_rk.parquet').set_index('MVT_ID_mvt').reindex(ids)
P=g[[f'P{i}' for i in range(5)]].to_numpy(float)
assert np.isfinite(P).all() and np.abs(P.sum(1)-1).max()<1e-6, 'gate rows do not sum to 1'
Q=P**TEMP; Q/=Q.sum(1,keepdims=True)
if GFLOOR>0:
    R=Q.copy(); R[:,1:][R[:,1:]<GFLOOR]=0.0
    R[:,0]=1.0-R[:,1:].sum(1); R=np.clip(R,1e-12,None); R/=R.sum(1,keepdims=True); Q=R
vsd=np.clip(sd,FLOOR,CAP); vlb=np.clip(np.where(np.isfinite(lb),lb,sd),FLOOR,CAP)

def gate_pred(bulk):
    V=np.stack([bulk,vsd,vlb,np.clip(DAY+bulk,FLOOR,CAP),bulk],1)
    return np.clip(np.rint((Q*V).sum(1)),FLOOR,CAP)

# --- CONTROL: the v62-era bulk through the identical gate must reproduce v62 exactly ---------
ctl=gate_pred(bulk_v62)
v62=pq.read_table('submissions/quick-boat_v62.parquet').to_pandas()
assert bool((v62.MVT_ID_mvt.values==ids.values).all())
nd=int((ctl.astype('int32')!=v62.TAXITIME_SEC_mvt.values.astype('int32')).sum())
print(f'CONTROL vs quick-boat_v62.parquet: {nd} rows differ')
assert nd==0, 'CONTROL FAILED -- gate path not faithful to v62, do not ship'
print('  byte-identical. Only the bulk changes below.\n')

pred=gate_pred(bulk_v64)
names=['ordinary','SCHED copy','LOBT stale','date error','other extreme']
for i,nm in enumerate(names):
    print(f'  {nm:<16} argmax {int((Q.argmax(1)==i).sum()):>7,}   mean P {Q[:,i].mean():.6f}')
b=pq.read_table(BASE).to_pandas(); p0=b.TAXITIME_SEC_mvt.values.astype(float)
assert bool((b.MVT_ID_mvt.values==ids.values).all())
d=pred-p0; ch=int((pred.astype('int32')!=p0.astype('int32')).sum()); Qd=float((d**2).sum())
ext0=p0>=THR
print(f'\nvs {os.path.basename(BASE)} @ {BSCORE:.4f}: {ch:,} rows changed ({ch/len(pred):.1%})  '
      f'mean {d.mean():+.2f}s  sd {d.std():,.1f}s  |max| {np.abs(d).max():,.0f}s')
print(f'  Q = {Qd:.6e} = {Qd/1.842e8:+.2f} board-s if the direction is empty')
print(f'  changed SSE mass in the extreme band: {float((d**2)[ext0].sum())/Qd*100:.1f}%')
print(f'  base calls extreme, v70 does not: {int((ext0&(pred<THR)).sum())}')
print(f'  v70 calls extreme, base does not: {int((~ext0&(pred>=THR)).sum())}')
print(f'  both extreme: {int((ext0&(pred>=THR)).sum())}')
N=344841; S0=N*BSCORE*BSCORE
print(f'\n--- INVERSION, pre-registered. dSSE = 2T + Q, alpha* = -T/Q ---')
print(f'  {"if v70 scores":>14} {"alpha*":>9} {"best along":>11} {"gain":>8}')
for Bp in (258.0,260.0,261.5,262.5,263.0551,264.5,267.0,275.0):
    dS=N*(Bp*Bp-BSCORE*BSCORE); T=(dS-Qd)/2.0; a=-T/Qd
    Bs=np.sqrt(max(S0-T*T/Qd,0)/N)
    print(f'  {Bp:>14.4f} {a:>9.4f} {Bs:>11.4f} {BSCORE-Bs:>+8.4f}')
o=tb.merge(pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':pred}),on='MVT_ID_mvt',how='left')
o['TAXITIME_SEC_mvt']=o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
tt=pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']],schema=tpl.schema,preserve_index=False)
assert tt.schema.equals(tpl.schema)
pq.write_table(tt,OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
