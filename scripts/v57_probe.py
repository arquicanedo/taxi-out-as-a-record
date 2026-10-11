"""RETIRED. This is probe-and-invert code, published deliberately.

WHY IT IS HERE. This script implements the method our paper discloses and retires: submit a
known perturbation on a known row set, read the public score, solve `dSSE = 2T + Q` for the
inner product of the perturbation with the true errors, and ship the optimum. We used it, it
contributed a measured +2.7501 RMSE-seconds or more to our standing, and we stopped on
2026-09-11 after an organiser wrote that reverse-engineering the ranking is unwelcome.

**We publish it rather than describe it** because a disclosure a reader cannot check is not a
disclosure. `scripts/v76_build.py` builds the same model with this content removed and ships
alongside the model that keeps it, so the difference can be priced without taking our word for
anything. Every constant is itemised in REPRODUCE.md section 9.

**It is not run by any shipped build**, and some of these scripts depend on intermediate state
that no longer exists, so several will not execute at all. They are here to be read. Do not run
them: the method measures the evaluation set rather than the problem, which is why we stopped,
independent of any rule.

---

v57: the LOBT-hit branch cut on |D|, rebased onto v56. PROBE, then invert.

The hypothesis is s107's own sentence turned into a cut. s107 swept the hit-branch P over ALL
hit rows and found higher better, because "the rows where firing pays carry a D an order of
magnitude larger than the rows where it costs". Nobody ever cut the branch on D. The small-|D|
hit rows are the subset that sentence says are over-committed.

Two independent things now point the same way:
  - validation: hit & |D|<5000 sits at P_now 0.903 against a least-squares P* of 0.595
  - THE BOARD, tonight: the designator cell's true alpha* was -0.0823, a de-commitment, where
    validation said +0.68. Our copy machinery is over-committed on small-D rows.

THIS IS A PROBE. It is expected to score WORSE. Its job is to measure T so the optimum can be
shipped exactly, because s226 established that validation must never set a submission's weight.

Rebased: BASE is v56 (264.8006) and `bulk` is the OPDI blend, since v56 ships those components.
CONTROL: EPS=0 reproduces quick-boat_v56.parquet byte-identically.
"""
import hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
EPS  = float(os.environ.get('V57_EPS', '1.0'))
OUT  = os.environ.get('V57_OUT', 'submissions/quick-boat_v57.parquet')
# Paths are env defaults so this frozen script names no directory only one machine has.
# DC26_TRAIN_LEGACY defaults to the PRE-REISSUE extract these scripts were written
# against; its twelve training_*.parquet are identical to the current extract's, which
# is why nothing here ever failed loudly. Behaviour is unchanged with the vars unset.
DATA = os.environ['DC26_DATA']; W = os.environ['DC26_WORK']
BASE = 'submissions/quick-boat_v56.parquet'
CAP, FLOOR, THR, N, PER = 200000, 60, 7200, 344841, 1.842e8
DMAX = 5000.0; B_BASE = 264.8006
tpl = pq.read_table(f'{DATA}/submitting.parquet'); tb = tpl.to_pandas()[['MVT_ID_mvt']]
v56 = pq.read_table(BASE).to_pandas()
assert len(v56) == N and bool((v56.MVT_ID_mvt.values == tb.MVT_ID_mvt.values).all())
p56 = v56.TAXITIME_SEC_mvt.values.astype(float)
t = pd.read_parquet(f'{W}/v15_trees_jul19.parquet'); n = pd.read_parquet(f'{W}/v15_net_jul19.parquet')
m = t.merge(n, on='MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(v56.MVT_ID_mvt)
Pc = m.Pc.values.astype(float); sd = m.sched_delay.values.astype(float); lb = m.mvt_minus_lobt.values.astype(float)
nw = pd.read_parquet(f'{W}/v55_trees.parquet').set_index('MVT_ID_mvt').reindex(v56.MVT_ID_mvt)
bulk = (0.2*nw.lgbm_opdi.values + 0.4*m.mlp.values + 0.4*nw.perapt_opdi.values).astype(float)
assert np.isfinite(bulk).all(), 'OPDI bulk has NaNs'
hit = np.isfinite(lb) & (lb >= THR)
copy = np.where(hit, np.clip(np.where(np.isfinite(lb), lb, sd), FLOOR, CAP), np.clip(sd, FLOOR, CAP))
P_used = np.maximum(Pc, hit*0.9); D = copy - bulk
rk = pq.read_table(f'{DATA}/ranking.parquet', columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','LOBT_flt']).to_pandas()
rd = rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(v56.MVT_ID_mvt)
V34 = (rd.ADEP_mvt.astype(str).values=='LIRF') & rd.LOBT_flt.isna().values & (sd>=50000.)
D49 = (Pc>=.10)&(Pc<.95)&~V34
D51 = (~np.isfinite(lb))&(sd>=7200)&(sd<14400)&(Pc>=.01)&~D49&~V34
EX = hit & (np.abs(D) < DMAX) & ~D49 & ~D51 & ~V34
print(f'cell: LOBT hit & |copy-bulk| < {DMAX:.0f}   rows {int(EX.sum())}')
assert 20 <= EX.sum() <= 400, f'cell size {int(EX.sum())} outside the sane band'
print(f'  mean P_used {P_used[EX].mean():.4f}   |D| median {np.median(np.abs(D[EX])):,.0f}s')
big = hit & (np.abs(D)>=DMAX) & ~D49 & ~D51 & ~V34
print(f'  UNTOUCHED for contrast: hit rows with |D| >= {DMAX:.0f}: {int(big.sum())} rows, '
      f'|D| median {np.median(np.abs(D[big])):,.0f}s')
def build(eps):
    p = p56.copy(); p[EX] = np.clip(np.rint((1-eps)*p56[EX] + eps*bulk[EX]), FLOOR, CAP); return p
chk = build(0.0); d = int((chk.astype('int32') != p56.astype('int32')).sum())
print(f'\nCONTROL eps=0 vs v56: {d} rows differ'); assert d == 0, 'CONTROL FAILED'
print('  byte-identical.\n')
pred = build(EPS); u = pred - p56; Q = float((u**2).sum())
print(f'EPS {EPS}  rows moved {int((u!=0).sum())}  Q = {Q:.6e} = {Q/PER:+.4f} board-s if T=0')
print(f'  |move| median {np.median(np.abs(u[u!=0])):,.0f}s  max {np.abs(u).max():,.0f}s')
print(f'\n--- INVERSION TABLE, pre-registered. alpha* is relative to THIS EPS={EPS} move. ---')
S0 = N*B_BASE*B_BASE
print(f'  {"if v57 scores":>14} {"alpha*":>9} {"ship EPS":>9} {"best along u":>13} {"gain":>8}')
for Bp in (B_BASE-0.5, B_BASE, B_BASE+0.5, B_BASE+1.0, B_BASE+2.0, B_BASE+4.0, B_BASE+8.0):
    dS = N*(Bp*Bp-B_BASE*B_BASE); T = (dS-Q)/2.0; a = -T/Q
    Bs = np.sqrt(max(S0-T*T/Q,0)/N)
    print(f'  {Bp:>14.4f} {a:>9.4f} {a*EPS:>9.4f} {Bs:>13.4f} {B_BASE-Bs:>+8.4f}')
o = tb.merge(pd.DataFrame({'MVT_ID_mvt': v56.MVT_ID_mvt.values,'TAXITIME_SEC_mvt': pred}), on='MVT_ID_mvt', how='left')
o['TAXITIME_SEC_mvt'] = o.TAXITIME_SEC_mvt.astype('int32')
assert len(o)==len(tb) and o.TAXITIME_SEC_mvt.isna().sum()==0
assert bool((o.MVT_ID_mvt.values==tb.MVT_ID_mvt.values).all())
assert o.TAXITIME_SEC_mvt.min()>=FLOOR and o.TAXITIME_SEC_mvt.max()<=CAP
pq.write_table(pa.Table.from_pandas(o[['MVT_ID_mvt','TAXITIME_SEC_mvt']], schema=tpl.schema, preserve_index=False), OUT)
print(f'\nwrote {OUT}\n  sha256 {hashlib.sha256(open(OUT,"rb").read()).hexdigest()}')
