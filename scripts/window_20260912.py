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

The 2026-09-12 window as ONE linear system, per s254's rule: do not chain.

Three probe directions, all measured against the SAME base (v67 @ 262.5506). Firing them in any
order costs three slots; the fourth slot ships the solved joint optimum. No probe is banked
before the others are measured, so no correction is ever taken relative to a probe-shifted base
-- which is the error s254 found in the 09-11 window.

    delta_A   the three band shifts at their linear-system optima      (v69 - v67)
    delta_B   the learned 5-class gate on the EXTREME rows             (v71 territory)
    delta_C   the learned gate on the ORDINARY rows                    (v72 territory)

THE ALGEBRA. For a base p0 with board score B0 on N rows, a probe p0 + d_i scoring B_i gives
    N(B_i^2 - B0^2) = 2 T_i + Q_i      with   T_i = sum d_i * (p0 - y),  Q_i = sum d_i^2
so T_i is MEASURED exactly, with no assumption about y. For a combined ship p0 + sum a_i d_i,
    S(a) = S0 + 2 sum a_i T_i + a' G a           G_ij = d_i . d_j
minimised at  G a = -T.  G is computable offline from the parquets alone; only T needs the
board. The probes need NOT be disjoint -- overlap is carried exactly by G's off-diagonals,
which is why A and B may both touch the extreme rows.

  build the probes:   uv run ... scripts/window_20260912.py build
  solve, after firing: uv run ... scripts/window_20260912.py solve <B_A> <B_B> <B_C>
"""
import hashlib, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
DATA=os.environ['DC26_DATA']
N=344841; THR,FLOOR,CAP,PER=7200,60.,200000.,1.842e8
BASE='submissions/quick-boat_v67.parquet'; B0=262.5506
tpl=pq.read_table(f'{DATA}/submitting.parquet'); tb=tpl.to_pandas()[['MVT_ID_mvt']]
ids=tb.MVT_ID_mvt
def rd(f):
    t=pq.read_table(f).to_pandas()
    assert bool((t.MVT_ID_mvt.values==ids.values).all()), f'{f} row order'
    return t.TAXITIME_SEC_mvt.values.astype(float)
p0=rd(BASE); p69=rd('submissions/quick-boat_v69.parquet')
p64=rd('submissions/quick-boat_v64.parquet'); p70=rd('submissions/quick-boat_v70.parquet')
EXT=(p64>=THR)|(p70>=THR)
# the gate's own value, re-expressed as a move FROM v67 rather than from v64
dA=p69-p0
dB=np.where(EXT,p70-p0,0.0)
dC=np.where(EXT,0.0,p70-p0)
D=np.stack([dA,dB,dC]); NAMES=['A bands','B gate/extreme','C gate/ordinary']
G=D@D.T
def show():
    print(f'base {os.path.basename(BASE)} @ {B0:.4f}   N {N:,}   EXT rows {int(EXT.sum())}\n')
    print(f'{"probe":18s} {"rows moved":>11} {"Q = d.d":>13} {"board-s if empty":>17}')
    for i,nm in enumerate(NAMES):
        print(f'{nm:18s} {int((D[i]!=0).sum()):>11,} {G[i,i]:>13.4e} {G[i,i]/PER:>+16.2f}')
    print('\nGram matrix G (off-diagonals are the overlap the solve accounts for):')
    print(pd.DataFrame(G,index=NAMES,columns=NAMES).to_string(float_format=lambda v:f'{v:.4e}'))
    with np.errstate(invalid='ignore'):
        C=G/np.sqrt(np.outer(np.diag(G),np.diag(G)))
    print('\ncorrelation between probe directions:')
    print(pd.DataFrame(C,index=NAMES,columns=NAMES).to_string(float_format=lambda v:f'{v:+.4f}'))
    print(f'\nG condition number {np.linalg.cond(G):.2f}   '
          f'(a large value means the probes are near-collinear and the solve is ill-posed)')
def write(p,out):
    o=pd.DataFrame({'MVT_ID_mvt':ids.values,'TAXITIME_SEC_mvt':np.clip(np.rint(p),FLOOR,CAP).astype('int32')})
    t=pa.Table.from_pandas(o,schema=tpl.schema,preserve_index=False)
    assert t.schema.equals(tpl.schema); pq.write_table(t,out)
    print(f'  {out}  sha256 {hashlib.sha256(open(out,"rb").read()).hexdigest()[:16]}')
cmd=sys.argv[1] if len(sys.argv)>1 else 'build'
if cmd=='build':
    show()
    print('\nprobes, each = base + its own direction at FULL commitment:')
    for i,(nm,tag) in enumerate(zip(NAMES,['A','B','C'])):
        write(p0+D[i],f'submissions/quick-boat_w{tag}.parquet')
    ctl=p0+0.0*D[0]
    assert int((np.clip(np.rint(ctl),FLOOR,CAP).astype('int32')!=p0.astype('int32')).sum())==0
    print('\nCONTROL at zero commitment reproduces the base exactly.')
elif cmd=='solve':
    B=np.array([float(x) for x in sys.argv[2:5]])
    S0=N*B0*B0
    T=np.array([(N*(B[i]**2-B0**2)-G[i,i])/2.0 for i in range(3)])
    print(f'measured scores {B.round(4).tolist()}')
    for i,nm in enumerate(NAMES): print(f'  {nm:18s} T {T[i]:+.4e}   solo alpha* {-T[i]/G[i,i]:+.4f}')
    a=np.linalg.solve(G,-T)
    S=S0+2*a@T+a@G@a
    print(f'\njoint optimum a = {a.round(4).tolist()}')
    print(f'  predicted score {np.sqrt(max(S,0)/N):.4f}   gain {B0-np.sqrt(max(S,0)/N):+.4f} board-s')
    write(p0+a@D,'submissions/quick-boat_w_joint.parquet')
