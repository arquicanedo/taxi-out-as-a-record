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

s219 control: run v54's inversion estimator on VALIDATION, where y is known.

CLAUDE.md: any quantity inferred about the scored set from an observable must be computed by
the same formula on the window where the unobservable is KNOWN. The v54 projection (+2.28
board-s at alpha* 0.590) used p^2 * sum(D^2), which assumes y equals the copy on a true row
and the CURRENT PREDICTION on a false one. The second half is false here: the shipped stack
already commits at mean P_used 0.715 on this cell, so on a false fire y is near the bulk and
the current prediction is far from it. Writing D = copy - p = (1-P)*Dfull:

    true row  gains (1-P)^2 Dfull^2      false row costs P(1-P) Dfull^2

At P = 0.715 the penalty per false row is 2.5x the gain per true row, so at 59% precision the
optimum can be NEGATIVE -- de-commit, not commit. This script settles it by measurement.
"""
import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.environ['DC26_WORK']; FLOOR, CAP, THR, BS = 60, 200000, 7200, 1.842e8
# Paths are env defaults so this frozen script names no directory only one machine has.
# DC26_TRAIN_LEGACY defaults to the PRE-REISSUE extract these scripts were written
# against; its twelve training_*.parquet are identical to the current extract's, which
# is why nothing here ever failed loudly. Behaviour is unchanged with the vars unset.
TRAIN = os.environ.get('DC26_TRAIN_LEGACY', os.environ['DC26_DATA'])
z = np.load(f'{W}/gate_stage_a2.npz'); zm = np.load(f'{W}/gate_mlp.npz')
va = pd.read_parquet(f'{W}/night_va_all.parquet')
# s222 (code audit): these arrays are combined POSITIONALLY with `va`, and `mon` is used as
# "1 else July". Both held when written; neither was asserted. Guard both.
assert np.allclose(va.TAXITIME_SEC_mvt.values.astype(float), z['yv']), 'npz/parquet row misalignment'
assert set(va.month.unique()) <= {1, 7}, f'month not in {{1,7}}: the v45 shift would be wrong'
y, Pc, sd, lb = z['yv'], z['Pc'], z['sd_v'], z['lb_v']
bulk = 0.2*z['lgb_va'] + 0.4*zm['mlp_va'] + 0.4*z['pa_va']
sdc = np.clip(sd, FLOOR, CAP); lbc = np.clip(np.where(np.isnan(lb), sd, lb), FLOOR, CAP)
hit = (lb >= THR) & ~np.isnan(lb); copy = np.where(hit, lbc, sdc); P = np.maximum(Pc, hit*0.9)
apt = va.apt.astype(str).values; mon = va.month.values
base = np.clip(P*copy + (1-P)*bulk, FLOOR, CAP)

ROLL = np.clip(bulk + 86400., FLOOR, CAP)
V34 = (apt=='LIRF') & (~np.isfinite(lb)) & (sd >= 50000.)
p = np.where(V34, np.clip(np.rint(0.40*ROLL + 0.60*base), FLOOR, CAP), base)            # v34
p = np.clip(np.rint(p + np.where(mon==1, -8.778, -3.861)), FLOOR, CAP)                   # v45
C49 = (Pc >= 0.10) & (Pc < 0.95) & ~V34
p = np.where(C49, np.clip(np.rint((1-0.3376)*p + 0.3376*copy), FLOOR, CAP), p)           # v49
C51 = (~np.isfinite(lb)) & (sd >= 7200) & (sd < 14400) & (Pc >= 0.01) & ~C49 & ~V34
p = np.where(C51, np.clip(np.rint((1-0.2922)*p + 0.2922*bulk), FLOOR, CAP), p)           # v51
r = (p - y)**2
print(f'shipped-stack reconstruction on validation: RMSE {np.sqrt(r.mean()):.3f}\n')

raw = pd.concat([pq.read_table(f, columns=['MVT_ID_mvt','FLIGHT_mvt','CALLSIGN_flt']).to_pandas()
                 for f in sorted(glob.glob(f'{TRAIN}/training_*.parquet'))],
                ignore_index=True).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(va.MVT_ID_mvt)
allf = pd.concat([pq.read_table(f, columns=['PHASE_mvt','FLIGHT_mvt']).to_pandas()
                  for f in sorted(glob.glob(f'{TRAIN}/training_*.parquet'))], ignore_index=True)
allf = allf[allf.PHASE_mvt=='DEP'].FLIGHT_mvt.astype(str)
KNOWN = frozenset(allf[allf.str.match(r'^[A-Z]{3}\d')].str[:3].value_counts()[lambda s: s>=50].index)
def dup_gen(x):
    x = str(x)
    if len(x) < 4 or not x[:3].isalpha() or x[:3] not in KNOWN: return False
    return any(len(x) >= 3+k and x[3:3+k] == x[:3][-k:] for k in (1,2,3))
def dup_mine(x):
    x = str(x); return len(x) >= 5 and x[:3].isalpha() and (x[1:3] == x[3:5] or x[2] == x[3])
fs = raw.FLIGHT_mvt.astype(str); CN = raw.CALLSIGN_flt.isna().values
G = np.array([dup_gen(x) for x in fs]) & CN
M = np.array([dup_mine(x) for x in fs]) & CN
NL = raw.FLIGHT_mvt.isna().values & CN

def show(mask, name):
    EX = mask & ~V34
    if EX.sum() == 0: print(f'{name:<32} empty after the v34 carve-out'); return
    D = np.clip(copy, FLOOR, CAP) - p                       # v54's exact move direction
    ok = EX & (D != 0)
    Dn, yn, pn = D[ok], y[ok], p[ok]
    Q = float((Dn**2).sum()); T = float((Dn*(pn - yn)).sum()); a = -T/Q
    gain = T*T/Q/BS
    q = p.copy(); q[ok] = np.clip(np.rint(pn + a*Dn), FLOOR, CAP)
    r2 = (q - y)**2; o = np.argsort(-np.maximum(r, r2))
    ks = ' '.join(f'k={k}:{(r[o[k:]].sum()-r2[o[k:]].sum())/BS:+.3f}' for k in (1,5,20))
    print(f'{name:<32} n={int(ok.sum()):>3}  ex={int((yn>=THR).sum()):>3} ({(yn>=THR).mean():>5.1%})  '
          f'meanP={P[ok].mean():.3f}  alpha*={a:+.4f}  gain={gain:+.4f}  [{ks}]')

print(f'{"cell":<32} {"":>28} {"":>14} TRUE least-squares optimum, real y')
show(G,  'A. generalised x cs-null')
show(M & ~np.array([dup_gen(x) for x in fs]), 'B. s218 extras only')
show(M,  'C. union = s218 as built')
show(NL, 'D. FLIGHT_mvt null x cs-null')
print('\nprojection that the p^2*sumD^2 form gave for the scored set, for comparison:')
print('  A +2.2777   B +0.0216   C +1.9580   D +0.0279  (all POSITIVE by construction)')
