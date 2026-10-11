"""The CLEAN base model: v100's architecture with nothing fitted to scores returned by the 2026 set.

The submitted v100 applies its mixture through a hand-built chain whose copy probabilities are reconstructed by
inverting an archived calibration base (v29, whose weight 0.35 was chosen on the leaderboard) and which carries
constants solved from leaderboard scores (REPRODUCE.md section 9). The clean base instead uses, on the scored rows,
exactly the construction the labelled validation window has always used:

  probabilities   the learned five-class gate (scripts/arch/b4_rank.py, trained on labelled 2025; b4_gate_rk.parquet)
  mixture         pi = P^0.90 renormalised, branches 1-4 below 0.10 zeroed (both swept on the window)
  branch values   [bulk, clip(MVT - SCHED), clip(MVT - LOBT) or the schedule copy where LOBT is null,
                   clip(86,400 + bulk), bulk]
  bulk            v100's members with their out-of-fold weights (full_package.npz), first mixed as the 4-member
                  blend and then moved to the 9-member blend on rows the mixture does not commit, as on the window
  month offsets   the mean residual of the 9-member bulk on ordinary window rows, by month: labelled 2025 data,
                  replacing the leaderboard-fitted SHIFT_1 = -8.778 and SHIFT_7 = -3.861
  bulk weights    WF=refit (default): the nine-member weights refitted by the same simplex on ordinary window rows
                  with a free intercept per month, so that no leaderboard offset enters their target; the intercepts
                  are the month offsets. WF=archived: full_package.npz's weights as they are, which were fitted
                  against y minus the leaderboard offsets (scripts/p2p_checks.py, s561); this reproduces the files
                  submitted as v115 and v116 byte-identically, including their offsets rounded to 3 decimals

Writes $CLEAN_DIR/submissions/quick-boat_v100.parquet (the clean chain names its base v100),
$CLEAN_DIR/offsets.json and $CLEAN_DIR/wF.npz (the weights the CLEAN-mode pricers read through DC26_WF), and prints the window RMSE of the clean base and its distance from the shipped v100.
  CLEAN_DIR=... uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python scripts/clean_base_build.py
"""
import itertools, json, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); __file__ = os.path.join(HERE, 'adsb_stage_exp.py')
os.environ.pop('DC26_SUBDIR', None); os.environ.pop('CLEAN', None)   # the window side reads nothing from a scored chain
WF = os.environ.get('WF', 'refit'); assert WF in ('refit', 'archived'), WF
exec(open(os.path.join(HERE, 'adsb_price.py')).read().split('# ---------------- 1. v100 on the window')[0])
CLEAN_DIR = os.environ.get('CLEAN_DIR', f"{os.environ['DC26_WORK']}/clean")
os.makedirs(f'{CLEAN_DIR}/submissions', exist_ok=True)
WIN = {'L55': [f'{P2}/trees/win_lf_pooled_s{s}.npy' for s in (0, 1, 2)], 'P55': [f'{P2}/trees/win_lf_perapt_s{s}.npy' for s in (0, 1, 2)],
       'K32': [f'{P2}/nets/k32lf_s{s}.npy' for s in (0, 1, 2)], 'K128': [f'{P2}/nets/k128lf_s{s}.npy' for s in (0, 1, 2)],
       'ID': [f'{P2}/arms/win_idcat_s{s}.npy' for s in (0, 1)], 'RQ': [f'{P2}/arms/win_rwyq_s{s}.npy' for s in (0, 1)],
       'CL': [f'{P2}/clean/win_clean_s{s}.npy' for s in (0, 1)], 'L55o': [f'{P2}/trees/win_oof_pooled_s{s}.npy' for s in (0, 1)],
       'P55o': [f'{P2}/trees/win_oof_perapt_s{s}.npy' for s in (0, 1)]}
fp = np.load(f'{P2}/full_package.npz'); FULL = [str(k) for k in fp['members']]; wF_arch = fp['w'].astype(float)
w0 = np.array([.2230, .2030, .0880, .4860]); FOUR = ('L55', 'P55', 'K32', 'K128')   # v73's fit, target y (s264)
Mw = {k: mean_of(WIN[k]) for k in FULL}; M9 = np.stack([Mw[k] for k in FULL], 1)
def simplex(A, t):   # the KKT enumeration of scripts/adsb_price_v110.py and p2p_checks.py
    A = A / 1000.0; t = t / 1000.0; G = A.T @ A; b = A.T @ t; n = A.shape[1]; best = None
    for rr in range(1, n + 1):
        for S in itertools.combinations(range(n), rr):
            S = list(S); k = len(S); K = np.zeros((k + 1, k + 1)); K[:k, :k] = G[np.ix_(S, S)]; K[:k, k] = 1; K[k, :k] = 1
            q = np.zeros(k + 1); q[:k] = b[S]; q[k] = 1
            try: ws = np.linalg.solve(K, q)[:k]
            except np.linalg.LinAlgError: continue
            if (ws < -1e-10).any(): continue
            wf = np.zeros(n); wf[S] = np.clip(ws, 0, None); wf /= wf.sum(); f = float(wf @ G @ wf - 2 * wf @ b)
            if best is None or f < best[0]: best = (f, wf)
    return best[1]
if WF == 'refit':   # free intercept per month: demean target and members within each month on ordinary rows
    Mc, yc = M9.copy(), y.astype(float).copy()
    for m in (1, 7):
        r = mon == m; Mc[r] -= M9[O & r].mean(0); yc[r] -= y[O & r].mean()
    wF = simplex(Mc[O], yc[O])
else:
    wF = wF_arch
print(f'bulk weights ({WF}): ' + '  '.join(f'{k} {w:.4f}' for k, w in zip(FULL, wF)) + f'   max |w - archived| {np.abs(wF - wF_arch).max():.4f}')
B0w_raw = np.stack([Mw[k] for k in FOUR], 1) @ w0; B1w_raw = M9 @ wF
# ---- labelled month offsets
off = {m: float((y - B1w_raw)[O & (mon == m)].mean()) for m in (1, 7)}
print(f'labelled month offsets (mean residual of the 9-member bulk, ordinary window rows): January {off[1]:+.4f}  July {off[7]:+.4f}')
rnd = (lambda v: round(v, 3)) if WF == 'archived' else (lambda v: v)   # archived keeps v115/v116's rounding
# The final phase (2026-10-08) adds February and June. The window holds only January and July, so their offsets come
# from the twelve-month cross-fit (scripts/xfit12.py, mean residual on ordinary rows: February +3.800, June +2.727),
# passed as SHIFT_2 and SHIFT_6. A scored month without an offset stops the build.
for m in (2, 6):
    if os.environ.get(f'SHIFT_{m}'): off[m] = float(os.environ[f'SHIFT_{m}'])
json.dump({**{f'SHIFT_{m}': rnd(v) if m in (1, 7) else v for m, v in sorted(off.items())}, 'WF': WF}, open(f'{CLEAN_DIR}/offsets.json', 'w'))
np.savez(f'{CLEAN_DIR}/wF.npz', members=np.array(FULL), w=wF, mode=WF)
def mixq(P):
    Q = P ** 0.90; Q = Q / Q.sum(1, keepdims=True); Q[:, 1:][Q[:, 1:] < 0.10] = 0.0; Q[:, 0] = 1.0 - Q[:, 1:].sum(1)
    Q = np.clip(Q, 1e-12, None); return Q / Q.sum(1, keepdims=True)
def mix(Q, bulk, vsd, vlb):
    b = cl(bulk); V = np.stack([b, vsd, vlb, cl(I.DAY + b), b], 1); return cl(np.rint((Q * V).sum(1)))
def base(Q, vsd, vlb, B0, B1):
    inc = mix(Q, B0, vsd, vlb); app = np.abs(inc - vsd) > 60
    out = inc.copy(); out[app] = cl(np.rint(inc[app] + (B1 - B0)[app])); return out
# ---- window: the same construction with the labelled offsets (the paper's evaluated model)
shw = np.where(mon == 1, off[1], off[7])
assert np.allclose(ctx['Qg'], mixq(np.load(f'{I.W}/b4ab_P_u.npy'))), 'window mixture weights'
v100w_clean = base(ctx['Qg'], ctx['vsd'], ctx['vlb'], B0w_raw + shw, B1w_raw + shw)
SHW_B = np.where(mon == 1, -8.778, -3.861)
v100w_board = base(ctx['Qg'], ctx['vsd'], ctx['vlb'], B0w_raw + SHW_B, B1w_raw + SHW_B)
rm = lambda p, m: float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
allm = np.ones(len(y), bool)
print(f'window RMSE, all / ordinary: with the leaderboard offsets {rm(v100w_board, allm):.3f} / {rm(v100w_board, O):.3f};  '
      f'clean {rm(v100w_clean, allm):.3f} / {rm(v100w_clean, O):.3f}')
# ---- scored rows
R = pq.read_table(f'{D}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']).to_pandas()
R = R[R.PHASE_mvt.eq('DEP')].drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(sub_ids)
E0 = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E0).dt.total_seconds().values
ms = sec(R.MVT_TIME_UTC_mvt) - sec(R.SCHED_TIME_UTC_mvt); lb = sec(R.MVT_TIME_UTC_mvt) - sec(R.LOBT_flt)
vsd_r = cl(ms); vlb_r = np.where(np.isnan(lb), vsd_r, cl(np.nan_to_num(lb, nan=0.0)))
assert not np.isnan(vsd_r).any()
G = pd.read_parquet(f'{I.W}/b4_gate_rk.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)
P_r = G[[f'P{i}' for i in range(5)]].to_numpy(float); assert np.isfinite(P_r).all(), 'gate probabilities for every scored row'
MRk = pd.read_parquet(f'{I.W}/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(sub_ids)
Mr = {k: MRk[k].values.astype(float) for k in FULL}; assert all(np.isfinite(v).all() for v in Mr.values())
mon_r = pd.to_datetime(R.MVT_TIME_UTC_mvt, utc=True).dt.month.values
assert set(np.unique(mon_r)) <= set(off), f'scored months {sorted(set(np.unique(mon_r)))} without an offset (set SHIFT_2 / SHIFT_6)'
shr = pd.Series(mon_r).map(off).to_numpy(float)
B0r = np.stack([Mr[k] for k in FOUR], 1) @ w0 + shr; B1r = np.stack([Mr[k] for k in FULL], 1) @ wF + shr
v100r_clean = base(mixq(P_r), vsd_r, vlb_r, B0r, B1r)
ship = I.sub('v100') if os.path.exists('submissions/quick-boat_v100.parquet') else None
tpl = pq.read_table(f'{D}/submitting.parquet')
o = pd.DataFrame({'MVT_ID_mvt': sub_ids, 'TAXITIME_SEC_mvt': v100r_clean.astype('int32')})
t = pa.Table.from_pandas(o, schema=tpl.schema, preserve_index=False); assert t.schema.equals(tpl.schema)
OUT = f'{CLEAN_DIR}/submissions/quick-boat_v100.parquet'; pq.write_table(t, OUT)
np.savez(f'{CLEAN_DIR}/base_window.npz', win_ids=ctx['ids'], newW=v100w_clean, ids=sub_ids, new=v100r_clean)
if ship is not None and len(ship) == len(v100r_clean):
    d = v100r_clean - ship
    print(f'scored rows: {int((d != 0).sum()):,} of {len(d):,} differ from the shipped v100, RMS {np.sqrt(np.mean(d ** 2)):.2f} s, '
          f'Q_b {float((d ** 2).sum()):.4e}; rows moved more than 600 s: {int((np.abs(d) > 600).sum())}')
print(f'wrote {OUT}')
