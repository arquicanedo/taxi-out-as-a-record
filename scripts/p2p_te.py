"""p2p step 0(b): leak-free target encodings for the Phase 2 bulk package (s492 P).

The eight te_* columns stored in v15_tr_jul19.parquet were computed by v8s_build.py from ALL twelve 2025
months (s487), so a window row's encoding contains its own label. This script recomputes them with the
v8s_build.py formula (smoothed mean, q=50; 100 for te_des / te_apt_des; 20 for te_apt_null; source = all
labelled rows y>0 of the fitting months, extremes included; unseen key -> NaN) and writes:

  te12        all twelve months, applied to every v15_tr row.  CONTROL: must equal the stored columns.
  te12_rk     all twelve months, applied to the 344,841 v15_rk rows. CONTROL: must equal v15_rk's stored columns.
  te10        the ten non-window months (2-6, 8-12), applied to every v15_tr row (training rows in-sample,
              window rows out-of-sample): the leak-free window replica's encodings.
  te10_oof    variant: training-month rows get OUT-OF-FOLD encodings (5 folds by calendar day, fitted on the
              other four folds of the ten months); window rows get te10.
  te12_oof    variant, ranking side: all twelve months' rows get out-of-fold encodings (5 folds by calendar day
              over the twelve months); ranking rows keep te12_rk (the stored production encodings).

Output: <OUT>/te.npz (arrays in v15_tr_jul19 / v15_rk_jul19 file order, plus the MVT_ID order for assertion).
"""
import os, time, numpy as np, pandas as pd
W = os.environ['DC26_WORK']
OUT = os.environ.get('P2P_OUT', os.environ.get('DC26_SESSION', os.environ['DC26_WORK'] + '/session') + '/p2P')
os.makedirs(OUT, exist_ok=True); t0 = time.time()
SPEC = [(['apt', 'STAND_mvt', 'RUNWAY_mvt'], 'te_stand_rwy', 50), (['apt', 'STAND_mvt'], 'te_stand', 50),
        (['apt', 'RUNWAY_mvt'], 'te_rwy', 50), (['apt', 'AIRCRAFT_OPERATOR_flt'], 'te_op', 50), (['apt'], 'te_apt', 50),
        (['ADES_mvt'], 'te_des', 100), (['apt', 'ADES_mvt'], 'te_apt_des', 100), (['apt', 'aobt_null'], 'te_apt_null', 20)]
TE = [s[1] for s in SPEC]
KC = ['apt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_mvt', 'aobt_null']
tr = pd.read_parquet(f'{W}/v15_tr_jul19.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt', 'y'] + KC + TE)
rk = pd.read_parquet(f'{W}/v15_rk_jul19.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'] + KC + TE)
for d in (tr, rk):
    for c in KC[:-1]: d[c] = d[c].astype(object)
tm = pd.to_datetime(tr.MVT_TIME_UTC_mvt, utc=True); mon = tm.dt.month.values
day = tm.dt.floor('D').values
WIN = np.isin(mon, [1, 7]); y = tr.y.values.astype(float)
N_RK = len(pd.read_parquet(os.environ['DC26_DATA'] + '/submitting.parquet', columns=['MVT_ID_mvt']))   # 344,841 for Jan/Jul; 670,790 for the final's four months
assert len(tr) == 2084659 and int(WIN.sum()) == 344336 and len(rk) == N_RK, (len(tr), int(WIN.sum()), len(rk), N_RK)
assert (y > 0).all(), 'v8s encodes on y>0 rows; v15_tr should hold only those'
print(f'v15_tr {len(tr):,} rows (window {int(WIN.sum()):,}), v15_rk {len(rk):,}   [{time.time()-t0:.0f}s]', flush=True)

def enc(src_mask, targets):
    """v8s formula fitted on tr[src_mask]; returns {te_name: values on each frame in targets}."""
    src = tr[src_mask]; out = {}
    for k, n, q in SPEC:
        gm = src.y.mean(); g = src.groupby(k, observed=True).y.agg(['mean', 'size'])
        sm = (g['mean'] * g['size'] + gm * q) / (g['size'] + q)
        out[n] = [d.set_index(k).index.map(sm).astype(float).values for d in targets]
    return out

def same(a, b): return np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b))

# ---- CONTROL: twelve months reproduce the stored columns, on both files --------------------------------
e12 = enc(np.ones(len(tr), bool), [tr, rk])
te12 = np.stack([e12[n][0] for n in TE], 1); te12_rk = np.stack([e12[n][1] for n in TE], 1)
for j, n in enumerate(TE):
    a = same(te12[:, j], tr[n].values); b = same(te12_rk[:, j], rk[n].values)
    print(f'  CONTROL {n:<13} 12-month recompute == stored: v15_tr {a.mean():.6%}  v15_rk {b.mean():.6%}', flush=True)
    assert a.all() and b.all(), f'{n}: twelve-month recompute does not reproduce the stored encoding'
print('  CONTROL PASSED: the v8s formula on all twelve months reproduces every stored te_* on 100% of rows', flush=True)

# ---- ten months ---------------------------------------------------------------------------------------
e10 = enc(~WIN, [tr])
te10 = np.stack([e10[n][0] for n in TE], 1)
for j, n in enumerate(TE):
    d = np.abs(te10[:, j] - te12[:, j])
    print(f'  te10 {n:<13} window mean|te10-te12| {np.nanmean(d[WIN]):8.3f}  exact-equal on window {same(te10[WIN, j], te12[WIN, j]).mean():.4%}'
          f'  NaN on window {np.isnan(te10[WIN, j]).mean():.4%}', flush=True)

# ---- out-of-fold by calendar day ----------------------------------------------------------------------
def oof(base_mask, seed):
    days = np.unique(day[base_mask]); perm = np.random.default_rng(seed).permutation(len(days))
    fmap = pd.Series(np.arange(len(days)) % 5, index=days[perm])                # each day -> fold 0..4
    fold = np.full(len(tr), -1); fold[base_mask] = fmap.reindex(day[base_mask]).values
    out = np.full((len(tr), len(TE)), np.nan)
    for k in range(5):
        e = enc(base_mask & (fold != k), [tr]); app = base_mask & (fold == k)
        for j, n in enumerate(TE): out[app, j] = e[n][0][app]
    return out, fold

o10, f10 = oof(~WIN, 10)
te10_oof = np.where(WIN[:, None], te10, o10)
o12, f12 = oof(np.ones(len(tr), bool), 12)
te12_oof = o12
for nm, a, m in (('te10_oof (training rows)', te10_oof, ~WIN), ('te12_oof (all 2025 rows)', te12_oof, np.ones(len(tr), bool))):
    ref = te10 if nm.startswith('te10') else te12
    print(f'  {nm}: folds sizes {np.bincount((f10 if nm.startswith("te10") else f12)[m])}  '
          f'mean|oof - in-sample| by column ' + ' '.join(f'{n}:{np.nanmean(np.abs(a[m, j]-ref[m, j])):.2f}' for j, n in enumerate(TE)), flush=True)
assert same(te10_oof[WIN], te10[WIN]).all()
np.savez(f'{OUT}/te.npz', te_cols=np.array(TE), ids_tr=tr.MVT_ID_mvt.values, ids_rk=rk.MVT_ID_mvt.values,
         te12=te12, te12_rk=te12_rk, te10=te10, te10_oof=te10_oof, te12_oof=te12_oof, fold10=f10, fold12=f12)
print(f'wrote {OUT}/te.npz   [{time.time()-t0:.0f}s]', flush=True)
