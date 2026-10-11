"""The Istanbul local-state correction (s601, s604): LTFM departures move by half the predicted residual of a model that
sees the airport's state around each take-off. Built from labelled 2025 data and the scored file's own columns.

STATUS: A CANDIDATE THAT WAS TESTED AND NOT ADOPTED. It is not part of the entry and run_final.sh does not call it. Its
January and July rows were scored once, as read 4 (quick-boat_v131.parquet, scripts/final/read4_build.py): 234.2287
against the entry's 233.7464 (v130), expected 233.27 [233.51, 233.04] beforehand. It is kept so that v131 can be rebuilt.

WHY. Istanbul's network off-block (AOBT_3) is a lookup, take-off minus a runway constant, on ~90% of rows (s601), so the
model has no off-block observation there. What it can see is the state around a departure: how late the neighbouring
departures are against their plans, the take-off sequence on the runway, and the labelled arrivals' taxi-in. An
independent session found that a correction learned from these is positive in every 2025 month (s601; its exploratory
scripts s09 and s12 are not part of the repository). This script ports it, for LTFM only, to the four scored months.

THE CORRECTION.
  features  per departure, from the challenge columns only (no departure label):
              base        MVT - AOBT_3, MVT - LOBT, MVT - SCHED, MVT - EOBT_1, LOBT - SCHED, hour, weekday, runway, stand,
                          type, operator, destination, wake class, market segment (codes)
              sequence    same-runway take-offs in the previous 5/15/30 min and the next 5 min, gaps to the previous and
                          next same-runway take-off, airport take-offs in the previous 15 min
              neighbours  leave-one-out medians over the other departures within 30 min (same runway, whole airport)
                          of MVT - AOBT_3 and MVT - LOBT, of MVT - SCHED (30 min) and MVT - AOBT_3 (60 min), and the
                          share of them more than an hour behind their LOBT
              arrivals    the labelled arrivals' taxi-in anomaly (against the 2025 median by runway and stand, months
                          other than January and July) within 30/60/120 min and over the day, their taxi-in and counts
  base      LightGBM on y (y < 7,200), 2025 LTFM departures, cross-fitted by UTC month (own month left out); rb = y - base
  residual  LightGBM on rb over all four feature groups (y < 7,200): cross-fitted by month for the 2025 window replay,
            fitted on all of 2025 for the scored rows
  apply     new = clip(rint(current + 0.5 * residual), 60, 200000) on LTFM departures; every other row is unchanged.
            The strength 0.5 was chosen on the 2025 window by the session that found the lever (scale 1.0 did not clear
            zero) and fixed before this port was run.

THE GATES, printed and asserted before anything is written (fixed in advance; all three passed on 2026-10-11):
  (a) the replay's all-row window gain at 0.5 against the final model's own window vector has a day-clustered 95%
      interval above zero; (b) it is positive in January and in July; (c) no feature lies outside its 2025 range on
      more than 1% of the scored LTFM rows. GATES=0 writes the file anyway, for inspection only.

  uv run --quiet --with pandas --with numpy --with pyarrow --with lightgbm --with scikit-learn python scripts/final/ltfm_local.py \\
      --data DIR --base PARQUET --window V117_NPZ --out PARQUET
BASE is submissions/quick-boat_final.parquet (the entry); OUT is the candidate, sha256
ba5b0186bb250118f01de4b72977191aa7851cadadbb32c436bca92f88d387af with the pinned library versions of run_final.sh.
DIR holds the twelve training_2025-*.parquet and ranking.parquet (final_ranking); V117_NPZ is the clean chain's window
vector at v117 ($DC26_WORK/clean/cand/v117_adsb.npz: win_ids, newW); the join-bound clip, the only later rule that
touches LTFM, is applied to it here.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--base', required=True)
ap.add_argument('--window', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
APT, S, FLOOR, CAP, TOL, YMAX = 'LTFM', 0.5, 60.0, 200000.0, 3606.0, 7200.0
GATES = os.environ.get('GATES', '1') == '1'
C = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt',
     'EOBT_1_flt', 'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'WK_TBL_CAT_flt',
     'MARKET_SEGMENT_flt', 'TAXITIME_SEC_mvt']
LGB = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, deterministic=True,
           force_col_wise=True, n_jobs=8)
BASE_P = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, **LGB)
RES_P = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, reg_lambda=10, **LGB)

# ---------------------------------------------------------------- rows touching LTFM, both years
files = sorted(glob.glob(os.path.join(a.data, 'training_2025-*.parquet')))
assert len(files) == 12, f'expected twelve 2025 files in {a.data}'
def load(f, year):
    d = pq.read_table(f, columns=C).to_pandas()
    d = d[d.ADEP_mvt.eq(APT) & d.PHASE_mvt.eq('DEP') | d.ADES_mvt.eq(APT) & d.PHASE_mvt.eq('ARR')].copy()
    d['year'] = year; return d
al = pd.concat([load(f, 2025) for f in files] + [load(os.path.join(a.data, 'ranking.parquet'), 2026)], ignore_index=True)
for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt', 'AOBT_3_flt', 'EOBT_1_flt'):
    al[c] = pd.to_datetime(al[c], utc=True)
E = pd.Timestamp('1970-01-01', tz='UTC')
al['t'] = (al.MVT_TIME_UTC_mvt - E).dt.total_seconds()
assert 1.73e9 < al.t.median() < 1.80e9, al.t.median()
assert al.MVT_TIME_UTC_mvt.dt.year.eq(al.year).all(), 'a row outside its file year'
al['y'] = al.TAXITIME_SEC_mvt.astype(float)
sec = lambda x: x.dt.total_seconds()
al['ma'] = sec(al.MVT_TIME_UTC_mvt - al.AOBT_3_flt); al['ml'] = sec(al.MVT_TIME_UTC_mvt - al.LOBT_flt)
al['ms'] = sec(al.MVT_TIME_UTC_mvt - al.SCHED_TIME_UTC_mvt); al['me'] = sec(al.MVT_TIME_UTC_mvt - al.EOBT_1_flt)
al['ls'] = sec(al.LOBT_flt - al.SCHED_TIME_UTC_mvt)
al['date'] = al.MVT_TIME_UTC_mvt.dt.floor('D'); al['month'] = al.MVT_TIME_UTC_mvt.dt.month
al['hod'] = al.MVT_TIME_UTC_mvt.dt.hour; al['dow'] = al.MVT_TIME_UTC_mvt.dt.dayofweek
CAT = ['RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'ADES_mvt', 'WK_TBL_CAT_flt', 'MARKET_SEGMENT_flt']
for c in CAT: al[c + '_c'] = al[c].astype('category').cat.codes
dep, arr = al[al.PHASE_mvt.eq('DEP')].copy(), al[al.PHASE_mvt.eq('ARR')].copy()
assert not dep.MVT_ID_mvt.duplicated().any()
assert dep[dep.year == 2025].y.notna().all() and dep[dep.year == 2026].y.isna().all(), 'labels: 2025 present, 2026 blank'
assert arr.y.notna().all(), 'arrivals are labelled in both years'
print(f'{APT} departures 2025 {int((dep.year == 2025).sum()):,}, 2026 {int((dep.year == 2026).sum()):,} '
      f'(by month {dict(dep[dep.year == 2026].month.value_counts().sort_index())}); arrivals {len(arr):,}')

# ---------------------------------------------------------------- arrivals' taxi-in anomaly
tr = arr[(arr.year == 2025) & ~arr.month.isin([1, 7]) & (arr.y < YMAX)]
e3 = tr.groupby(['RUNWAY_mvt', 'STAND_mvt']).y.agg(['median', 'size']); e3 = e3[e3['size'] >= 20]['median'].rename('e3')
e2 = tr.groupby('RUNWAY_mvt').y.median().rename('e2')
arr = arr.join(e3, on=['RUNWAY_mvt', 'STAND_mvt']).join(e2, on='RUNWAY_mvt')
arr['anom'] = (arr.y - arr.e3.fillna(arr.e2).fillna(tr.y.median())).clip(-1800, 1800)

def win_stat(tq, tv, v, half, stat):
    o = np.argsort(tv); tv, v = tv[o], v[o]
    lo, hi = np.searchsorted(tv, tq - half, 'left'), np.searchsorted(tv, tq + half, 'right'); out = np.full(len(tq), np.nan)
    for i in range(len(tq)):
        w = v[lo[i]:hi[i]]; w = w[~np.isnan(w)]
        if len(w): out[i] = np.median(w) if stat == 'median' else w.mean()
    return out

def loo_median(ts, v, half):
    lo, hi = np.searchsorted(ts, ts - half, 'left'), np.searchsorted(ts, ts + half, 'right'); out = np.full(len(ts), np.nan)
    for i in range(len(ts)):
        w = np.delete(v[lo[i]:hi[i]], i - lo[i]); w = w[~np.isnan(w)]
        if len(w): out[i] = np.median(w)
    return out

def build(g, ar):
    g = g.sort_values('t').copy(); t = g.t.values; F = {}
    for _, idx in g.groupby('RUNWAY_mvt').indices.items():
        o = np.argsort(t[idx], kind='stable'); ts = t[idx][o]; at = idx[o]
        for name, half in (('cnt_rwy_prev5', 300), ('cnt_rwy_prev15', 900), ('cnt_rwy_prev30', 1800)):
            F.setdefault(name, np.full(len(g), np.nan))[at] = np.searchsorted(ts, ts, 'left') - np.searchsorted(ts, ts - half, 'left')
        F.setdefault('cnt_rwy_next5', np.full(len(g), np.nan))[at] = np.searchsorted(ts, ts + 300, 'right') - np.searchsorted(ts, ts, 'right')
        prev = np.full(len(ts), np.nan); prev[1:] = np.diff(ts); nxt = np.full(len(ts), np.nan); nxt[:-1] = np.diff(ts)
        F.setdefault('gap_prev_rwy', np.full(len(g), np.nan))[at] = prev; F.setdefault('gap_next_rwy', np.full(len(g), np.nan))[at] = nxt
        for col, name in (('ma', 'nb_ma_rwy30'), ('ml', 'nb_ml_rwy30')):
            F.setdefault(name, np.full(len(g), np.nan))[at] = loo_median(ts, g[col].values[at].astype(float), 1800)
    F['cnt_apt_prev15'] = np.searchsorted(t, t, 'left') - np.searchsorted(t, t - 900, 'left')
    for col, half, name in (('ma', 1800, 'nb_ma_apt30'), ('ml', 1800, 'nb_ml_apt30'), ('ms', 1800, 'nb_ms_apt30'), ('ma', 3600, 'nb_ma_apt60')):
        F[name] = loo_median(t, g[col].values.astype(float), half)
    late = (g.ml.values > 3600).astype(float); late[np.isnan(g.ml.values)] = np.nan
    F['nb_late_share30'] = win_stat(t, t, late, 1800, 'mean')
    ta, an, ay = ar.t.values, ar.anom.values.astype(float), ar.y.values.astype(float)
    F['arr_anom30'] = win_stat(t, ta, an, 1800, 'median'); F['arr_anom60'] = win_stat(t, ta, an, 3600, 'median')
    F['arr_anom120'] = win_stat(t, ta, an, 7200, 'median'); F['arr_taxi30'] = win_stat(t, ta, ay, 1800, 'median')
    tas = np.sort(ta)
    F['arr_cnt30'] = np.searchsorted(tas, t + 1800, 'right') - np.searchsorted(tas, t - 1800, 'left')
    F['arr_cnt15'] = np.searchsorted(tas, t + 900, 'right') - np.searchsorted(tas, t - 900, 'left')
    g = g.join(ar.groupby('date').anom.median().rename('arr_anom_day'), on='date')
    for k, v in F.items(): g[k] = v
    return g

x = pd.concat([build(dep[dep.year == yr], arr[arr.year == yr]) for yr in (2025, 2026)])
BASE = ['ma', 'ml', 'ms', 'me', 'ls', 'hod', 'dow'] + [c + '_c' for c in CAT]
NB = ['nb_ma_rwy30', 'nb_ml_rwy30', 'nb_ma_apt30', 'nb_ml_apt30', 'nb_ms_apt30', 'nb_ma_apt60', 'nb_late_share30']
SEQ = ['cnt_rwy_prev5', 'cnt_rwy_prev15', 'cnt_rwy_prev30', 'cnt_rwy_next5', 'gap_prev_rwy', 'gap_next_rwy', 'cnt_apt_prev15']
AR = ['arr_anom30', 'arr_cnt30', 'arr_anom60', 'arr_anom120', 'arr_taxi30', 'arr_cnt15', 'arr_anom_day']
FEATS = NB + SEQ + AR + BASE
x25, x26 = x[x.year == 2025].copy(), x[x.year == 2026].copy()
print(f'features built: 2025 {len(x25):,}, 2026 {len(x26):,}; median neighbour MVT - LOBT 2025 {x25.nb_ml_apt30.median():.0f} s, '
      f'2026 {x26.nb_ml_apt30.median():.0f} s; median arrival taxi-in 2025 {x25.arr_taxi30.median():.0f} s')

# ---------------------------------------------------------------- base and residual, cross-fitted by month
ordm = x25.y < YMAX
x25['bp'] = np.nan
for m in range(1, 13):
    trm = (x25.month != m) & ordm
    x25.loc[x25.month == m, 'bp'] = lgb.LGBMRegressor(**BASE_P).fit(x25.loc[trm, BASE], x25.loc[trm, 'y']).predict(x25.loc[x25.month == m, BASE])
x25['rb'] = x25.y - x25.bp
x25['rp'] = np.nan
for m in range(1, 13):
    trm = (x25.month != m) & ordm
    x25.loc[x25.month == m, 'rp'] = lgb.LGBMRegressor(**RES_P).fit(x25.loc[trm, FEATS], x25.loc[trm, 'rb']).predict(x25.loc[x25.month == m, FEATS])
o = x25[ordm]; gm = (o.rb ** 2 - (o.rb - o.rp) ** 2).groupby(o.month).sum()
print(f'own-base gain by month (ordinary rows, cross-fitted): ' + ' '.join(f'{m}:{v:+.2e}' for m, v in gm.items())
      + f'   positive in {int((gm > 0).sum())}/12')

# ---------------------------------------------------------------- the replay on the final model's window vector
z = np.load(a.window); wid, wv = z['win_ids'], z['newW'].astype(float)
W = x25.set_index('MVT_ID_mvt').reindex(pd.Index(wid, name='MVT_ID_mvt')).dropna(subset=['month'])
W = W[W.month.isin([1, 7])]
W['cur'] = pd.Series(wv, index=wid).reindex(W.index).values
lo, hi = np.maximum(W.ml - TOL, FLOOR), W.ml + TOL; jp = W.ml.notna()
assert int(((W.y < W.ml - TOL) | (W.y > hi))[jp].sum()) == 0, 'a window label outside the join bound'
W.loc[jp, 'cur'] = np.clip(np.rint(np.clip(W.cur[jp], lo[jp], hi[jp])), FLOOR, CAP)
W['new'] = np.clip(np.rint(W.cur + S * W.rp), FLOOR, CAP)
print(f'window: {len(W):,} LTFM rows (Jan {int((W.month == 1).sum()):,}, Jul {int((W.month == 7).sum()):,}); '
      f'current RMSE {np.sqrt(((W.cur - W.y) ** 2).mean()):.3f}, new {np.sqrt(((W.new - W.y) ** 2).mean()):.3f}')
def ci(g, days):
    gd = g.groupby(days).sum(); se = np.sqrt(len(gd) * gd.var(ddof=1)); return gd.sum(), gd.sum() - 1.96 * se, gd.sum() + 1.96 * se
res = {}
for nm, m in (('all rows', np.ones(len(W), bool)), ('ordinary (y < 7200)', (W.y < YMAX).values)):
    w = W[m]; g = (w.cur - w.y) ** 2 - (w.new - w.y) ** 2; G, l, h = ci(g, w.date)
    res[nm] = (G, l, h, g[w.month == 1].sum(), g[w.month == 7].sum())
    print(f'  {nm:20s} gain {G:+.4e} [{l:+.4e}, {h:+.4e}]   Jan {res[nm][3]:+.4e}  Jul {res[nm][4]:+.4e}')
for s in (0.25, 0.5, 0.75, 1.0):   # printed for the record; the strength is fixed at 0.5
    g = (W.cur - W.y) ** 2 - (np.clip(np.rint(W.cur + s * W.rp), FLOOR, CAP) - W.y) ** 2
    print(f'    strength {s:.2f}: all-row gain {g.sum():+.4e}')
gtop = ((W.cur - W.y) ** 2 - (W.new - W.y) ** 2).sort_values(ascending=False)
print(f'  largest single-row gain {gtop.iloc[0]:+.3e} ({100 * gtop.iloc[0] / max(res["all rows"][0], 1):.1f}% of the total); '
      f'largest loss {gtop.iloc[-1]:+.3e}; top two days ' + ', '.join(f'{d.date()} {v:+.2e}' for d, v in
      ((W.cur - W.y) ** 2 - (W.new - W.y) ** 2).groupby(W.date).sum().sort_values(ascending=False).head(2).items()))

if os.environ.get('DUMP'):   # inspection only: the cross-fitted 2025 rows and the window replay
    x25[['MVT_ID_mvt', 'date', 'month', 'y', 'ml', 'bp', 'rb', 'rp']].to_parquet(os.environ['DUMP'] + '_x25.parquet')
    W.reset_index()[['MVT_ID_mvt', 'date', 'month', 'y', 'cur', 'new', 'rp', 'ml']].to_parquet(os.environ['DUMP'] + '_window.parquet')

# ---------------------------------------------------------------- the scored rows
fit = x25[ordm]
x26['rp'] = lgb.LGBMRegressor(**RES_P).fit(fit[FEATS], fit.rb).predict(x26[FEATS])
out_sup = {c: float(((x26[c] < x25[c].min()) | (x26[c] > x25[c].max())).mean()) for c in FEATS}
worst = max(out_sup, key=out_sup.get)
print(f'support: the feature most often outside its 2025 range is {worst} at {100 * out_sup[worst]:.3f}% of scored rows')

bt = pq.read_table(a.base); yt = bt.schema.field('TAXITIME_SEC_mvt').type
ids = bt.column('MVT_ID_mvt').to_numpy(); base = bt.column('TAXITIME_SEC_mvt').to_numpy(zero_copy_only=False).astype(float)
assert len(np.unique(ids)) == len(ids)
pos = pd.Index(ids).get_indexer(x26.MVT_ID_mvt.values); assert (pos >= 0).all(), 'an LTFM departure missing from the base'
cur = base[pos]; new = np.clip(np.rint(cur + S * x26.rp.values), FLOOR, CAP); d = new - cur
print(f'scored LTFM rows {len(pos):,}: moved {int((d != 0).sum()):,}, by more than 60 s {int((np.abs(d) > 60).sum()):,}; '
      f'mean move {d.mean():+.2f} s, sd {d.std():.1f}')
for m in sorted(x26.month.unique()):
    k = x26.month.values == m
    print(f'  month {m}: rows {int(k.sum()):6,}  mean move {d[k].mean():+6.2f}  sd {d[k].std():6.1f}  |move| > 60 s {int((np.abs(d[k]) > 60).sum()):5,}  Q {float((d[k] ** 2).sum()):.4e}')
print(f'  Q all months {float((d ** 2).sum()):.4e}')

ga, gl, gh, gj, gu = res['all rows']
gates = {'(a) all-row window gain 95% interval above zero': gl > 0, '(b) positive in January and in July': gj > 0 and gu > 0,
         '(c) no feature outside its 2025 range on more than 1% of scored rows': out_sup[worst] <= 0.01}
for k, v in gates.items(): print(f'GATE {k}: {"PASS" if v else "FAIL"}')
if not all(gates.values()) and GATES:
    sys.exit('a gate failed: nothing written (GATES=0 writes for inspection only)')

out = base.copy(); out[pos] = new
keep = np.setdiff1d(np.arange(len(out)), pos); assert np.array_equal(out[keep], base[keep]), 'a non-LTFM row changed'
arr_out = pa.array(np.rint(out).astype(yt.to_pandas_dtype()) if pa.types.is_integer(yt) else out, type=yt)
ot = bt.set_column(bt.schema.get_field_index('TAXITIME_SEC_mvt'), bt.schema.field('TAXITIME_SEC_mvt'), arr_out)
assert ot.schema.equals(bt.schema, check_metadata=True)
pq.write_table(ot, a.out)
import hashlib
print(f'wrote {a.out}  {len(out):,} rows, schema = base, sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()}')
