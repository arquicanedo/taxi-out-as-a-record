"""v106 = v104 + the REFINED ADS-B stage alone: v104's stage features plus operator, wake class and the parked /
first-sighting position (s523). No weather (s528 killed it). Priced as an increment over the shipped v104 on the VALIDATED
Jan|Jul instrument, which retrodicted +2.8692e7 at 3.76 SE for exactly this change (s528).
Feature table: artifacts/adsb_push_feat_v105.parquet (ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_XY=1; ODbL).
  uv run --with pandas --with pyarrow --with numpy --with lightgbm python scripts/adsb_price_v106.py  -> cand/v106_adsb.npz
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v105.parquet')); os.environ.setdefault('ADSB_FOLDS', 'janjul')   # the validated instrument (v104 at 1.05x, s521)
__file__ = os.path.join(HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
assert EXTRA == ['ab_path', 'ab_tmove', 'ab_hold', 'ab_pk_dist', 'ab_pk_age', 'ab_pk_gs', 'ab_taxi_pk', 'ab_px', 'ab_py', 'ab_qx', 'ab_qy'], EXTRA
# operator (top 60 on covered window rows, callsign-prefix fallback) and wake class, as s523
cols_ = ['MVT_ID_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt', 'WK_TBL_CAT_flt']
OP = pd.concat([pq.read_table(f, columns=cols_).to_pandas() for f in (f'{D}/training_2025-01-01_2025-02-01.parquet',
     f'{D}/training_2025-07-01_2025-08-01.parquet', f'{D}/ranking.parquet')]).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt')
opn = OP.AIRCRAFT_OPERATOR_flt.fillna(OP.FLIGHT_mvt.astype(str).str.extract(r'^([A-Z]{2,3})', expand=False)).fillna('?')
top = opn.reindex(ctx['ids'])[has].value_counts().index[:60]; OC = {k: i for i, k in enumerate(top)}
enc = lambda s: pd.Categorical([OC.get(v, -1) for v in s], categories=range(-1, len(OC)))
X['oper'] = enc(opn.reindex(ctx['ids']).values); Xr['oper'] = enc(opn.reindex(sub_ids).values)
WCL = {'L': 0, 'M': 1, 'H': 2, 'J': 3}
X['wtc'] = OP.WK_TBL_CAT_flt.reindex(ctx['ids']).map(WCL).values; Xr['wtc'] = OP.WK_TBL_CAT_flt.reindex(sub_ids).map(WCL).values
tmw_ = X.tm.values == 1; NEW = ('px', 'py', 'qx', 'qy', 'oper', 'wtc')
C104 = [c for c in X.columns if c != 'rwy' and c not in NEW]; C105 = [c for c in X.columns if c != 'rwy']
kw = dict(app_w=has & (v100w < 3600), app_r=hasr & (v100r < 3600), fit_w=has & ~tmw_ & (y < 7200))
r104 = run('v104 stage (incumbent)', cols=C104, **kw)
v104r = I.sub('v104'); assert np.array_equal(r104['nR'], v104r), 'day6 refit must leave the scored side equal to shipped v104'
r106 = run('v106 refined stage', cols=C105, **kw)
iw, v106w, v106r = r104['nW'], r106['nW'], r106['nR']
assert np.array_equal(iw, np.load(f'{W}/adsblol/cand/v104_adsb.npz')['newW'].astype(float)), 'incumbent window vector must be v104 as priced (s520)'
lin = -2.0 * (iw - y) * (v106w - iw); own = (iw - y) ** 2 - (v106w - y) ** 2
Qb = float(((v106r - v104r) ** 2).sum()); m2T = float(lin.sum()) * sc
cs = np.bincount(clus, weights=lin); rng = np.random.default_rng(5); k = len(cs)
SE = float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(2000)]).std()) * sc; G = m2T - Qb
print(f'\nINCREMENT v106 - v104: window own {own.sum()*sc:+.4e}  rows R {int((v106r != v104r).sum()):,}  max |change| {np.abs(v106r - v104r).max():.0f} s\n'
      f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  = {G/ONE:+.3f} bs  -> {np.sqrt(233.1847**2 - G/344841):.4f}')
print('  per airport own: ' + '  '.join(f'{a} {own[apt == a].sum()*sc:+.2e}' for a in sorted(AC)))
np.savez(f'{W}/adsblol/cand/v106_adsb.npz', ids=sub_ids, v104=v104r, new=v106r, win_ids=ctx['ids'], incW=iw, newW=v106w)
