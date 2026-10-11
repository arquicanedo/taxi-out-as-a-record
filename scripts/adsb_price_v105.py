"""v105 = v104 + (1) the REFINED ADS-B stage: v104's features plus operator, wake class and the parked / first-sighting
position (s523), and (2) the hour-level WEATHER stage applied only to rows ADS-B cannot see (s524). Priced as an
increment over the shipped v104, on day-fold window vectors (ADSB_FOLDS=day6, both months in training, as the scored
side is trained); the day-fold refit of v104's own stage must leave the scored side equal to shipped v104 (asserted).
Feature table: artifacts/adsb_push_feat_v105.parquet (ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_XY=1; ODbL); weather:
artifacts/metar_iem_asos.parquet (scripts/pull_metar.py, IEM ASOS, public domain, archived 2026-09-25).
  uv run --with pandas --with pyarrow --with numpy --with lightgbm python scripts/adsb_price_v105.py  -> cand/v105_adsb.npz
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ.setdefault('ADSB_FEAT', os.path.join(REPO, 'artifacts', 'adsb_push_feat_v105.parquet')); os.environ.setdefault('ADSB_FOLDS', 'day6')   # s528: ADSB_FOLDS=janjul retrodicts on the validated instrument
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
r104 = run('v104 stage, day6 (incumbent)', cols=C104, **kw)
v104r = I.sub('v104'); assert np.array_equal(r104['nR'], v104r), 'day6 refit must leave the scored side equal to shipped v104'
r105 = run('v105 refined stage, day6', cols=C105, **kw)
bw, br = r105['nW'], r105['nR']
# ---------------- weather at or before take-off ------------------------------------------------------------------
Mt = pd.read_parquet(os.environ.get('METAR', os.path.join(REPO, 'artifacts', 'metar_iem_asos.parquet')))   # archived IEM pull, public domain.sort_values('valid')
wx = Mt.wxcodes.fillna('').astype(str)
for k in ('TS', 'SN', 'FZ', 'RA', 'SH', 'FG', 'BR', 'DZ'): Mt[f'w_{k}'] = wx.str.contains(k).astype(float)
Mt['ceil'] = np.where(Mt.skyc1.isin(['BKN', 'OVC', 'VV']), Mt.skyl1, 99999.0); Mt['lvp'] = ((Mt.vsby < 0.5) | (Mt.ceil < 200)).astype(float)
Mt = Mt.sort_values(['station', 'valid'], kind='stable').reset_index(drop=True)
for k in ('TS', 'SN', 'FZ'):   # any in the last 2 h, per station; rows are station-then-time ordered, so the concat aligns
    Mt[f'w_{k}_2h'] = np.concatenate([g.set_index('valid')[f'w_{k}'].rolling('2h').max().values for _, g in Mt.groupby('station', sort=False)])
Mt = Mt.sort_values('valid', kind='stable').reset_index(drop=True)
WC = ['tmpf', 'dwpf', 'sknt', 'vsby', 'ceil', 'p01i', 'lvp'] + [f'w_{k}' for k in ('TS', 'SN', 'FZ', 'RA', 'SH', 'FG', 'BR', 'DZ')] + ['w_TS_2h', 'w_SN_2h', 'w_FZ_2h']
def wfeat(ts, ap, rwy, base):
    d = pd.DataFrame({'i': np.arange(len(ap)), 'station': ap, 'valid': pd.to_datetime(ts, utc=True).reset_index(drop=True)}).sort_values('valid')
    j = pd.merge_asof(d, Mt[['station', 'valid', 'drct'] + WC], on='valid', by='station', direction='backward', tolerance=pd.Timedelta('70min')).sort_values('i')
    Z = j[WC].reset_index(drop=True)
    hd = pd.to_numeric(pd.Series(rwy).astype(str).str.extract(r'^(\d{2})', expand=False), errors='coerce').values * 10.0
    ang = np.radians(j.drct.values - hd); Z['headwind'] = j.sknt.values * np.cos(ang); Z['xwind'] = np.abs(j.sknt.values * np.sin(ang))
    Z['spread'] = Z.tmpf - Z.dwpf; Z['base'] = base; Z['hour'] = pd.to_datetime(ts, utc=True).dt.hour.values
    Z['apt'] = pd.Categorical(ap, categories=sorted(AC)); return Z

Zw = wfeat(tr.MVT_TIME_UTC_mvt, apt, RW.reindex(ctx['ids']).values, bw)
Zr = wfeat(R.MVT_TIME_UTC_mvt, aptr, RW.reindex(sub_ids).values, br)
print(f'METAR joined: window {Zw.tmpf.notna().mean():.1%}, scored {Zr.tmpf.notna().mean():.1%}')
WX = [c for c in Zw.columns if c not in ('base', 'hour', 'apt')]; CW = ['base', 'hour', 'apt'] + WX
appw_ = (bw < 3600) & (np.abs(bw - ctx['vsd']) > 60); appr_ = (br < 3600) & (np.abs(br - msr) > 60)
fitw_ = appw_ & (y < 7200); rw_ = y - bw; U, Ur = appw_ & ~has, appr_ & ~hasr     # weather: fit on all rule rows, apply where ADS-B cannot see
dW = np.zeros(nw)
for f_, a_ in FOLDS(fitw_, U):
    dW[a_] = np.mean([lgb.train({**PRM, 'seed': s}, lgb.Dataset(Zw.loc[f_, CW], rw_[f_]), NR).predict(Zw.loc[a_, CW]) for s in SEEDS], 0)
dR = np.zeros(len(sub_ids))
dR[Ur] = np.mean([lgb.train({**PRM, 'seed': s}, lgb.Dataset(Zw.loc[fitw_, CW], rw_[fitw_]), NR).predict(Zr.loc[Ur, CW]) for s in SEEDS], 0)
nW = bw.copy(); nW[U] = cl(np.rint(bw[U] + dW[U])); nR = br.copy(); nR[Ur] = cl(np.rint(br[Ur] + dR[Ur]))
iw = r104['nW']
def price(name, new_w, new_r):
    lin = -2.0 * (iw - y) * (new_w - iw); own = (iw - y) ** 2 - (new_w - y) ** 2
    Qb = float(((new_r - v104r) ** 2).sum()); m2T = float(lin.sum()) * sc
    cs = np.bincount(clus, weights=lin); rng = np.random.default_rng(5); k = len(cs)
    SE = float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(2000)]).std()) * sc
    G = m2T - Qb
    print(f'\n{name}: window own {own.sum()*sc:+.4e} (covered {own[has].sum()*sc:+.3e}, uncovered {own[~has].sum()*sc:+.3e})  rows R {int((new_r!=v104r).sum()):,}\n'
          f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {G:+.4e}  SE {SE:.3e}  G/SE {G/SE:+.2f}  = {G/ONE:+.3f} bs  -> {np.sqrt(233.1847**2 - G/344841):.4f}')
    print('  per airport own: ' + '  '.join(f'{a} {own[apt == a].sum()*sc:+.2e}' for a in sorted(AC)))
    return G
price('refined stage alone (v104 -> stage part of v105)', bw, br)
G = price('v105 = refined stage + weather where ADS-B cannot see', nW, nR)
np.savez(f'{W}/adsblol/cand/v105_adsb.npz', ids=sub_ids, v104=v104r, new=nR, win_ids=ctx['ids'], incW=iw, newW=nW)
