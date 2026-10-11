"""Per-departure ADS-B pushback features from the adsb_stream_day.py outputs, for every parsed day.
Match: ADEP + callsign (CALLSIGN_flt, normalised) + ADS-B take-off within 300 s of MVT_TIME_UTC_mvt. Pushback, as in
p3q_adsb_push.py (s512): after the last >900 s gap in the (hex, airport) ground track within 3 h before take-off,
if the aircraft is parked when first seen, the first movement after that dwell ('dwell'), else the first sighting
('first-seen', transponder switched on during the push). Ground points of day D-1 are included for early departures, except on the 1st of the month (D-1 is not pulled).
  uv run --with pandas --with pyarrow --with numpy python scripts/adsb_push_features.py
Output: $DC26_WORK/adsblol/push_feat.parquet, one row per departure on a parsed day (matched or not).
"""
import os, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
from multiprocessing import Pool
A = os.path.join(os.environ['DC26_WORK'], 'adsblol'); DY = os.environ.get('ADSB_DAYS', f'{A}/days')
D = os.environ['DC26_DATA']
APT = {'EHAM': (52.3105, 4.7683), 'EDDF': (50.0379, 8.5622), 'EGLL': (51.4700, -0.4543), 'LIRF': (41.8003, 12.2389), 'LFPG': (49.0097, 2.5479),
       'EDDM': (48.3538, 11.7861), 'LSZH': (47.4582, 8.5555), 'LEMD': (40.4719, -3.5626), 'LEBL': (41.2974, 2.0833), 'LTFM': (41.2753, 28.7519)}
COLS = ['MVT_ID_mvt', 'ADEP_mvt', 'CALLSIGN_flt', 'MVT_TIME_UTC_mvt', 'PHASE_mvt']
norm = lambda s: s.fillna('').astype(str).str.replace(' ', '').str.upper()
E = pd.Timestamp('1970-01-01', tz='UTC'); sec = lambda x: (pd.to_datetime(x, utc=True) - E).dt.total_seconds().values

ALL_MONTHS = os.environ.get('ADSB_ALL_MONTHS', '0') == '1'   # s548: the full-year table; default = the 124-day chain of v102-v109
# The scored months of 2026 (final phase, 2026-10-08: ADSB_SCORED_MONTHS=01,02,06,07). 2025 stays the Jan+Jul window, or
# every month under ADSB_ALL_MONTHS. With the default this selects exactly the days and D-1 inclusions it always did.
SCORED = tuple(os.environ.get('ADSB_SCORED_MONTHS', '01,07').split(','))
def allowed(d): return (d[:4] == '2025' and (d[4:6] in ('01', '07') or ALL_MONTHS)) or (d[:4] == '2026' and d[4:6] in SCORED)
def departures():
    fs = ([f'{D}/training_2025-01-01_2025-02-01.parquet', f'{D}/training_2025-07-01_2025-08-01.parquet'] if not ALL_MONTHS
          else sorted(glob.glob(f'{D}/training_2025-*.parquet'))) + [f'{D}/ranking.parquet']   # s548: all twelve 2025 months
    d = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in fs])
    d = d[d.PHASE_mvt.eq('DEP')].copy(); d['t_mvt'] = sec(d.MVT_TIME_UTC_mvt); d['cs'] = norm(d.CALLSIGN_flt)
    d['day'] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y%m%d')
    return d[['MVT_ID_mvt', 'ADEP_mvt', 'cs', 't_mvt', 'day']]

def features(g, t_to, t_mvt):
    g = g[(g.t <= t_to) & (g.t >= t_to - 3 * 3600)]
    if len(g) < 3: return None
    t = g.t.values; gap = np.where(np.diff(t) > 900)[0]
    if len(gap): g = g.iloc[gap[-1] + 1:]; t = g.t.values
    if len(g) < 3: return None
    la0, lo0 = APT[g.apt.iat[0]]
    lat = g.lat.values; lon = g.lon.values; gs = np.nan_to_num(g.gs.values, nan=0.0)
    dx = (lon - lon[0]) * 111320 * np.cos(np.radians(lat[0])); dy = (lat - lat[0]) * 110570; disp = np.hypot(dx, dy)
    arp = np.hypot((lon - lo0) * 111.32 * np.cos(np.radians(la0)), (lat - la0) * 110.57)
    if gs[0] <= 1.0 and disp[min(3, len(disp) - 1)] < 10:
        mv = np.where((disp > 15) | (gs > 2.0))[0]
        if not len(mv): return None
        i = mv[0]; how = 1
    else: i = 0; how = 0
    tp = t[i:]; dt = np.diff(tp); still = (gs[i:-1] <= 1.0)
    seg = np.hypot(np.diff(dx[i:]), np.diff(dy[i:])); fast = np.where(gs[i:] > 5.0)[0]
    late = (tp[:-1] >= t_to - 900) & (gs[i:-1] <= 3.0); hold = 0.0; run = 0.0   # s518: longest stop in the last 15 min
    for k in range(len(dt)):
        run = run + dt[k] if late[k] else 0.0; hold = max(hold, run)
    ext = dict(ab_path=float(seg.sum()), ab_tmove=float(tp[fast[0]] - tp[0]) if len(fast) else np.nan, ab_hold=hold)
    return dict(**ext, ab_push=t[i], ab_taxi=t_mvt - t[i], ab_how=how, ab_dwell=t[i] - t[0], ab_d0=arp[0], ab_dpush=arp[i],
                ab_n=len(tp), ab_gap=float(dt.max()) if len(dt) else np.nan, ab_stop=float(dt[still].sum()) if len(dt) else 0.0,
                ab_gs0=gs[0], ab_dto=t_to - t_mvt)

TIME_MATCH = os.environ.get('ADSB_TIME_MATCH', '0') == '1'; TM_WIN = 60.0   # 0 = v102's callsign-only chain
def time_match(dep, to, used_dep, used_to):
    """Pairs (departure, ADS-B take-off) at the same airport with |t_to - t_mvt| <= TM_WIN, neither already matched by
    callsign, and each the other's ONLY candidate. Returns rows shaped like the callsign match, with tm = 1."""
    d = dep[~dep['index'].isin(used_dep)]; t = to[[(h, x) not in used_to for h, x in zip(to.hex, to.t_to)]]
    c = d.merge(t, left_on='ADEP_mvt', right_on='apt', suffixes=('', '_to'))
    c = c[(c.t_to - c.t_mvt).abs() <= TM_WIN]
    c = c[c.groupby('index').index.transform('size').eq(1)]
    c = c[c.groupby(['hex', 't_to']).hex.transform('size').eq(1)]
    return c.assign(tm=1)
TM_RELAX = float(os.environ.get('ADSB_TM_RELAX', '0'))   # > 0: s518b relaxed rule, margin in seconds; 0 = v103's strict rule
def time_match_relaxed(dep, to, used_dep, used_to, margin):
    """Like time_match, but a departure keeps its NEAREST take-off when the runner-up is >= margin s further, and the
    take-off must choose that departure back by the same rule (mutual nearest with a margin)."""
    d = dep[~dep['index'].isin(used_dep)]; t = to[[(h, x) not in used_to for h, x in zip(to.hex, to.t_to)]]
    c = d.merge(t, left_on='ADEP_mvt', right_on='apt', suffixes=('', '_to'))
    c = c[(c.t_to - c.t_mvt).abs() <= TM_WIN].copy(); c['adt'] = (c.t_to - c.t_mvt).abs()
    c['tk'] = c.hex.astype(str) + '|' + c.t_to.astype(str)
    def winners(key):
        o = c.sort_values([key, 'adt'], kind='stable'); r = o.groupby(key).cumcount()
        first = o[r.values == 0]; second = o[r.values == 1].set_index(key).adt
        gap = first[key].map(second).values - first.adt.values
        return first[np.isnan(gap) | (gap >= margin)]
    k = winners('index').merge(winners('tk')[['index', 'tk']], on=['index', 'tk'])
    return k.drop(columns=['tk']).assign(tm=1)
PARK = os.environ.get('ADSB_PARK', '0') == '1'; XY = os.environ.get('ADSB_XY', '0') == '1'   # s520: where the aircraft was last seen parked before its departure track
def parking(g, f, t_mvt):
    """The departure segment starts at t0 = ab_push - ab_dwell. The last ground sighting of the same airframe at the same
    airport in the 18 h before t0 (its inbound arrival or an earlier tow) is where it was last seen parked. Returns the
    distance from there to the first departure sighting, the time between them, and a taxi corrected for the unseen
    part of a late first sighting (first-seen rows only: distance / 5 m/s added back)."""
    t0 = f['ab_push'] - f['ab_dwell']
    pre = g[(g.t < t0 - 1) & (g.t >= t0 - 18 * 3600)]; cur = g[g.t >= t0 - 1]
    if not len(pre) or not len(cur): return dict(ab_pk_dist=np.nan, ab_pk_age=np.nan, ab_pk_gs=np.nan, ab_taxi_pk=f['ab_taxi'],
                                                 **(dict(ab_px=np.nan, ab_py=np.nan, ab_qx=np.nan, ab_qy=np.nan) if XY else {}))
    p, q = pre.iloc[-1], cur.iloc[0]
    dist = float(np.hypot((q.lon - p.lon) * 111320 * np.cos(np.radians(p.lat)), (q.lat - p.lat) * 110570))
    tx = f['ab_taxi'] + (dist / 5.0 if f['ab_how'] == 0 else 0.0)
    out = dict(ab_pk_dist=dist, ab_pk_age=float(q.t - p.t), ab_pk_gs=float(np.nan_to_num(p.gs)), ab_taxi_pk=tx)
    if XY:   # s523: WHERE, not just how far: parked and first-sighting positions in km east/north of the ARP
        la0, lo0 = APT[p.apt]; kx = 111.32 * np.cos(np.radians(la0))
        out.update(ab_px=(p.lon - lo0) * kx, ab_py=(p.lat - la0) * 110.57, ab_qx=(q.lon - lo0) * kx, ab_qy=(q.lat - la0) * 110.57)
    return out
def one_day(day):
    prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime('%Y%m%d')
    if not allowed(prev) and not ALL_MONTHS: prev = day   # fixed day set: the pulled days only (full-year mode: any parsed day)
    gp = pd.concat([pd.read_parquet(f'{DY}/d{k}_ground.parquet') for k in sorted({prev, day}) if os.path.exists(f'{DY}/d{k}.done')])
    to = pd.concat([pd.read_parquet(f'{DY}/d{k}_takeoffs.parquet') for k in sorted({prev, day}) if os.path.exists(f'{DY}/d{k}.done')])
    dep = DEP[DEP.day.eq(day)]; to = to.assign(cs=norm(to.callsign))
    m = dep.reset_index(drop=True).reset_index().merge(to, left_on=['ADEP_mvt', 'cs'], right_on=['apt', 'cs'], how='inner')
    m = m[(m.cs != '') & ((m.t_to - m.t_mvt).abs() <= 300)].copy(); m['adt'] = (m.t_to - m.t_mvt).abs()
    m = m.sort_values(['adt', 'hex', 't_to'], kind='stable').drop_duplicates('index')
    G = {k: v for k, v in gp.sort_values(['hex', 't', 'lat', 'lon'], kind='stable').groupby(['hex', 'apt'])}
    m['tm'] = 0
    if TIME_MATCH:   # s518: departures the callsign missed, matched by take-off time alone, unique both ways
        tmf = (lambda *x: time_match_relaxed(*x, TM_RELAX)) if TM_RELAX > 0 else time_match
        m = pd.concat([m, tmf(dep.reset_index(drop=True).reset_index(), to, set(m['index']), set(zip(m.hex, m.t_to)))], ignore_index=True)
    rows = []
    for r in m.itertuples():
        g = G.get((r.hex, r.apt)); f = features(g, r.t_to, r.t_mvt) if g is not None else None
        if f and PARK: f.update(parking(g, f, r.t_mvt))
        rows.append(dict(MVT_ID_mvt=r.MVT_ID_mvt, ab_hex=r.hex, ab_match=1, ab_tm=r.tm, **(f or {})))
    out = dep[['MVT_ID_mvt', 'ADEP_mvt', 'day']].merge(pd.DataFrame(rows, columns=['MVT_ID_mvt', 'ab_hex', 'ab_match']) if not rows else pd.DataFrame(rows), on='MVT_ID_mvt', how='left')
    out['ab_match'] = out.ab_match.fillna(0).astype(int)
    return out

DEP = departures()
if __name__ == '__main__':
    days = sorted(os.path.basename(p)[1:9] for p in glob.glob(f'{DY}/d*.done'))
    days = [d for d in days if allowed(d)]
    with Pool(8) as pool: res = pool.map(one_day, days)
    f = pd.concat(res, ignore_index=True); f.to_parquet(f'{A}/push_feat' + ('_tm' if TIME_MATCH else '') + (f'_r{int(TM_RELAX)}' if TM_RELAX > 0 else '') + ('_pk' if PARK else '') + ('_xy' if XY else '') + ('_all' if ALL_MONTHS else '') + '.parquet', index=False)
    f['has'] = f.ab_push.notna()
    print(f'{len(days)} days, {len(f):,} departures, matched {f.ab_match.mean():.1%}, with pushback {f.has.mean():.1%}')
    print(f.assign(yr=f.day.str[:4]).groupby(['yr', 'ADEP_mvt']).agg(n=('has', 'size'), matched=('ab_match', 'mean'), push=('has', 'mean'),
          dwell=('ab_how', 'mean')).round(3).unstack(0).to_string())
