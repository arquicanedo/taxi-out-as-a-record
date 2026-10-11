"""v104 = v103's stage plus the INBOUND-PARKING features (s520), priced as an INCREMENT over the shipped v103.
(v103's description follows; v104 changes only the feature table and the incumbent.)
Changes from v102's stage (scripts/adsb_price.py), each measured in scripts/adsb_stage_exp.py:
  - coverage: departures the callsign missed are matched by take-off time alone (|dt| <= 60 s, unique both ways;
    99.78% precise: 10,499 of 10,522 blind matches on 17,474 known callsign pairs, s518), ab_tm = 1
  - trained on callsign-matched rows only (time-matched rows in training cost the callsign rows 2.6e8), applied to all
  - applied to copy-fired rows too, and NOT to hedged rows (v100 >= 3600: s518's D^2 check, 5.9% ordinary by mass)
  - three detection features: observed taxi path, push-to-taxi-speed time, longest stop in the last 15 min
Feature table: artifacts/adsb_push_feat_v104.parquet (adsb_push_features.py with ADSB_TIME_MATCH=1 ADSB_PARK=1; ODbL).
Incumbent: v102 on the window (adsb_price.py's saved newW) and on the scored set (the shipped parquet).
  ADSB_FEAT=artifacts/adsb_push_feat_v104.parquet python scripts/adsb_price_v104.py   ->  cand/v104_adsb.npz (needs v103_adsb.npz with newW)
"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('ADSB_FEAT', os.path.join(os.path.dirname(_HERE), 'artifacts/adsb_push_feat_v104.parquet'))
__file__ = os.path.join(_HERE, 'adsb_stage_exp.py'); sys.argv = ['x', 'none']
exec(open(__file__).read().split("# ---------------- neighbour")[0])
assert 'tm' in X.columns and EXTRA == ['ab_path', 'ab_tmove', 'ab_hold', 'ab_pk_dist', 'ab_pk_age', 'ab_pk_gs', 'ab_taxi_pk'], 'v104 needs the parking table'
tmw_ = X.tm.values == 1
COLS = [c for c in X.columns if c != 'rwy']
res = run('v104 stage', cols=COLS, app_w=has & (v100w < 3600), app_r=hasr & (v100r < 3600), fit_w=has & ~tmw_ & (y < 7200))
z = np.load(f'{I.CAND}/v103_adsb.npz')
assert (z['win_ids'] == ctx['ids']).all() and (z['ids'] == sub_ids).all()
v102w, v102r = z['newW'].astype(float), I.sub('v103')   # names kept: v102* is the INCUMBENT, here v103
assert np.array_equal(z['new'].astype(float), v102r), 'v103 npz != shipped v103'
v103w, v103r = res['nW'], res['nR']
lin = -2.0 * (v102w - y) * (v103w - v102w); own = (v102w - y) ** 2 - (v103w - y) ** 2
Qb = float(((v103r - v102r) ** 2).sum()); m2T = float(lin.sum()) * sc
cs = np.bincount(clus, weights=lin); k = len(cs); rng = np.random.default_rng(5)
SE = float(np.array([cs[rng.integers(0, k, k)].sum() for _ in range(2000)]).std()) * sc
print(f'\nINCREMENT v104 - v103 (window: Jan|Jul cross-fitted both; scored: exact)\n'
      f'  window own dSSE {own.sum()*sc:+.4e} ({own.sum()*sc/ONE:+.3f} bs)   rows changed W {int((v103w!=v102w).sum()):,} R {int((v103r!=v102r).sum()):,}\n'
      f'  -2T_w*sc {m2T:+.4e}  Q_b {Qb:.4e}  G_hat {m2T-Qb:+.4e}  SE {SE:.3e}  G/SE {(m2T-Qb)/SE:+.2f}  = {(m2T-Qb)/ONE:+.3f} board-s  (Q-anchored: a LOWER bound, s517)')
tot = 0.0
for a in AC:
    cw, cr = (v103w != v102w) & (apt == a), (v103r != v102r) & (aptr == a)
    if not cw.any(): continue
    qa = float(((v103r - v102r) ** 2)[aptr == a].sum()); g = float(lin[cw].sum()) * sc / cw.sum() * cr.sum() - qa; tot += g
    print(f'    {a}  changed W {cw.sum():>6,} R {cr.sum():>6,}   lin {float(lin[cw].sum())*sc:+.3e}  Q_b {qa:.3e}   coverage-scaled {g:+.3e}')
print(f'  coverage-scaled increment {tot:+.4e} = {tot/ONE:+.3f} board-s  (estimator; v102 paid 1.44x this estimator)')
N = 344841; s102 = 234.7244   # v103's board score
for lab, G in (('Q-anchored G_hat', m2T - Qb), ('coverage-scaled', tot), ('coverage-scaled x2.50 (v103 ratio)', 2.50 * tot)):
    print(f'  {lab:<36} {G:+.4e}  ->  {np.sqrt(s102**2 - G/N):.4f}')
np.savez(f'{I.CAND}/v104_adsb.npz', ids=sub_ids, v103=v102r, v100=v100r, new=v103r, win_ids=ctx['ids'], newW=v103w)
