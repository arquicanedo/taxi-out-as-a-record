"""The submitted chain's two LIRF seam rules, v95 and v96, as one step on any base and any scored months.

  uv run --with pandas --with numpy --with pyarrow python scripts/final/seam_rules.py --data DIR --base PARQUET --out PARQUET

Both rules are restated exactly from scripts/v95_build.py and scripts/v96_build.py; every quantity they use is computed
here from the labelled 2025 files, none from a leaderboard score.

  v95  LIRF departures with LOBT null, MVT - SCHED >= 7,200 s, at a stand whose number starts with 4 (stands 402-412,
       one pier and handling contract). 2025: 0 schedule copies in the cell (asserted, >= 25 rows). A row predicted above
       the cell's labelled median taxi + 60 s is moved to that median.
  v96  LIRF departures with MVT - SCHED > 600 s whose carrier (first three characters of FLIGHT_mvt) is one of the
       carriers whose labelled 2025 rows above 600 s are >= 95% schedule copies (>= 25 rows, >= 5 months): the handling
       agent does not feed the off-block system. P = their pooled copy rate, ALT = the median taxi of their non-copies.
       A row is committed to the copy clip(MVT - SCHED) when that lowers the expected squared error,
       P (inc - ms)^2 > (1 - P) ((ms - ALT)^2 - (inc - ALT)^2), and it is more than 60 s from the copy.
The output keeps the base's rows and order; only rule rows change.
"""
import argparse, glob, hashlib
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--base', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
THR, FLOOR, CAP = 7200., 60., 200000.
C = ['MVT_ID_mvt', 'ADEP_mvt', 'PHASE_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'LOBT_flt', 'STAND_mvt']

def prep(files):
    r = pd.concat([pq.read_table(f, columns=C).to_pandas() for f in files], ignore_index=True)
    for c in ('MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt'): r[c] = pd.to_datetime(r[c], errors='coerce', utc=True)
    d = r[r.PHASE_mvt.eq('DEP') & r.ADEP_mvt.eq('LIRF')].copy()
    d['ms'] = (d.MVT_TIME_UTC_mvt - d.SCHED_TIME_UTC_mvt).dt.total_seconds()
    d['sp'] = d.STAND_mvt.astype(str).str[:1]; d['car'] = d.FLIGHT_mvt.astype(str).str[:3]; d['mon'] = d.MVT_TIME_UTC_mvt.dt.month
    return d

L = prep(sorted(glob.glob(f'{a.data}/training_2025-*.parquet'))); L['y'] = L.TAXITIME_SEC_mvt.astype(float)
L['cp'] = np.abs(L.y - L.ms) <= 60
assert L.MVT_TIME_UTC_mvt.dt.year.eq(2025).all() and len(L) > 100000

# v95's labelled support
S5 = L[L.LOBT_flt.isna() & (L.ms >= THR) & L.sp.eq('4')]; T5 = float(np.median(S5.y))
assert len(S5) >= 25 and int(S5.cp.sum()) == 0, (len(S5), int(S5.cp.sum()))
print(f'v95 2025: {len(S5)} LIRF stand-4xx cell rows, {int(S5.cp.sum())} copies, {S5.mon.nunique()} months; target (median taxi) {T5:,.0f} s')
# v96's labelled support
A6 = L[L.ms > 600]; g = A6.groupby('car').agg(n=('cp', 'size'), cp=('cp', 'mean'), mons=('mon', 'nunique'))
HOT = sorted(g[(g.n >= 25) & (g.cp >= 0.95) & (g.mons >= 5)].index); H = A6[A6.car.isin(HOT)]
P = float(H.cp.mean()); ALT = float(H[~H.cp].y.median()); worst = min(float(x.cp.mean()) for _, x in H.groupby('mon'))
assert P >= 0.95 and worst >= 0.90 and float((H[~H.cp].y <= 2000).mean()) == 1.0
h1, h2 = H[H.mon <= 6].cp.mean(), H[H.mon >= 7].cp.mean()
print(f'v96 2025: carriers {HOT}; {len(H):,} rows, copy {P:.4f}, worst month {worst:.3f}, months 1-6 {h1:.4f} / 7-12 {h2:.4f}, ALT {ALT:,.0f} s')

tpl = pq.read_table(f'{a.data}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values
b = pq.read_table(a.base).to_pandas(); assert (b.MVT_ID_mvt.values == ids).all(), 'base must be in submitting order'
inc = b.TAXITIME_SEC_mvt.values.astype(float); p = inc.copy(); pos = pd.Series(np.arange(len(ids)), index=ids)
R = prep([f'{a.data}/ranking.parquet']); assert R.MVT_TIME_UTC_mvt.dt.year.eq(2026).all()
R['i'] = pos.reindex(R.MVT_ID_mvt).values; assert R.i.notna().all(); R['i'] = R.i.astype(int); R['inc'] = inc[R.i.values]

Z5 = R[R.LOBT_flt.isna() & (R.ms >= THR) & R.sp.eq('4')]; f5 = Z5[Z5.inc > T5 + 60]
p[f5.i.values] = T5
Z6 = R[(R.ms > 600) & R.car.isin(HOT)].copy()
Z6['pc'] = p[Z6.i.values]   # after v95, as in the chain
gain = (Z6.pc - Z6.ms) ** 2; loss = (Z6.ms - ALT) ** 2 - (Z6.pc - ALT) ** 2
f6 = Z6[(P * (-gain) + (1 - P) * loss < 0) & (np.abs(Z6.ms - Z6.pc) > 60)]
assert ((f6.ms >= FLOOR) & (f6.ms <= CAP)).all()
p[f6.i.values] = f6.ms.values

q = np.clip(np.rint(p), FLOOR, CAP); d = q - inc
for tag, f, Z in (('v95', f5, Z5), ('v96', f6, Z6)):
    dd = d[f.i.values]
    print(f'{tag}: cell rows by month {Z.mon.value_counts().sort_index().to_dict()}; fired {f.mon.value_counts().sort_index().to_dict()}; '
          f'Q {float(np.sum(dd ** 2)):.4e}; largest row share {float((dd ** 2).max() / max((dd ** 2).sum(), 1)):.1%}')
assert np.count_nonzero(d) <= len(f5) + len(f6)
o = pa.Table.from_pandas(pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': q.astype(tpl.schema.field('TAXITIME_SEC_mvt').type.to_pandas_dtype())}),
                         schema=tpl.schema, preserve_index=False)
pq.write_table(o, a.out)
print(f'wrote {a.out}  rows changed {int(np.count_nonzero(d)):,}  sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()}')
