"""The LIRF day-scale rule, built from labelled 2025 data and applied to any scored file.

POPULATION. Fiumicino (LIRF) departures with no flight-plan LOBT (`LOBT_flt` null) and
ms = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt >= 20,000 s.

WHAT THE LABEL IS A RECORD OF (v34, v91, dateerr_sweep.py, dayerr_decision.py). On these rows the
airport has no actual off-block, and BLOCK_TIME holds one of three things:
  copy      BLOCK = SCHED, so y = ms                                   (|y - ms| <= 60 s)
  rollover  BLOCK = the schedule's local DATE joined to the real off-block's local TIME of day,
            so y = 86400 * nd + taxi, nd = Rome local days from SCHED to MVT
                                                                       (nd >= 1, |y - (86400 nd + 1021)| <= 3600 s)
  ordinary  anything else (a real off-block)
A rollover needs the take-off to cross Rome's local date relative to the schedule (nd >= 1).

BANDS, all from labelled 2025 (training files in --data DIR):
  A  20,000 <= ms < 50,000            no date branch:  copy at its labelled rate, base otherwise
  B  ms >= 50,000 and nd >= 1         copy, rollover and base at their labelled rates
  C  ms >= 50,000 and nd == 0         no date branch:  copy at its labelled rate, base otherwise
The rollover target is the chain's: clip(86400 * nd + TAXI) with TAXI the median implied taxi
(y - 86400 nd) of the labelled band-B rollovers. FLOOR 60 and CAP 200,000 are the chain's constants;
the mixture is rounded and clipped as in v34.

It prints the 2025 table, the temporal-split gate (rates on months 1-6 against 7-12, a gap over 5pp
is flagged) and the support-range gate (each fired row against its band's labelled ms range), then
the fired rows per band per month, Q = sum (out - base)^2 per band, and every row the rule moves by
more than 600 s. The output keeps the base's rows, order, schema and every non-rule value.

  uv run --quiet --with pandas --with numpy --with pyarrow python scripts/final/lirf_dayscale.py \\
      --data DIR --base PARQUET --out PARQUET [--months 2,6] [--drop-char8]
"""
import argparse, glob, hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

FLOOR, CAP, DAY, TZ = 60.0, 200000.0, 86400.0, 'Europe/Rome'
LO, MID = 20000.0, 50000.0
TOL_COPY, TOL_ROLL, ROLL_CENTRE = 60.0, 3600.0, 1021.0      # dateerr_sweep.py's date-error label
GAP_PP = 5.0
BANDS = ('A', 'B', 'C')
BAND_NAME = {'A': 'A 20k<=ms<50k', 'B': 'B ms>=50k crossing', 'C': 'C ms>=50k not crossing'}
COLS = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'LOBT_flt', 'MVT_TIME_UTC_mvt',
        'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
# v126's CHAR(8) test: the airline designator's tail cycled to pad the flight number (the v54 cell)
char8 = lambda s: (s.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d')
                   | s.astype(str).str.match(r'^[A-Z]{2}([A-Z])\1')).values

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--base', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--months', default='')
ap.add_argument('--drop-char8', action='store_true', help='leave CHAR(8) identifiers out of the fired set')
a = ap.parse_args()
MONTHS = sorted({int(x) for x in a.months.split(',') if x.strip()})


def prep(df):
    """LIRF, LOBT-null departures with ms >= LO; adds ms, nd (Rome local days crossed), month, band."""
    x = df[df.PHASE_mvt.eq('DEP') & df.ADEP_mvt.eq('LIRF') & df.LOBT_flt.isna()].copy()
    m = pd.to_datetime(x.MVT_TIME_UTC_mvt, utc=True); s = pd.to_datetime(x.SCHED_TIME_UTC_mvt, utc=True)
    x['ms'] = (m - s).dt.total_seconds()
    x['nd'] = (m.dt.tz_convert(TZ).dt.normalize() - s.dt.tz_convert(TZ).dt.normalize()).dt.days
    x['mon'] = m.dt.month.values                       # UTC month of the take-off, as the files are cut
    x = x[x.ms >= LO].copy()
    x['band'] = np.where(x.ms < MID, 'A', np.where(x.nd >= 1, 'B', 'C'))
    x['c8'] = char8(x.FLIGHT_mvt.fillna(''))
    return x


# ---------------------------------------------------------------- 2025 labelled table
files = sorted(glob.glob(os.path.join(a.data, 'training_2025-*.parquet')))
assert len(files) == 12, f'expected 12 monthly training files in {a.data}, found {len(files)}'
T = pd.concat([pq.read_table(f, columns=COLS).to_pandas() for f in files], ignore_index=True)
assert not T.MVT_ID_mvt.duplicated().any(), 'duplicate MVT_ID in training'
L = prep(T)
yr = pd.to_datetime(L.MVT_TIME_UTC_mvt, utc=True).dt.year
assert yr.eq(2025).all() and L.TAXITIME_SEC_mvt.notna().all(), 'training rows must be labelled 2025 rows'
L['y'] = L.TAXITIME_SEC_mvt.astype(float)
cp = (L.y - L.ms).abs() <= TOL_COPY
rl = (L.nd >= 1) & ((L.y - (DAY * L.nd.clip(lower=1) + ROLL_CENTRE)).abs() <= TOL_ROLL)
L['cls'] = np.where(cp, 'copy', np.where(rl, 'roll', 'ord'))
print(f'2025 labelled: {len(T):,} rows in {len(files)} files; LIRF LOBT-null departures with ms >= {LO:,.0f}: {len(L)}'
      f'   (copy and rollover both true on {int((cp & rl).sum())} rows; copy wins)')
print(f'  copy: |y - ms| <= {TOL_COPY:.0f}   rollover: nd >= 1 and |y - (86400 nd + {ROLL_CENTRE:.0f})| <= {TOL_ROLL:.0f}'
      f'   ordinary: neither (max ordinary y {L.y[L.cls.eq("ord")].max():,.0f})')
print(f'\n  {"band":24s} {"n":>4s} {"copy":>5s} {"roll":>5s} {"ord":>5s}  {"ms min":>8s} {"ms max":>8s}  {"CHAR(8)":>7s}')
R = {}
for b in BANDS:
    s = L[L.band.eq(b)]; n = len(s); k = s.cls.value_counts()
    R[b] = dict(n=n, copy=int(k.get('copy', 0)), roll=int(k.get('roll', 0)), ord=int(k.get('ord', 0)),
                lo=float(s.ms.min()), hi=float(s.ms.max()))
    print(f'  {BAND_NAME[b]:24s} {n:4d} {R[b]["copy"]:5d} {R[b]["roll"]:5d} {R[b]["ord"]:5d}  '
          f'{R[b]["lo"]:8,.0f} {R[b]["hi"]:8,.0f}  {int(s.c8.sum()):7d}')
assert all(R[b]['n'] > 0 for b in BANDS), 'an empty band'
assert R['A']['roll'] == 0 and R['C']['roll'] == 0, 'a rollover outside band B: the no-date-branch premise failed'
assert R['B']['roll'] > 0

print('\n  the chain\'s other cuts (v126: 30,000 / 46,800 / 86,000; v91: crossing)')
sub = pd.cut(L.ms, [LO, 30000, 46800, MID, 86000, 1e9], right=False,
             labels=['20.0-30.0k', '30.0-46.8k', '46.8-50.0k', '50.0-86.0k', '>=86.0k'])
xt = pd.crosstab([sub.rename('ms'), L.nd.ge(1).map({True: 'cross', False: 'same-day'}).rename('date')],
                 L.cls).reindex(columns=['copy', 'roll', 'ord'], fill_value=0)
print('    ' + xt.to_string().replace('\n', '\n    '))
xc = pd.crosstab([L.band.rename('band'), pd.Series(L.c8, index=L.index).map({True: 'CHAR8', False: 'clean'}).rename('id')],
                 L.cls).reindex(columns=['copy', 'roll', 'ord'], fill_value=0)
print('    ' + xc.to_string().replace('\n', '\n    '))
rr = L[L.cls.eq('roll')]
TAXI = float(np.median(rr.y - DAY * rr.nd))
print(f'\n  rollover implied taxi y - 86400 nd: n {len(rr)}, nd {dict(rr.nd.value_counts())}, '
      f'min {(rr.y - DAY * rr.nd).min():,.0f}, median {TAXI:,.0f}, max {(rr.y - DAY * rr.nd).max():,.0f}')

print(f'\nGATE 1, temporal split (UTC months 1-6 against 7-12; a gap over {GAP_PP:.0f}pp is flagged)')
print(f'  {"band":24s} {"rate":5s} {"H1":>12s} {"H2":>12s} {"gap pp":>7s}')
for b in BANDS:
    s = L[L.band.eq(b)]; h1 = s[s.mon <= 6]; h2 = s[s.mon >= 7]
    for c in ('copy', 'roll', 'ord'):
        if R[b][c] == 0 and c != 'copy': continue
        r1 = h1.cls.eq(c).mean() if len(h1) else np.nan; r2 = h2.cls.eq(c).mean() if len(h2) else np.nan
        g = 100 * abs(r1 - r2) if len(h1) and len(h2) else np.nan
        f = 'FLAG' if not np.isfinite(g) or g > GAP_PP else 'ok'
        print(f'  {BAND_NAME[b]:24s} {c:5s} {int(h1.cls.eq(c).sum()):3d}/{len(h1):<3d}{100*r1:5.1f}% '
              f'{int(h2.cls.eq(c).sum()):3d}/{len(h2):<3d}{100*r2:5.1f}% {g:7.1f}  {f}')

W = {b: dict(copy=R[b]['copy'] / R[b]['n'], roll=R[b]['roll'] / R[b]['n'], ord=R[b]['ord'] / R[b]['n']) for b in BANDS}
# Band C rests on a handful of labelled rows. Not crossing the local date rules a date error out, as in band A, so C takes
# the pooled rate of every non-crossing row (A + C) rather than its own.
_nA = R['A']['n'] + R['C']['n']
W['C'] = dict(copy=(R['A']['copy'] + R['C']['copy']) / _nA, roll=0.0, ord=(R['A']['ord'] + R['C']['ord']) / _nA)
print(f"  band C takes the pooled non-crossing rate: copy {R['A']['copy'] + R['C']['copy']}/{_nA} = {W['C']['copy']:.4f}")
print('\nTHE RULE (weights = labelled 2025 rates)')
for b in BANDS:
    print(f'  {BAND_NAME[b]:24s} copy {R[b]["copy"]}/{R[b]["n"]} = {W[b]["copy"]:.4f}   '
          f'roll {R[b]["roll"]}/{R[b]["n"]} = {W[b]["roll"]:.4f}   base {R[b]["ord"]}/{R[b]["n"]} = {W[b]["ord"]:.4f}')
print(f'  copy target clip(ms, {FLOOR:.0f}, {CAP:.0f});  roll target clip(86400 nd + {TAXI:,.0f}, {FLOOR:.0f}, {CAP:.0f});  '
      f'prediction clip(rint(mixture), {FLOOR:.0f}, {CAP:.0f})')

# ---------------------------------------------------------------- scored file
rk = pq.read_table(os.path.join(a.data, 'ranking.parquet'), columns=COLS).to_pandas()
dep = rk[rk.PHASE_mvt.eq('DEP')]
assert not dep.MVT_ID_mvt.duplicated().any(), 'duplicate departure MVT_ID in ranking'
S = prep(rk)
bt = pq.read_table(a.base)
assert bt.schema.field('MVT_ID_mvt').type == pa.float64(), 'MVT_ID_mvt must be float64'
ids = bt.column('MVT_ID_mvt').to_numpy()
assert not pd.Series(ids).duplicated().any(), 'duplicate MVT_ID in base'
yt = bt.schema.field('TAXITIME_SEC_mvt').type
base = bt.column('TAXITIME_SEC_mvt').to_numpy(zero_copy_only=False).astype(float)
print(f'\nscored: {a.data}/ranking.parquet  departures {len(dep):,}, rule population {len(S)} '
      f'(by UTC month {dict(S.mon.value_counts().sort_index())})')
print(f'base:   {a.base}  rows {len(ids):,}, TAXITIME type {yt}, ids also in ranking departures '
      f'{int(np.isin(ids, dep.MVT_ID_mvt.values).sum()):,}')
pos = pd.Index(ids).get_indexer(S.MVT_ID_mvt.values)
if (pos < 0).any(): print(f'  rule rows absent from the base, left alone: {int((pos < 0).sum())}')
S = S[pos >= 0].copy(); S['pos'] = pos[pos >= 0]
if MONTHS:
    print(f'  --months {MONTHS}: rule rows outside them left alone: {int((~S.mon.isin(MONTHS)).sum())}')
    S = S[S.mon.isin(MONTHS)]
if a.drop_char8:
    print(f'  --drop-char8: CHAR(8) rule rows left alone: {int(S.c8.sum())}')
    S = S[~S.c8]
S['base'] = base[S.pos.values]
assert np.isfinite(S.base).all(), 'base has no prediction on a rule row'

copy_t = np.clip(S.ms.values, FLOOR, CAP)
roll_t = np.clip(DAY * S.nd.values + TAXI, FLOOR, CAP)
w = np.array([[W[b]['copy'], W[b]['roll'], W[b]['ord']] for b in S.band.values]).reshape(-1, 3)
S['out'] = np.clip(np.rint(w[:, 0] * copy_t + w[:, 1] * roll_t + w[:, 2] * S.base.values), FLOOR, CAP)
S['d'] = S.out - S.base

out = base.copy(); out[S.pos.values] = S.out.values
assert np.array_equal(out[np.setdiff1d(np.arange(len(out)), S.pos.values)],
                      base[np.setdiff1d(np.arange(len(base)), S.pos.values)]), 'a non-rule row changed'

print(f'\nGATE 2, support range (each fired row against its band\'s 2025 labelled ms range)')
for b in BANDS:
    s = S[S.band.eq(b)]; o = s[(s.ms < R[b]['lo']) | (s.ms > R[b]['hi'])]
    print(f'  {BAND_NAME[b]:24s} 2025 ms {R[b]["lo"]:8,.0f} .. {R[b]["hi"]:8,.0f}   fired {len(s):3d}   outside {len(o):3d}'
          + (f'   FLAG ms {sorted(o.ms.round().astype(int).tolist())}' if len(o) else ''))

print('\nFIRED ROWS per band per UTC month, and Q = sum (out - base)^2')
mons = sorted(S.mon.unique().tolist())
print(f'  {"band":24s} ' + ' '.join(f'{"m" + str(m):>5s}' for m in mons) + f' {"total":>6s} {"moved":>6s} {"CHAR8":>6s} {"Q":>11s}')
for b in BANDS:
    s = S[S.band.eq(b)]
    print(f'  {BAND_NAME[b]:24s} ' + ' '.join(f'{int(s.mon.eq(m).sum()):5d}' for m in mons)
          + f' {len(s):6d} {int((s.d != 0).sum()):6d} {int(s.c8.sum()):6d} {float((s.d ** 2).sum()):11.4e}')
print(f'  {"all":24s} ' + ' '.join(f'{int(S.mon.eq(m).sum()):5d}' for m in mons)
      + f' {len(S):6d} {int((S.d != 0).sum()):6d} {int(S.c8.sum()):6d} {float((S.d ** 2).sum()):11.4e}')

big = S[S.d.abs() > 600].sort_values('d', key=np.abs, ascending=False)
print(f'\nROWS WHERE THE RULE DISAGREES WITH THE BASE BY MORE THAN 600 s: {len(big)}'
      f'   (their Q {float((big.d ** 2).sum()):.4e}, {100 * float((big.d ** 2).sum()) / max(float((S.d ** 2).sum()), 1):.1f}% of the total)')
if len(big):
    print(f'  {"MVT_ID":>11s} {"flight":10s} {"m":>2s} {"band":4s} {"ms":>8s} {"nd":>2s} {"c8":>2s} {"base":>8s} {"out":>8s} {"out-base":>9s}')
    for r in big.itertuples():
        print(f'  {r.MVT_ID_mvt:11.0f} {str(r.FLIGHT_mvt):10s} {r.mon:2d} {r.band:4s} {r.ms:8,.0f} {r.nd:2d} {int(r.c8):2d} '
              f'{r.base:8,.0f} {r.out:8,.0f} {r.d:+9,.0f}')

# ---------------------------------------------------------------- write, preserving the base exactly
assert np.isfinite(out).all() or not np.isfinite(base).all()
arr = pa.array(np.rint(out).astype(yt.to_pandas_dtype()) if pa.types.is_integer(yt) else out, type=yt)
ot = bt.set_column(bt.schema.get_field_index('TAXITIME_SEC_mvt'), bt.schema.field('TAXITIME_SEC_mvt'), arr)
assert ot.schema.equals(bt.schema, check_metadata=True)
pq.write_table(ot, a.out)
chk = pq.read_table(a.out)
assert chk.num_rows == bt.num_rows, 'row count changed'
assert chk.schema.field('MVT_ID_mvt').type == pa.float64()
assert np.array_equal(chk.column('MVT_ID_mvt').to_numpy(), ids), 'ID order changed'
print(f'\nwrote {a.out}  rows {chk.num_rows:,} (= base), ID order identical, '
      f'sha256 {hashlib.sha256(open(a.out, "rb").read()).hexdigest()}')
