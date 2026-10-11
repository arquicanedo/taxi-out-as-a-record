"""Figure: the recording failure follows the aircraft, and is far too small to act on (section 2.2).

Reproduces `scripts/fallback_follows_airframe.py`. Chain each departure back to its own previous
departure at a DIFFERENT airport, via the stand-occupancy arrival that precedes it (type-agreement
validated) and that arrival's `FLIGHT_ID_mvt`, which pairs it to its inbound leg. Then compare
P(schedule copy | the previous leg was a copy) with P(schedule copy | it was not), within airport,
so the station effect is held fixed.

Controls printed against the paper:
  277,672 chained departures, overall copy rate 9.287%
  previous leg WAS a copy 25,657 rows at 11.763%; was NOT 252,015 at 9.035%; lift 1.30x
  within airport EDDF 1.19  EDDM 1.22  EGLL 1.24  EHAM 1.33  LEBL 1.47
                 LEMD 1.32  LFPG 1.25  LIRF 1.28  LSZH 1.57  LTFM 1.22
"""
import os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

D = os.environ['DC26_DATA']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Charter', 'Georgia', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#4a4a4a', 'axes.linewidth': 0.6,
    'xtick.color': '#000000', 'ytick.color': '#000000',
    'axes.labelcolor': '#000000', 'text.color': '#000000',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})
from fig_palette import INK, REF, REF_LIGHT, MEASURED, RECORD, RECORD_TINT, CONTEXT
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')
COMMIT = 0.50      # a false commitment costs (86,400 - taxi)^2 against a gain of the same order


def secs(s):
    return (pd.to_datetime(s, errors='coerce', utc=True) - EPOCH).dt.total_seconds()


C = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'MVT_TIME_UTC_mvt',
     'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'AIRCRAFT_TYPE_mvt', 'STAND_mvt', 'FLIGHT_mvt']
R = pd.concat([pq.read_table(f, columns=C).to_pandas()
               for f in sorted(glob.glob(f'{D}/training_*.parquet'))], ignore_index=True)
R['mvt'] = secs(R.MVT_TIME_UTC_mvt); R['blk'] = secs(R.BLOCK_TIME_UTC_mvt); R['sch'] = secs(R.SCHED_TIME_UTC_mvt)
assert 1.735e9 < float(R.mvt.median()) < 1.80e9, f'timestamp units {R.mvt.median():.4g}'
R['apt'] = np.where(R.PHASE_mvt.eq('DEP'), R.ADEP_mvt.astype(str), R.ADES_mvt.astype(str))
R['ty'] = R.AIRCRAFT_TYPE_mvt.astype(str)
R['key'] = R.apt + '|' + R.STAND_mvt.astype(str)
R['cp'] = (R.blk - R.sch).abs() <= 60

ARR = R[R.PHASE_mvt.eq('ARR') & R.blk.notna()]
DEP = R[R.PHASE_mvt.eq('DEP') & R.mvt.notna()]
A = ARR[['key', 'blk', 'ty', 'FLIGHT_ID_mvt']].sort_values('blk').reset_index(drop=True)
Dq = DEP[['MVT_ID_mvt', 'key', 'mvt', 'ty', 'apt', 'cp']].sort_values('mvt').reset_index(drop=True)
m = pd.merge_asof(Dq, A.rename(columns={'blk': 'ain', 'ty': 'aty', 'FLIGHT_ID_mvt': 'afid'}),
                  left_on='mvt', right_on='ain', by='key', direction='backward',
                  allow_exact_matches=False)
m = m[(m.aty == m.ty) & ((m.mvt - m.ain) <= 86400)]
# the window is a turnaround: print it and assert it is physically possible
q = np.quantile((m.mvt - m.ain).values, [.05, .50, .95]) / 3600
print(f'turnaround quantiles (h): 5% {q[0]:.2f}  50% {q[1]:.2f}  95% {q[2]:.2f}')
assert 0.2 < q[1] < 8.0, 'median turnaround is not physically plausible'
# pandas matches NaN to NaN on a merge key, so a null FLIGHT_ID would join every null to every null
m = m[m.afid.notna()]
prevdep = DEP[DEP.FLIGHT_ID_mvt.notna()][['FLIGHT_ID_mvt', 'cp', 'apt']].rename(
    columns={'FLIGHT_ID_mvt': 'afid', 'cp': 'prev_cp', 'apt': 'prev_apt'}).drop_duplicates('afid')
n0 = len(m)
m = m.merge(prevdep, on='afid', how='inner')
assert len(m) <= n0, f'join grew {n0:,} -> {len(m):,}'
m = m[m.apt != m.prev_apt]                       # a DIFFERENT airport: the station is held apart

r1 = float(m.cp[m.prev_cp].mean()); r0 = float(m.cp[~m.prev_cp].mean())
print(f'CONTROL chained departures {len(m):,}, overall copy rate {m.cp.mean():.3%}   '
      f'paper 277,672 / 9.287%')
print(f'CONTROL previous leg WAS a copy {int(m.prev_cp.sum()):,} at {r1:.3%}, '
      f'was NOT {int((~m.prev_cp).sum()):,} at {r0:.3%}, lift {r1/r0:.2f}x   '
      f'paper 25,657 / 11.763% / 252,015 / 9.035% / 1.30x')
assert abs(len(m) - 277672) < 200 and abs(r1 / r0 - 1.30) < 0.02, 'disagrees with the paper'

rows = []
for a, g in m.groupby('apt'):
    n1, n0_ = int(g.prev_cp.sum()), int((~g.prev_cp).sum())
    p1, p0 = float(g.cp[g.prev_cp].mean()), float(g.cp[~g.prev_cp].mean())
    se1 = np.sqrt(p1 * (1 - p1) / n1); se0 = np.sqrt(p0 * (1 - p0) / n0_)
    # delta method on log lift
    sl = np.sqrt((1 - p1) / (p1 * n1) + (1 - p0) / (p0 * n0_))
    rows.append((a, len(g), p1, p0, se1, se0, p1 / p0, np.exp(np.log(p1 / p0) - 1.96 * sl),
                 np.exp(np.log(p1 / p0) + 1.96 * sl), COMMIT / p0))
T = pd.DataFrame(rows, columns=['apt', 'n', 'p1', 'p0', 'se1', 'se0', 'lift', 'lo', 'hi', 'need'])
T = T.sort_values('lift').reset_index(drop=True)
print(f'  {"apt":<6} {"n":>8} {"P(copy|prev copy)":>18} {"P(copy|not)":>12} {"lift":>7} '
      f'{"95% CI":>14} {"lift needed":>12}')
for _, r in T.iterrows():
    print(f'  {r.apt:<6} {r.n:>8,.0f} {r.p1:>17.3%} {r.p0:>11.3%} {r.lift:>6.2f}x '
          f'{r.lo:>6.2f}-{r["hi"]:<6.2f} {r.need:>11.1f}x')
print(f'CONTROL positive at {int((T.lift > 1).sum())} airports of {len(T)}   paper 10 of 10')
print(f'CONTROL lift needed to reach the {COMMIT:.0%} commitment threshold '
      f'{T.need.min():.1f}-{T.need.max():.1f}x against a measured {T.lift.min():.2f}-{T.lift.max():.2f}x; '
      f'pooled, the effect delivers {100*(r1-r0):.1f} points of the {100*(COMMIT-r0):.1f} needed, '
      f'a shortfall of {(COMMIT-r0)/(r1-r0):.0f}x')
print('         the paper says "about forty times short"; the measured shortfall in probability '
      'points is the number above')

# ---------------------------------------------------------------- draw
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.89, 2.62), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
yy = np.arange(len(T))

for i, r in T.iterrows():
    a1.plot([100 * r.p0, 100 * r.p1], [i, i], color=CONTEXT, lw=1.0, zorder=1)
a1.errorbar(100 * T.p0, yy, xerr=100 * 1.96 * T.se0, fmt='none', ecolor=REF_LIGHT, elinewidth=0.7)
a1.errorbar(100 * T.p1, yy, xerr=100 * 1.96 * T.se1, fmt='none', ecolor=RECORD_TINT, elinewidth=0.7)
a1.scatter(100 * T.p0, yy, s=17, marker='s', facecolor='white', edgecolor=INK, linewidth=0.8, zorder=3)
a1.scatter(100 * T.p1, yy, s=20, marker='o', color=RECORD, linewidth=0, zorder=4)
a1.set_yticks(yy); a1.set_yticklabels(T.apt)
a1.set_ylim(-0.75, len(T) + 0.9); a1.set_xlim(0, 21)
a1.set_xlabel('departures recorded by schedule copy   (%)')
a1.set_title('a.  same airport, same operator mix', loc='left', pad=4)
a1.scatter([16.0], [1.75], s=20, marker='o', color=RECORD, linewidth=0, clip_on=False)
a1.text(16.7, 1.75, 'the aircraft\'s previous\nleg WAS a copy', fontsize=7.3, color=RECORD,
        va='center')
a1.scatter([16.0], [0.35], s=17, marker='s', facecolor='white', edgecolor=INK,
           linewidth=0.8, clip_on=False)
a1.text(16.7, 0.35, 'it was not', fontsize=7.3, color=INK, va='center')

a2.axvline(1.0, color=INK, lw=0.7)
a2.hlines(yy, T.lo, T['hi'], color=CONTEXT, lw=1.0, zorder=1)
a2.scatter(T.lift, yy, s=20, marker='o', color=RECORD, linewidth=0, zorder=3)
a2.scatter(T.need, yy, s=70, marker='|', color=MEASURED, linewidth=1.3, zorder=3)
a2.set_xscale('log'); a2.set_xlim(0.9, 9.5)
a2.set_yticks(yy); a2.set_yticklabels([]); a2.set_ylim(-0.75, len(T) + 0.9)
a2.set_xticks([1, 1.5, 2, 3, 5, 8])
a2.set_xticklabels(['1x', '1.5x', '2x', '3x', '5x', '8x'])
a2.xaxis.set_minor_formatter(NullFormatter())
a2.tick_params(axis='x', which='minor', length=0)
a2.set_xlabel('lift in the schedule-copy rate   (log)')
a2.set_title('b.  positive at ten airports of ten, and not nearly enough', loc='left', pad=4)
a2.text(1.02, -0.45, 'no effect', fontsize=7.3, color=INK, ha='left', va='center')
a2.text(1.12, len(T) + 0.45, 'measured, with 95% interval', fontsize=7.3, color=RECORD,
        ha='left', va='center')
a2.text(5.6, len(T) + 0.45, 'needed to act', fontsize=7.3, color=MEASURED, ha='center', va='center')

fig.tight_layout(pad=0.35)
for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'airframe-carryover.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/airframe-carryover.{pdf,png}')
