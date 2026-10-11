"""Figure: a pooled slice mean that survives a permutation null and is one month (section 4.5).

Reproduces the headline slice of `scripts/bias_sweep.py`: the out-of-fold residual of the
twelve-month cross-fit bulk model (`$DC26_WORK/xfit12_bulk.npy`, aligned positionally to
night_tr_all + night_va_all), restricted to ordinary rows, inside the slice LTFM x runway 34L.
That slice scored +157.7 s on the sweep's own statistic against a permutation-null maximum of
36.07 s over 60 shuffles, and the pooled mean residual is +92.6 s.

Controls printed against the paper:
  LTFM x 34L  n = 8,778, pooled mean residual +92.6 s, months positive 2 of 12
  Jan -829  Feb +477  Mar +175  Apr -24  May -33  Jun -8
  Jul  -17  Aug   -6  Sep  -34  Oct -30  Nov  -3  Dec -29

What the paper's twelve numbers do not carry is the row count behind each one, which the figure
adds and which changes the reading.
"""
import os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.environ['DC26_DATA']
W = os.environ['DC26_WORK']
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
from fig_palette import INK, REF, MEASURED, MEASURED_TINT, RECORD
THR = 7200.

CC = ['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'y']
q = pd.concat([pq.read_table(f'{W}/{f}', columns=CC).to_pandas()
               for f in ('night_tr_all.parquet', 'night_va_all.parquet')], ignore_index=True)
bulk = np.load(f'{W}/xfit12_bulk.npy')
assert len(bulk) == len(q), f'cross-fit array is not aligned: {len(bulk)} vs {len(q)}'
q['bulk'] = bulk
q['MVT_TIME_UTC_mvt'] = pd.to_datetime(q.MVT_TIME_UTC_mvt, utc=True)
meta = pd.concat([pq.read_table(f, columns=['MVT_ID_mvt', 'RUNWAY_mvt', 'PHASE_mvt']).to_pandas()
                  for f in sorted(glob.glob(f'{D}/training_*.parquet'))], ignore_index=True)
meta = meta[meta.PHASE_mvt.eq('DEP')]
n0 = len(q)
d = q.merge(meta, on='MVT_ID_mvt', how='inner')
assert len(d) > 0.95 * n0, f'metadata join lost rows: {n0:,} -> {len(d):,}'
print(f'cross-fit rows {n0:,}, after the MVT_ID join {len(d):,}')
d = d[d.y < THR].copy()
d['res'] = d.y - np.clip(d.bulk, 60, 200000)
d['mon'] = d.MVT_TIME_UTC_mvt.dt.month
print(f'ordinary rows {len(d):,}   bulk RMSE {np.sqrt(np.mean(d.res**2)):.3f}   '
      f'mean residual {d.res.mean():+.3f} s (a correct model is 0)')

m = d[d.ADEP_mvt.astype(str).eq('LTFM') & d.RUNWAY_mvt.astype(str).eq('34L')]
g = m.groupby('mon').res.agg(['mean', 'count', 'sem']).reindex(range(1, 13))
PAPER = [-829, 477, 175, -24, -33, -8, -17, -6, -34, -30, -3, -29]
print(f'CONTROL LTFM x 34L n = {len(m):,}, pooled mean residual {m.res.mean():+.1f} s, '
      f'months positive {int((g["mean"] > 0).sum())} of 12   paper 8,778 / +92.6 / 2 of 12')
assert len(m) == 8778 and abs(m.res.mean() - 92.6) < 0.5, 'disagrees with the paper'
print(f'  {"month":>6} {"mean residual":>14} {"paper":>7} {"rows":>7} {"s.e.":>8}')
for i, mm in enumerate(range(1, 13)):
    r = g.loc[mm]
    sv = float(r['sem']) if pd.notna(r['sem']) else float('nan')
    se = '' if not np.isfinite(sv) else f'{sv:.1f}'
    print(f'  {mm:>6} {float(r["mean"]):>+13.1f} {PAPER[i]:>+7} {int(r["count"]):>7,} {se:>8}')
    assert abs(float(r['mean']) - PAPER[i]) < 1.0, f'month {mm} disagrees with the paper'
print(f'CONTROL the two positive months carry {int(g.loc[[2,3],"count"].sum()):,} of the '
      f'{len(m):,} rows; January carries {int(g.loc[1,"count"])} and March {int(g.loc[3,"count"])}')
w = g['mean'] * g['count']
print(f'CONTROL February alone supplies {100*w.loc[2]/w.sum():.0f}% of the pooled sum, '
      f'and dropping it leaves {(w.drop(2).sum()/g["count"].drop(2).sum()):+.1f} s over '
      f'{int(g["count"].drop(2).sum()):,} rows')

# ---------------------------------------------------------------- draw
fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.72, 2.95), sharex=True,
                             gridspec_kw=dict(height_ratios=[3.05, 1.0], hspace=0.13))
x = np.arange(1, 13)
mu = g['mean'].values.astype(float); se = g['sem'].values.astype(float); cnt = g['count'].values.astype(float)
col = [RECORD if v > 0 else MEASURED for v in mu]
a1.bar(x, mu, width=0.66, color=col, linewidth=0)
a1.errorbar(x, mu, yerr=1.96 * np.nan_to_num(se), fmt='none', ecolor='#4a4a4a', elinewidth=0.7,
            capsize=1.8, capthick=0.7)
a1.axhline(0, color='#4a4a4a', lw=0.7)
a1.axhline(m.res.mean(), color=INK, lw=0.9, ls=(0, (4, 2)))
a1.text(12.55, m.res.mean(), f'  pooled\n  {m.res.mean():+.1f} s', fontsize=7.3, color=INK,
        va='center', ha='left')
a1.set_ylim(-980, 700)
a1.set_ylabel('mean out-of-fold residual   (s)')
a1.set_title('LTFM × runway 34L, 8,778 ordinary departures', loc='left', pad=4)
a1.annotate('one row', xy=(1.36, -790), xytext=(2.15, -880), fontsize=7.3, color=MEASURED,
            ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', color=MEASURED, lw=0.7, shrinkA=2, shrinkB=3))
a1.annotate('five rows', xy=(3.36, 320), xytext=(4.1, 610), fontsize=7.3,
            color=RECORD, ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', color=RECORD, lw=0.7, shrinkA=2, shrinkB=3))
a1.text(8.5, -430, 'nine months within 34 s of zero', fontsize=7.4, color=MEASURED, ha='center')
a1.annotate('', xy=(4.6, -330), xytext=(12.4, -330),
            arrowprops=dict(arrowstyle='|-|,widthA=0.25,widthB=0.25', color=MEASURED, lw=0.7))

a2.bar(x, cnt, width=0.66, color=MEASURED_TINT, linewidth=0)
a2.set_ylim(0, 2400); a2.set_yticks([0, 1000, 2000])
a2.set_yticklabels(['0', '1,000', '2,000'])
a2.set_ylabel('rows', labelpad=1)
a2.set_xticks(x)
a2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
a2.set_xlabel('month of 2025')
a2.set_xlim(0.4, 12.6)
a2.annotate(f'drop February and the slice reads '
            f'{(w.drop(2).sum()/g["count"].drop(2).sum()):+.1f} s'.replace('-', '\u2212'),
            xy=(2.4, 1900), xytext=(3.3, 2150), fontsize=7.3, color=INK, va='center',
            arrowprops=dict(arrowstyle='-|>', color=REF, lw=0.7, shrinkA=2, shrinkB=3))

fig.tight_layout(pad=0.35)
for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'slice-months.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/slice-months.{pdf,png}')
