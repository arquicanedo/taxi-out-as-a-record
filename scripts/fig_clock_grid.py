"""Figure: timestamp granularity identifies the system that wrote each column (paper section 1.3).

Two measurements over all 2,085,047 labelled departures:
  a.  the second-of-minute distribution of BLOCK_TIME per airport, wrapped about :00
  b.  the rate at which BLOCK_TIME and MVT_TIME land on an exact minute, per airport

Controls printed against the paper:
  seven airports at ~8.3% exact-minute BLOCK (EDDM EGLL LEBL LEMD LFPG LIRF LTFM), three at 1.7-2.0%
  the chance baseline under true 1-second resolution is 1/60 = 1.667%
  LEBL and LEMD write MVT_TIME at 1-second and BLOCK_TIME on the coarse grid

It also prints the shape that decides WHICH coarse grid it is, which the paper reads as a
five-second grid: the mass is confined to the twelve seconds {54..59, 0..5} and is flat inside
them, so 8.3% = 1/12 comes from minute resolution plus a uniform jitter of about +-6 s.
"""
import os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from fig_palette import INK, REF, MEASURED, RECORD, CONTEXT
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')


def secs(s):
    return (pd.to_datetime(s, errors='coerce', utc=True) - EPOCH).dt.total_seconds()


C = ['PHASE_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt']
R = pd.concat([pq.read_table(f, columns=C).to_pandas()
               for f in sorted(glob.glob(f'{D}/training_*.parquet'))], ignore_index=True)
d = R[R.PHASE_mvt.eq('DEP')].copy()
d['blk'] = secs(d.BLOCK_TIME_UTC_mvt); d['mvt'] = secs(d.MVT_TIME_UTC_mvt)
assert 1.735e9 < float(d.mvt.median()) < 1.80e9, f'timestamp units {d.mvt.median():.4g}'
d['apt'] = d.ADEP_mvt.astype(str)
print(f'labelled departures {len(d):,}   airports {d.apt.nunique()}')
assert len(d) == 2085047 and d.apt.nunique() == 10, 'population is not the paper\'s'

BASE = 1 / 60
rows = []
prof = {}
for a, g in d.groupby('apt'):
    b = g.blk.dropna().values % 60
    m = g.mvt.dropna().values % 60
    w = ((b + 30) % 60) - 30                       # wrap so :00 sits in the middle
    h = np.bincount(((w + 30).astype(int)), minlength=60) / len(w)
    prof[a] = h
    inband = float(((w >= -6) & (w <= 6)).mean())
    rows.append((a, len(g), (b == 0).mean(), (m == 0).mean(), inband))
T = pd.DataFrame(rows, columns=['apt', 'n', 'blk0', 'mvt0', 'inband']).sort_values('blk0')
coarse = T[T.blk0 > 0.05].apt.tolist()
print(f'CONTROL chance baseline at true 1-second resolution 1/60 = {BASE:.3%}')
print(f'CONTROL coarse-grid airports {sorted(coarse)}')
print(f'         paper EDDM EGLL LEBL LEMD LFPG LIRF LTFM')
assert sorted(coarse) == ['EDDM', 'EGLL', 'LEBL', 'LEMD', 'LFPG', 'LIRF', 'LTFM'], 'set disagrees'
print(f'  {"apt":<6} {"n":>9} {"BLOCK sec==0":>13} {"MVT sec==0":>11} {"BLOCK in [-6,+6]s":>18}')
for _, r in T.iterrows():
    print(f'  {r.apt:<6} {r.n:>9,} {r.blk0:>12.2%} {r.mvt0:>10.2%} {r.inband:>17.1%}')
print(f'CONTROL coarse group BLOCK sec==0 {T[T.blk0>.05].blk0.min():.2%}-{T.blk0.max():.2%}   '
      f'paper 8.3%')
print(f'CONTROL fine group   BLOCK sec==0 {T[T.blk0<.05].blk0.min():.2%}-{T[T.blk0<.05].blk0.max():.2%}'
      f'   paper 1.7-2.0%')
print(f'CONTROL LEBL/LEMD MVT sec==0 '
      f'{T.set_index("apt").loc["LEBL"].mvt0:.2%} / {T.set_index("apt").loc["LEMD"].mvt0:.2%} '
      f'against their BLOCK {T.set_index("apt").loc["LEBL"].blk0:.2%} / '
      f'{T.set_index("apt").loc["LEMD"].blk0:.2%}   paper: two systems, one airport')
g13 = np.mean([prof[a][24:37].sum() for a in coarse])
print(f'CONTROL coarse group: {g13:.3%} of BLOCK seconds fall in the 13 values -6..+6, '
      f'and 1/12 = {1/12:.3%} against the measured mean sec==0 {T[T.blk0>.05].blk0.mean():.3%}')

# ---------------------------------------------------------------- draw
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.89, 2.6), gridspec_kw=dict(width_ratios=[1.12, 1.0]))
x = np.arange(-30, 30)
for a in T.apt:
    col, lw = (RECORD, 1.0) if a in coarse else (MEASURED, 1.0)
    a1.step(x, 100 * prof[a], where='mid', color=col, lw=lw, alpha=0.75)
a1.axhline(100 * BASE, color=REF, lw=0.7, ls=(0, (3, 2)))
a1.set_xlim(-30, 29.5); a1.set_ylim(0, 11.6)
a1.set_xticks([-30, -20, -10, 0, 10, 20, 29])
a1.set_xlabel('second of the minute written into BLOCK_TIME   (wrapped about :00)')
a1.set_ylabel('share of departures   (%)')
a1.set_title('a.  two writing systems, ten airports', loc='left', pad=4)
a1.text(-29, 0.55, 'chance under true 1-second resolution, 1/60 = 1.67%',
        fontsize=7.2, color=INK)
a1.annotate('EDDF, EHAM and LSZH\nwrite true seconds', xy=(-19, 1.70), xytext=(-29, 4.3),
            fontsize=7.4, color=MEASURED,
            arrowprops=dict(arrowstyle='-|>', color=MEASURED, lw=0.7, shrinkA=2, shrinkB=3))
a1.annotate('the other seven confine every\nBLOCK_TIME to the twelve seconds\n'
            '−6 to +5 and are flat inside them:\na minute stamp plus a jitter',
            xy=(5.8, 8.35), xytext=(8.5, 11.4), fontsize=7.4, color=RECORD, va='top',
            arrowprops=dict(arrowstyle='-|>', color=RECORD, lw=0.7, shrinkA=2, shrinkB=3))

yy = np.arange(len(T))
a2.axvline(100 * BASE, color=REF, lw=0.7, ls=(0, (3, 2)))
a2.axvline(100 / 12, color=INK, lw=0.7, ls=(0, (1, 2)))
for i, (_, r) in enumerate(T.iterrows()):
    a2.plot([100 * r.mvt0, 100 * r.blk0], [i, i], color=CONTEXT, lw=0.8, zorder=1)
a2.scatter(100 * T.mvt0, yy, s=17, marker='s', facecolor='white', edgecolor=INK,
           linewidth=0.8, zorder=3)
a2.scatter(100 * T.blk0, yy, s=20, marker='o', color=RECORD, linewidth=0, zorder=4)
a2.set_yticks(yy); a2.set_yticklabels(T.apt)
a2.set_ylim(-0.7, len(T) + 0.15); a2.set_xlim(0, 11.4)
a2.set_xlabel('timestamps landing on an exact minute   (%)')
a2.set_title('b.  and within one airport, by column', loc='left', pad=4)
a2.text(100 * BASE + 0.15, len(T) - 0.15, '1/60', fontsize=7.2, color=INK, va='center')
a2.text(100 / 12 + 0.15, len(T) - 0.15, '1/12', fontsize=7.2, color=INK, va='center')
a2.scatter([7.3], [1.15], s=20, marker='o', color=RECORD, linewidth=0, clip_on=False)
a2.text(7.6, 1.15, 'BLOCK_TIME', fontsize=7.4, color=RECORD, va='center')
a2.scatter([7.3], [0.35], s=17, marker='s', facecolor='white', edgecolor=INK,
           linewidth=0.8, clip_on=False)
a2.text(7.6, 0.35, 'MVT_TIME', fontsize=7.4, color=INK, va='center')
a2.annotate('Barcelona and Madrid write\nthe two columns differently',
            xy=(5.0, 8.0), xytext=(0.45, 6.15), fontsize=7.4, color=INK, va='center',
            arrowprops=dict(arrowstyle='-|>', color=REF, lw=0.7, shrinkA=3, shrinkB=3,
                            connectionstyle='arc3,rad=-0.25'))

fig.tight_layout(pad=0.35)
for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'clock-grid.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/clock-grid.{pdf,png}')
