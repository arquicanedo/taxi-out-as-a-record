"""Figure: the target's tail, and where the squared error actually sits.

Left panel is label-only over the full labelled year (2,085,047 departures). Right panel needs the
deployed model's residuals, so it runs on the 344,336-row validation window whose shipped-blend
context is cached in $DC26_WORK/b4ab_ctx.npz (positional, nothing is joined to it: s390).

Controls printed against the paper:
  labelled departures  2,085,047      extremes (y >= 7200) 584 = 0.0280%
  window               344,336 rows   149 extremes = 0.0433% of rows, 40.12% of SSE
  the single LFPG row  6.969e+09 = 30.61% of the scored SSE at RMSE 256.9600
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
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
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
EPOCH = pd.Timestamp('1970-01-01', tz='UTC')


def secs(s):
    return (pd.to_datetime(s, errors='coerce', utc=True) - EPOCH).dt.total_seconds()


# ---------------------------------------------------------------- the labelled year
C = ['PHASE_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt']
R = pd.concat([pq.read_table(f, columns=C).to_pandas()
               for f in sorted(glob.glob(f'{D}/training_*.parquet'))], ignore_index=True)
d = R[R.PHASE_mvt.eq('DEP')].copy()
d['mvt'] = secs(d.MVT_TIME_UTC_mvt); d['blk'] = secs(d.BLOCK_TIME_UTC_mvt)
assert 1.735e9 < float(d.mvt.median()) < 1.80e9, f'timestamp units {d.mvt.median():.4g}'
y = d.TAXITIME_SEC_mvt.astype(float).values
assert np.nanmax(np.abs(y - (d.mvt - d.blk).values)) == 0.0, 'y is not MVT - BLOCK'
ext_year = y >= 7200
print(f'CONTROL labelled departures {len(y):,}                    paper 2,085,047')
print(f'CONTROL extremes (y >= 7200) {int(ext_year.sum())} = {ext_year.mean():.4%}   paper 584 = 0.0280%')
day = (np.abs(y - 86400) < 7200)
print(f'CONTROL rows within 2 h of exactly one day  {int(day.sum())}')

# ---------------------------------------------------------------- the deployed model on the window
FLOOR, CAP, DAY, TEMP, GFLOOR = 60., 200000., 86400., 0.90, 0.10
c = np.load(f'{W}/b4ab_ctx.npz')
yv, bulkv, vsd, vlb, ext = c['yv'], c['bulkv'], c['vsd'], c['vlb'], c['ext'].astype(bool)
P = np.load(f'{W}/b4ab_P_u.npy')
Q = P ** TEMP; Q = Q / Q.sum(1, keepdims=True)
Q[:, 1:][Q[:, 1:] < GFLOOR] = 0.0; Q[:, 0] = 1.0 - Q[:, 1:].sum(1)
Q = np.clip(Q, 1e-12, None); Q = Q / Q.sum(1, keepdims=True)
V = np.stack([bulkv, vsd, vlb, np.clip(DAY + bulkv, FLOOR, CAP), bulkv], 1)
pred = np.clip(np.rint((Q * V).sum(1)), FLOOR, CAP)
se = (pred - yv) ** 2; tot = se.sum(); o = np.argsort(-se)
board_sse = 256.96 ** 2 * 344841
print(f'CONTROL window {len(yv):,} rows, {int(ext.sum())} extremes = {ext.mean():.4%} of rows, '
      f'{se[ext].sum()/tot:.2%} of SSE   paper 0.0433% / 40.12%')
print(f'CONTROL deployed RMSE {np.sqrt(se.mean()):.4f}                   paper 313.8931')
print(f'CONTROL largest single row {se[o[0]]:.4g} SSE = {se[o[0]]/board_sse:.2%} of the scored SSE '
      f'(y {yv[o[0]]:,.0f}, prediction {pred[o[0]]:,.0f})   paper 6.969e+09 = 30.61%, y 84,240, pred 759')
assert abs(se[o[0]] / board_sse - 0.3061) < 0.001, 'single-row share disagrees with the paper'
assert abs(se[ext].sum() / tot - 0.4012) < 0.0005, 'extreme SSE share disagrees with the paper'

k = np.arange(1, len(se) + 1)
cum = np.cumsum(se[o]) / tot
for kk in (1, 2, 20, 149, 1000):
    print(f'   top {kk:>6,} rows carry {cum[kk-1]:6.2%} of window SSE')

# ---------------------------------------------------------------- draw
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.89, 2.55))

bins = np.logspace(np.log10(60), np.log10(2e5), 110)
a1.hist(np.clip(y, 60, 2e5), bins=bins, color=MEASURED_TINT, edgecolor=MEASURED, linewidth=0.35)
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xlim(60, 2.2e5); a1.set_ylim(0.7, 2e7)
a1.axvline(7200, color=RECORD, lw=0.9, ls=(0, (4, 2)))
a1.axvline(86400, color=INK, lw=0.7, ls=(0, (1, 2)))
# three lines, so the label ends left of the 7,200 s line instead of running through it (author review, round 2)
a1.annotate(f'{int(ext_year.sum())} extremes beyond\ny = 7,200 s, 0.0280% of rows;\non the window, 40% of the loss',
            xy=(9500, 1.3e3), xytext=(75, 2.2e6), fontsize=7.4, color=RECORD, ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', color=RECORD, lw=0.7, shrinkA=3, shrinkB=3,
                            connectionstyle='arc3,rad=-0.18'))
a1.annotate('exactly\none day', xy=(86400, 25), xytext=(1.15e5, 1.3e3), fontsize=7.4, color=INK,
            ha='center', arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.6, shrinkA=1, shrinkB=3))
a1.set_xlabel('recorded taxi-out  y = MVT_TIME − BLOCK_TIME   (s, log)')
a1.set_ylabel('departures per bin')
a1.set_title(f'a.  the labelled year, {len(y):,} departures', loc='left', pad=4)
a1.set_xticks([60, 600, 3600, 20000, 86400])
a1.set_xticklabels(['60', '600', '3,600', '20,000', '86,400'])
a1.tick_params(axis='x', which='minor', length=0)

a2.plot(k, 100 * cum, color=MEASURED, lw=1.2)
a2.set_xscale('log'); a2.set_xlim(1, len(se)); a2.set_ylim(0, 100)
a2.plot(k, 100 * k / len(se), color=REF, lw=0.8, ls=(0, (3, 2)))
a2.text(1.1e5, 12, 'if every row\ncost the same', fontsize=7.3, color=INK, ha='right')
for kk, off in ((1, (9, -9)), (149, (11, -11)), (1000, (12, -10))):
    a2.plot([kk], [100 * cum[kk-1]], 'o', ms=3.4, color=RECORD, zorder=4)
    lab = {1: 'the single worst row, 20.5%',
           149: 'the 149 worst rows, 49.9%',
           1000: 'the 1,000 worst, 60.5%'}[kk]
    a2.annotate(lab, xy=(kk, 100 * cum[kk-1]), xytext=off, textcoords='offset points',
                fontsize=7.4, color=RECORD, ha='left', va='center')
from matplotlib.ticker import FixedLocator, FixedFormatter
a2.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000, 100000]))
a2.xaxis.set_minor_locator(FixedLocator([]))
a2.xaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1,000', '10,000', '100,000']))
a2.set_xlabel('rows, ranked by their squared error   (log)')
a2.set_ylabel('cumulative share of squared error  (%)')
a2.set_title("b.  the base model's validation version, 344,336-row window", loc='left', pad=4)

fig.tight_layout(pad=0.35)
for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'loss-concentration.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/loss-concentration.{pdf,png}')
