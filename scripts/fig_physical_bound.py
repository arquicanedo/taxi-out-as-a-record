"""Figure: the label violates a bound that no aircraft can violate, and only on the extremes.

Reproduces the construction of `scripts/opdi_veto_val.py` (paper section 1.1): link each January
and July 2025 departure to the previous arrival of the SAME REGISTRATION at the same airport,
using EUROCONTROL OPDI `flight_list` keyed on (callsign, day of flight). Then

    y = MVT_dep - BLOCK_dep   and   BLOCK_dep >= INBLOCK_prev_arrival
    =>  y <= MVT_dep - INBLOCK_prev_arrival  =:  bound

Controls printed against the paper:
  chained departures      306,667
  TRUE y exceeds bound    0.097% of chained rows (299)
  among extremes          34.1% (n = 88)
"""
import os, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, FixedFormatter

D = os.environ['DC26_DATA']
OP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'opdi')
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
from fig_palette import INK, REF, MEASURED, RECORD, DENSITY
EPOCH = pd.Timestamp('1970-01-01', tz='UTC'); THR = 7200.

# ---------------------------------------------------------------- external identity
fl = pd.concat([pq.read_table(f'{OP}/flight_list_{m}.parquet',
                              columns=['flt_id', 'dof', 'registration']).to_pandas()
                for m in ('202501', '202507')], ignore_index=True)
fl['dof'] = fl.dof.astype(str).str[:10]
fl['fid'] = fl.flt_id.astype(str).str.strip().str.upper()
fl = fl.dropna(subset=['registration']).drop_duplicates(['fid', 'dof'])
lut = fl.set_index(['fid', 'dof'])
print(f'OPDI Jan + Jul 2025: {len(lut):,} keyed flights, {fl.registration.nunique():,} airframes')

C = ['MVT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt',
     'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'FLIGHT_mvt', 'CALLSIGN_flt']
raw = pd.concat([pq.read_table(f, columns=C).to_pandas()
                 for f in sorted(glob.glob(f'{D}/training_*.parquet'))],
                ignore_index=True).drop_duplicates('MVT_ID_mvt')
raw['t'] = (pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
raw['bt'] = (pd.to_datetime(raw.BLOCK_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
assert 1.735e9 < float(raw.t.median()) < 1.80e9, f'timestamp units {raw.t.median():.4g}'
raw['dof'] = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d')
raw = raw[pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.month.isin([1, 7])].copy()
raw['apt'] = raw.ADEP_mvt.where(raw.PHASE_mvt == 'DEP', raw.ADES_mvt).astype(str)


def reg_of(df):
    out = pd.Series(index=df.index, dtype=object)
    for col in (df.CALLSIGN_flt, df.FLIGHT_mvt):
        k = pd.MultiIndex.from_arrays([col.astype(str).str.strip().str.upper(), df.dof])
        out = out.where(out.notna(), pd.Series(lut.reindex(k).registration.values, index=df.index))
    return out


raw['reg'] = reg_of(raw)
dep = raw[(raw.PHASE_mvt == 'DEP') & raw.TAXITIME_SEC_mvt.notna()].copy()
arr = raw[raw.PHASE_mvt == 'ARR'].copy()
print(f'Jan + Jul departures {len(dep):,}, registration found on {dep.reg.notna().mean():.1%}')

a = arr[arr.reg.notna() & arr.bt.notna()].sort_values(['apt', 'reg', 'bt'])
ai = {k: v.bt.values for k, v in a.groupby(['apt', 'reg'], sort=False)}
d = dep[dep.reg.notna()].copy()
prev = np.full(len(d), np.nan)
for i, (k, tt) in enumerate(zip(zip(d.apt, d.reg), d.t.values)):
    v = ai.get(k)
    if v is None:
        continue
    j = np.searchsorted(v, tt) - 1
    if j >= 0:
        prev[i] = v[j]
bound = d.t.values - prev
ch = np.isfinite(bound)
y = d.TAXITIME_SEC_mvt.values.astype(float)
# the window is a turnaround in seconds: assert it is physically possible before reading a rate
qs = np.quantile(bound[ch], [.01, .25, .50, .95]) / 3600
print(f'turnaround quantiles (h): 1% {qs[0]:.1f}  25% {qs[1]:.1f}  50% {qs[2]:.1f}  95% {qs[3]:.1f}')
assert 0.5 < qs[2] < 6.0, 'median turnaround is not physically plausible'

Y, B = y[ch], bound[ch]
viol = Y > B
ex = Y >= THR
print(f'CONTROL chained departures {int(ch.sum()):,}                  paper 306,667')
print(f'CONTROL TRUE y exceeds the bound {viol.mean():.3%} ({int(viol.sum())})   paper 0.097% (299)')
print(f'CONTROL among extremes n = {int(ex.sum())}: {viol[ex].mean():.1%}        paper 34.1%, n = 88')
print(f'CONTROL split extreme/ordinary {viol[ex].mean()/viol[~ex].mean():.0f}x, '
      f'extreme/all {viol[ex].mean()/viol.mean():.0f}x   paper quotes 350x = extreme/all')
assert abs(viol.mean() - 0.00097) < 5e-5 and abs(viol[ex].mean() - 0.341) < 0.01, 'disagrees with paper'

tol = np.arange(0, 3601, 60.)
r_ord = np.array([(Y[~ex] > B[~ex] + t).mean() for t in tol])
r_ext = np.array([(Y[ex] > B[ex] + t).mean() for t in tol])
print(f'   at a 1,800 s tolerance: ordinary {r_ord[30]:.4%}, extreme {r_ext[30]:.1%}')

# ---------------------------------------------------------------- draw
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.89, 2.75), gridspec_kw=dict(width_ratios=[1.28, 1.0]))

lo, hi = 300., 2.5e5
hb = a1.hexbin(np.clip(B[~ex], lo, hi), np.clip(Y[~ex], lo, hi), xscale='log', yscale='log',
               gridsize=58, cmap=DENSITY, norm=LogNorm(vmin=1, vmax=3e4),
               extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)),
               linewidths=0, mincnt=1)
a1.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
a1.scatter(np.clip(B[ex & ~viol], lo, hi), np.clip(Y[ex & ~viol], lo, hi),
           s=9, facecolor='none', edgecolor=REF, linewidth=0.7, zorder=3)
a1.scatter(np.clip(B[ex & viol], lo, hi), np.clip(Y[ex & viol], lo, hi),
           s=13, marker='D', color=RECORD, linewidth=0, zorder=4)
a1.set_xscale('log'); a1.set_yscale('log'); a1.set_xlim(lo, hi); a1.set_ylim(lo, hi)
a1.set_xlabel('bound = MVT − in-block of the same airframe   (s, log)')
a1.set_ylabel('recorded taxi-out  y   (s, log)')
a1.set_title('a.  the extremes sit on the impossible side', loc='left', pad=4)
a1.text(0.030, 0.975, 'IMPOSSIBLE: the aircraft had not yet arrived',
        transform=a1.transAxes, fontsize=7.4, color=RECORD, va='top', ha='left')
a1.scatter([0.055], [0.895], s=13, marker='D', color=RECORD, linewidth=0,
           transform=a1.transAxes, clip_on=False, zorder=5)
a1.text(0.085, 0.895, f'{int((ex & viol).sum())} of the {int(ex.sum())} extremes',
        transform=a1.transAxes, fontsize=7.4, color=RECORD, va='center', ha='left')
a1.scatter([0.055], [0.825], s=9, facecolor='none', edgecolor=REF, linewidth=0.7,
           transform=a1.transAxes, clip_on=False, zorder=5)
a1.text(0.085, 0.825, f'the other {int((ex & ~viol).sum())}',
        transform=a1.transAxes, fontsize=7.4, color=INK, va='center', ha='left')
a1.text(0.975, 0.055, 'physically possible: y ≤ bound', transform=a1.transAxes,
        fontsize=7.4, color=INK, va='bottom', ha='right')
cb = fig.colorbar(hb, ax=a1, pad=0.015, fraction=0.045)
cb.set_label('ordinary departures per cell', fontsize=7.4)
cb.ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000, 10000]))
cb.ax.yaxis.set_minor_locator(FixedLocator([]))
cb.ax.yaxis.set_major_formatter(FixedFormatter(['1', '10', '100', '1,000', '10,000']))
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)

a2.plot(tol, 100 * r_ext, color=RECORD, lw=1.4)
a2.plot(tol, 100 * r_ord, color=MEASURED, lw=1.4)
a2.set_yscale('log'); a2.set_ylim(2e-3, 100); a2.set_xlim(0, 3600)
a2.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100]))
a2.yaxis.set_minor_locator(FixedLocator([]))
a2.yaxis.set_major_formatter(FixedFormatter(['0.01', '0.1', '1', '10', '100']))
a2.set_xticks([0, 900, 1800, 2700, 3600])
a2.set_xlabel('tolerance allowed on the bound   (s)')
a2.set_ylabel('rows exceeding the bound   (%, log)')
a2.set_title('b.  no tolerance closes the gap', loc='left', pad=4)
XA = 160   # clear of the y-axis (author review, 2026-10-05): the gap at zero tolerance, drawn just inside the plot
XL = 1000  # line labels start right of the arrow and its ratio (author review, round 2)
a2.text(XL, 100 * r_ext[3] * 1.5, f'extremes ({int(ex.sum())} rows)', fontsize=7.6, color=RECORD)
a2.text(XL, 100 * r_ord[3] * 3.2, f'ordinary ({int((~ex).sum()):,} rows)', fontsize=7.6, color=MEASURED)
a2.annotate(f'{viol[ex].mean()/viol[~ex].mean():.0f}x', xy=(XA, 100 * np.sqrt(r_ext[0] * r_ord[0])),
            xytext=(7, 0), textcoords='offset points', fontsize=9, color=INK, va='center', fontweight='bold')
a2.annotate('', xy=(XA, 100 * r_ext[0]), xytext=(XA, 100 * r_ord[0]),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=1.6, shrinkA=0, shrinkB=0, mutation_scale=11))

fig.tight_layout(pad=0.35)
for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'physical-bound.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/physical-bound.{pdf,png}')
