"""Figure: the one-day date error is a date resolved against the wrong reference (paper section 1.7).

Measured on all Fiumicino LOBT-null departures of 2025, in Europe/Rome local time, which is the
population section 1.7 reports on. The schematic in panel a is drawn from one real row; every
timestamp in it is printed by the control block below.

Controls printed against the paper:
  LIRF LOBT-null departures                       1,488
  of them, take-off does NOT cross the local date 1,320
  rollovers, y = 86,400 s + an ordinary taxi         12
  every rollover:  localdate(BLOCK) == localdate(SCHED)
                   localdate(MVT)   == localdate(SCHED) + 1
  rollovers among the 1,320 non-crossing rows         0
  crossing rows with MVT - SCHED >= 20,000 s:  36 schedule-copy, 11 rollover-only, 0 ordinary
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
from fig_palette import INK, REF, REF_LIGHT, MEASURED, MEASURED_TINT, RECORD, CONTEXT
EPOCH = pd.Timestamp('1970-01-01', tz='UTC'); TZ = 'Europe/Rome'


def secs(s):
    return (pd.to_datetime(s, errors='coerce', utc=True) - EPOCH).dt.total_seconds()


C = ['PHASE_mvt', 'ADEP_mvt', 'FLIGHT_mvt', 'STAND_mvt', 'MVT_TIME_UTC_mvt',
     'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'TAXITIME_SEC_mvt', 'LOBT_flt']
R = pd.concat([pq.read_table(f, columns=C).to_pandas()
               for f in sorted(glob.glob(f'{D}/training_*.parquet'))], ignore_index=True)
d = R[R.PHASE_mvt.eq('DEP')].copy()
d['mvt'] = secs(d.MVT_TIME_UTC_mvt); d['blk'] = secs(d.BLOCK_TIME_UTC_mvt); d['sch'] = secs(d.SCHED_TIME_UTC_mvt)
assert 1.735e9 < float(d.mvt.median()) < 1.80e9, f'timestamp units {d.mvt.median():.4g}'
L = d[d.ADEP_mvt.astype(str).eq('LIRF') & d.LOBT_flt.isna()].copy()
L['y'] = L.TAXITIME_SEC_mvt.astype(float)
for c, nm in (('MVT_TIME_UTC_mvt', 'mvtl'), ('BLOCK_TIME_UTC_mvt', 'blkl'), ('SCHED_TIME_UTC_mvt', 'schl')):
    L[nm] = pd.to_datetime(L[c], utc=True).dt.tz_convert(TZ)
for nm in ('mvtl', 'blkl', 'schl'):
    L[nm + '_d'] = L[nm].dt.normalize()
    L[nm + '_t'] = (L[nm].dt.hour + L[nm].dt.minute / 60 + L[nm].dt.second / 3600)
L['nd'] = (L.mvtl_d - L.blkl_d).dt.days
L['cross'] = L.mvtl_d.ne(L.schl_d)
L['cp'] = (L.blk - L.sch).abs() <= 60
L['ms'] = L.mvt - L.sch
# "86,400 s plus an ordinary taxi": exclude the one 1.5-day row the paper's count excludes
L['roll'] = (L.nd == 1) & (L.y - 86400 >= 300) & (L.y - 86400 <= 7200)

print(f'CONTROL LIRF LOBT-null departures {len(L):,}                        paper 1,488')
print(f'CONTROL take-off does NOT cross the local date {int((~L.cross).sum()):,}          paper 1,320')
print(f'CONTROL rollovers, y = 86,400 + an ordinary taxi {int(L.roll.sum())}             paper 12')
r = L[L.roll]
print(f'CONTROL of them localdate(BLOCK) == localdate(SCHED) on {int((r.blkl_d == r.schl_d).sum())}/{len(r)}'
      f'   and localdate(MVT) == localdate(SCHED)+1 on '
      f'{int((r.mvtl_d == r.schl_d + pd.Timedelta(days=1)).sum())}/{len(r)}   paper 12/12')
print(f'CONTROL rollovers among the non-crossing rows {int(L[~L.cross].roll.sum())}                paper 0')
cr = L[L.cross & (L.ms >= 20000)]
print(f'CONTROL crossing, MVT - SCHED >= 20,000 s: n {len(cr)}, schedule-copy {int(cr.cp.sum())}, '
      f'rollover-and-not-a-copy {int((cr.roll & ~cr.cp).sum())}, neither '
      f'{int((~cr.cp & ~cr.roll).sum())}   paper 36 / 11 / 0')
nc = L[(~L.cross) & (L.ms >= 30000)]
print(f'CONTROL non-crossing, MVT - SCHED >= 30,000 s: n {len(nc)}, schedule-copy {int(nc.cp.sum())}, '
      f'rollover {int(nc.roll.sum())}   paper 8 / 8 / 0')
assert len(L) == 1488 and int((~L.cross).sum()) == 1320 and int(L.roll.sum()) == 12, 'disagrees with paper'
print(f'CONTROL y - 86,400 over the twelve: min {int((r.y-86400).min())}  median '
      f'{int((r.y-86400).median())}  max {int((r.y-86400).max())} s, against LIRF ordinary median '
      f'{int(d[d.ADEP_mvt.astype(str).eq("LIRF")].TAXITIME_SEC_mvt.median())} s')

# the row drawn in panel a: the one whose local clock times are roundest
ex = r.sort_values('y').iloc[6]
print(f'PANEL a row: {ex.FLIGHT_mvt} stand {ex.STAND_mvt}  y {ex.y:,.0f} s')
print(f'  SCHED  local {ex.schl}\n  BLOCK  local {ex.blkl}\n  MVT    local {ex.mvtl}')
print(f'  BLOCK local time-of-day {ex.blkl.strftime("%H:%M:%S")} against take-off '
      f'{ex.mvtl.strftime("%H:%M:%S")}: a {ex.y-86400:,.0f} s taxi one day later')

# ---------------------------------------------------------------- draw
# Explicit axes, in inches (author review, round 2): panel b square, both panels one height, its legend outside it.
FW, FH = 6.89, 2.62
fig = plt.figure(figsize=(FW, FH))
ax_in = lambda l, b, w, h: fig.add_axes([l / FW, b / FH, w / FW, h / FH])
a1 = ax_in(0.00, 0.10, 3.05, 2.32)
a2 = ax_in(3.62, 0.42, 2.00, 2.00)

# ---- panel a: the mechanism, drawn from the row above
T0 = 6   # the day lines start at 06:00: nothing on either day happens before it (author review, round 2)
a1.set_xlim(3.2, 25.4); a1.set_ylim(-0.95, 2.72); a1.axis('off')
D1, D2 = 1.75, 0.35
for yy, lab in ((D1, ex.schl.strftime('%-d %b')), (D2, ex.mvtl.strftime('%-d %b'))):
    a1.annotate('', xy=(24.3, yy), xytext=(T0, yy),
                arrowprops=dict(arrowstyle='-|>', color=REF_LIGHT, lw=0.8))
    for h in (6, 12, 18, 24):
        a1.plot([h, h], [yy - 0.055, yy + 0.055], lw=0.6, color=REF_LIGHT)
    a1.text(T0 - 0.35, yy, lab, ha='right', va='center', fontsize=7.6, color=INK)
a1.text(24.3, D2 - 0.30, 'local time of day', ha='right', va='top', fontsize=7.0, color=INK)

xs, xb, xm = ex.schl_t, ex.blkl_t, ex.mvtl_t
a1.plot([xs], [D1], 'o', ms=5.0, color=REF, zorder=4)
a1.text(xs, D1 + 0.17, f'SCHED {ex.schl.strftime("%H:%M")}', ha='center', fontsize=7.3, color=INK)
a1.plot([xb], [D1], 'D', ms=6.0, color=RECORD, zorder=4)
a1.text(xb, D1 + 0.17, f'recorded BLOCK {ex.blkl.strftime("%H:%M")}', ha='center',
        fontsize=7.3, color=RECORD)
a1.plot([xb], [D2], 'o', ms=5.0, markerfacecolor='white', markeredgecolor=MEASURED,
        markeredgewidth=1.1, zorder=4)
a1.plot([xm, xm], [D2, D2 + 0.40], lw=0.7, color=INK, zorder=3)
a1.plot([xm], [D2 + 0.40], '*', ms=11, color=INK, zorder=5)
a1.text(xm + 0.45, D2 + 0.40, f'take-off {ex.mvtl.strftime("%H:%M")},\n18 min after the push-back',
        ha='left', va='center', fontsize=7.3, color=INK)
a1.text(xb - 0.5, D2 - 0.18, 'the aircraft actually\npushed back here',
        ha='right', va='top', fontsize=7.3, color=MEASURED)

a1.annotate('', xy=(xb, D1 - 0.11), xytext=(xb, D2 + 0.11),
            arrowprops=dict(arrowstyle='-|>', color=RECORD, lw=1.0))
a1.text(xb - 0.35, (D1 + D2) / 2 + 0.05, 'the clock is kept,\nthe date is taken\nfrom SCHED',
        ha='right', va='center', fontsize=7.3, color=RECORD)
a1.annotate('', xy=(xs, D1 + 0.30), xytext=(xb + 0.55, D1 + 0.30),
            arrowprops=dict(arrowstyle='-|>', color=REF, lw=0.8,
                            connectionstyle='arc3,rad=-0.28'))
a1.text((xs + xb) / 2 + 0.3, D1 + 0.62, 'date from here', ha='center', fontsize=7.2, color=INK)

a1.annotate('', xy=(xm, -0.42), xytext=(xb, -0.42),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=0.9, shrinkA=0, shrinkB=0))
a1.plot([xb, xb], [D1 - 0.16, -0.42], ls=(0, (1, 2)), lw=0.7, color=INK)
a1.plot([xm, xm], [D2 - 0.16, -0.42], ls=(0, (1, 2)), lw=0.7, color=INK)
a1.text((xb + xm) / 2, -0.58, f'y = MVT − BLOCK = {ex.y:,.0f} s = 86,400 + {ex.y-86400:,.0f}',
        ha='center', va='top', fontsize=7.6, color=INK)
a1.set_title(f'a.  one real row: {ex.FLIGHT_mvt}, Fiumicino, '
             f'{ex.mvtl.strftime("%-d %b %Y")}', loc='left', pad=2)

# ---- panel b: on the clock the defect is invisible
oth = L[~L.roll]
bigcp = oth[oth.cp & (oth.y >= 20000)]
plain = oth[~(oth.cp & (oth.y >= 20000))]
a2.plot([0, 24], [0, 24], color=CONTEXT, lw=0.8, zorder=1)
a2.scatter(plain.blkl_t, plain.mvtl_t, s=4, color=MEASURED_TINT, linewidth=0, zorder=2)
a2.scatter(bigcp.blkl_t, bigcp.mvtl_t, s=15, facecolor='none', edgecolor=RECORD,
           linewidth=0.8, zorder=3)
a2.scatter(r.blkl_t, r.mvtl_t, s=22, marker='D', color=RECORD, linewidth=0, zorder=4)
a2.set_xlim(0, 24); a2.set_ylim(0, 24)
a2.set_xticks([0, 6, 12, 18, 24]); a2.set_yticks([0, 6, 12, 18, 24])
a2.set_xlabel('BLOCK, local time of day   (h)')
a2.set_ylabel('take-off, local time of day   (h)')
a2.set_title('b.  1,488 Fiumicino LOBT-null departures', loc='left', pad=4)
from matplotlib.lines import Line2D
H = [Line2D([], [], ls='none', marker='D', ms=4.7, mfc=RECORD, mec='none'),
     Line2D([], [], ls='none', marker='o', ms=3.9, mfc='none', mec=RECORD, mew=0.8),
     Line2D([], [], ls='none', marker='o', ms=2.0, mfc=MEASURED_TINT, mec='none')]
lg = a2.legend(H, ['the 12 rollovers', f'{len(bigcp)} schedule copies,\ny ≥ 20,000 s', f'the other {len(plain):,}\ndepartures'],
               loc='upper left', bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=7.4, handletextpad=0.5,
               labelspacing=0.9, borderaxespad=0.0, handlelength=1.0)
for t, c in zip(lg.get_texts(), (RECORD, RECORD, INK)): t.set_color(c)

for e in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT, f'date-error.{e}'), dpi=200)
plt.close(fig)
print('wrote analysis/figures/date-error.{pdf,png}')
