"""Figure: the final model's architecture, drawn from real rows rather than boxes.

Four bands, read top to bottom, with the score each one bought on the right:
  record      one real departure (TK1956, Amsterdam, 2025-07-30): every clock the file holds for it, and the ADS-B pushback
  base model  the nine members' blend weights (full_package.npz), and the mixture on one real hedged Fiumicino window row
  stage       every adsb.lol ground sample at Amsterdam that day, the same flight's track over them, and what the stage did
              to each airport's ordinary-row window RMSE (base and stage, the v121 columns of scripts/paper_baselines.py)
  feed rules  THY's early arrivals and departures at Fiumicino, September 2025 to January 2026, copied or not
  right       the RMSE on the 2026 set after each band (leaderboard scores of v100, v117, v118, v121, v126)

  uv run --quiet --with pandas --with pyarrow --with numpy --with matplotlib python scripts/fig_model_architecture.py
"""
import glob, os, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from fig_palette import (INK, REF, MEASURED, MEASURED_MID, RECORD, RECORD_TINT, CONTEXT, GROUND, BAND, SHADE,
                     MEMBER_NET, MEMBER_TREE, MEMBER_LINEAR, STEP, SEQ)
HALO = [pe.withStroke(linewidth=2.2, foreground=BAND[0])]

D = os.environ['DC26_DATA']
W = os.environ['DC26_WORK']
SS = os.environ.get('DC26_SESSION', f'{W}/session')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'analysis', 'figures'); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Charter', 'Georgia', 'DejaVu Serif'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#4a4a4a', 'axes.linewidth': 0.6,
    'xtick.color': '#000000', 'ytick.color': '#000000', 'axes.labelcolor': '#000000', 'text.color': '#000000',
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.dpi': 220, 'savefig.pad_inches': 0.03,
})
# (palette roles are imported above HALO, which needs BAND before this line)
FLOOR, CAP, DAY, TEMP, GFLOOR = 60., 200000., 86400., 0.90, 0.10
ts = lambda s: pd.to_datetime(s, utc=True)

# ------------------------------------------------------------------ data: the example departure
MID = 191610811
T7 = pq.read_table(f'{D}/training_2025-07-01_2025-08-01.parquet').to_pandas()
r = T7[T7.MVT_ID_mvt == MID].iloc[0]
F = pd.read_parquet(os.path.join(REPO, 'artifacts', 'adsb_push_feat_full.parquet')) if os.path.exists(os.path.join(REPO, 'artifacts', 'adsb_push_feat_full.parquet')) else None
a = F[F.MVT_ID_mvt == MID].iloc[0]
sched, block, mvt, aobt3 = (ts(r[c]).timestamp() for c in ('SCHED_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'MVT_TIME_UTC_mvt', 'AOBT_3_flt'))
push = float(a.ab_push); y = float(r.TAXITIME_SEC_mvt)
assert y == mvt - block and r.FLIGHT_mvt == 'TK1956', 'example row'
print(f'CONTROL TK1956: y {y:.0f} s, ADS-B taxi {mvt - push:.0f} s, copy would read {mvt - sched:.0f} s, network {mvt - aobt3:.0f} s')
G = pd.read_parquet(f'{W}/adsblol/days/d20250730_ground.parquet', columns=['hex', 't', 'apt', 'lat', 'lon', 'gs'])
G = G[G.apt.eq('EHAM')]
trk = G[G.hex.eq(a.ab_hex) & G.t.between(push - 1500, mvt + 5)].sort_values('t')

# ------------------------------------------------------------------ data: blend weights and one hedged row
fp = np.load(f'{SS}/p2P/full_package.npz')
names = {'L55': 'pooled trees', 'P55': 'per-airport trees', 'K32': 'neighbour network, 32', 'K128': 'neighbour network, 128',
         'ID': 'identity categoricals', 'RQ': 'runway neighbours', 'CL': 'independent build',
         'L55o': 'pooled trees, OOF enc.', 'P55o': 'per-airport, OOF enc.'}
wts = sorted(zip([names[str(k)] for k in fp['members']], fp['w'].astype(float)), key=lambda t: t[1])
c = np.load(f'{W}/b4ab_ctx.npz'); P = np.load(f'{W}/b4ab_P_u.npy')
wi = np.load(f'{SS}/trackA/window_ids.npz', allow_pickle=True); apt = wi['apt'].astype(str)
Qg = P ** TEMP; Qg = Qg / Qg.sum(1, keepdims=True); Qg[:, 1:][Qg[:, 1:] < GFLOOR] = 0.0; Qg[:, 0] = 1.0 - Qg[:, 1:].sum(1)
bulk = np.clip(c['bulkv'], FLOOR, CAP); vsd, yv = c['vsd'], c['yv']
pred = np.clip(np.rint(Qg[:, 0] * bulk + Qg[:, 1] * vsd + Qg[:, 2] * c['vlb'] + Qg[:, 3] * np.clip(DAY + bulk, FLOOR, CAP)
                       + Qg[:, 4] * bulk), FLOOR, CAP)
cand = np.where((apt == 'LIRF') & (Qg[:, 1] > 0.30) & (Qg[:, 1] < 0.70) & (Qg[:, 2] == 0) & (Qg[:, 3] == 0)
                & (vsd > 3500) & (vsd < 8000) & (bulk > 500) & (bulk < 2000) & (np.abs(yv - vsd) <= 60))[0]
print(f'  candidate hedged rows: {len(cand)}')
k = cand[np.argmin(np.abs(Qg[cand, 1] - 0.5))]
print(f'CONTROL hedged LIRF window row: bulk {bulk[k]:.0f}, copy {vsd[k]:.0f}, weight {Qg[k, 1]:.3f}, prediction {pred[k]:.0f}, recorded {yv[k]:.0f}')

# ------------------------------------------------------------------ data: THY at Fiumicino
cols = ['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt']
fs = [f for f in sorted(glob.glob(f'{D}/training_2025-*.parquet')) if any(m in f for m in ('2025-09', '2025-10', '2025-11', '2025-12'))]
X = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in fs + [f'{D}/ranking.parquet']]).drop_duplicates('MVT_ID_mvt')
for cc in ('MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'): X[cc] = ts(X[cc])
X = X[X.FLIGHT_mvt.astype(str).str.startswith('THY') & (X.MVT_TIME_UTC_mvt >= ts('2025-09-01')) & (X.MVT_TIME_UTC_mvt < ts('2026-02-01'))]
X['cp'] = (X.BLOCK_TIME_UTC_mvt - X.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
X['ms'] = (X.MVT_TIME_UTC_mvt - X.SCHED_TIME_UTC_mvt).dt.total_seconds()
arr = X[X.PHASE_mvt.eq('ARR') & X.ADES_mvt.eq('LIRF') & (X.ms < -600)].sort_values('MVT_TIME_UTC_mvt')
dep = X[X.PHASE_mvt.eq('DEP') & X.ADEP_mvt.eq('LIRF')].sort_values('MVT_TIME_UTC_mvt')
jan = dep.MVT_TIME_UTC_mvt >= ts('2026-01-01')
print(f'CONTROL THY LIRF: early arrivals {len(arr)}, Jan 2026 copied {arr.cp[arr.MVT_TIME_UTC_mvt >= ts("2026-01-01")].mean():.0%}; '
      f'departures 2025 {int((~jan).sum())}, Jan 2026 {int(jan.sum())} (blanked)')

# ------------------------------------------------------------------ the figure
fig = plt.figure(figsize=(7.3, 8.6))
# five bands, full width (author review, 2026-10-05): the score is a row of its own at the bottom, not a column
L, R0 = 0.075, 0.985
bands = [(0.816, 0.129), (0.586, 0.195), (0.321, 0.230), (0.156, 0.130), (0.030, 0.088)]   # (bottom, height) per band
labels = ['The record', 'Base model', 'Trajectory stage', 'Record rules', 'Score']
def clear(ax): ax.patch.set_alpha(0.0); return ax
fig.text(0.012, 0.985, 'Model architecture', fontsize=10, weight='bold', va='top')
for i, ((b, h), lab) in enumerate(zip(bands, labels)):
    fig.patches.append(Rectangle((0.005, b - 0.010), R0 - 0.01, h + 0.020, transform=fig.transFigure,
                                 facecolor=BAND[i % 2], edgecolor='none', zorder=-10))
    fig.text(0.016, b + h / 2, lab, rotation=90, va='center', ha='left', fontsize=8.4, weight='bold', color=INK)
for i in range(4):   # data flow between bands
    y0, y1 = bands[i][0] - 0.010, bands[i + 1][0] + bands[i + 1][1] + 0.010
    fig.patches.append(FancyArrowPatch((0.040, y0 + 0.004), (0.040, y1 - 0.004), transform=fig.transFigure, arrowstyle='-|>',
                                       mutation_scale=7, color=REF, lw=0.8))
def farrow(x0, y0, x1, y1, lab, col=REF, dy=0.006):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7,
                                       color=col, lw=0.8, connectionstyle='arc3,rad=0'))
    fig.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, lab, ha='center', va='bottom', fontsize=6.0, color=INK if col == REF else col)

# band 1: the record
b, h = bands[0]
ax = clear(fig.add_axes([L, b, R0 - L - 0.02, h]))
t0 = aobt3 - 120
xs = lambda t: (t - t0) / 60.0
ax.set_xlim(-6.2, xs(mvt) + 1.2); ax.set_ylim(-1.75, 1.75); ax.axis('off')
ax.plot([xs(aobt3) - 0.6, xs(mvt) + 0.8], [0, 0], color=INK, lw=0.8)
ax.text(-6.0, 0.0, 'TK1956\nAmsterdam\n30 Jul 2025', ha='left', va='center', fontsize=6.3, color=INK, linespacing=1.1)   # upright: Agg clips Charter Italic's overhang in the PNG
marks = [(aobt3, 'network "actual"\noff-block, AOBT$_3$', INK, 'v', 'up', 0.0),
         (sched, 'schedule', INK, 'v', 'down', 0.0),
         (push, 'pushback seen\nby ADS-B', MEASURED, 'o', 'up', -0.9),
         (block, 'off-block the\nairport recorded', RECORD, 's', 'down', 0.9),
         (mvt, 'take-off', INK, '^', 'up', 0.0)]
for t, lab, col, m, side, dx in marks:
    if t == push: ax.plot(xs(t), 0, marker='o', ms=9.5, mfc='none', mec=MEASURED, mew=1.1, zorder=4)
    else: ax.plot(xs(t), 0, marker=m, color=col if col != INK else REF, ms=5.4, zorder=3, mec='white', mew=0.5)
    up = side == 'up'
    ax.text(xs(t) + dx, 0.30 if up else -0.30, f'{lab}\n{pd.to_datetime(t, unit="s"):%H:%M:%S}', ha='right' if dx < 0 else 'left' if dx > 0 else 'center',
            va='bottom' if up else 'top', fontsize=6.2, color=col, linespacing=1.05)
def bracket(t1, t2, yb, lab, col, above=True, side=False):
    ax.annotate('', xy=(xs(t1), yb), xytext=(xs(t2), yb), arrowprops=dict(arrowstyle='|-|', color=col, lw=0.7, mutation_scale=2.2))
    if side: ax.text(xs(t1) - 0.4, yb, lab, ha='right', va='center', fontsize=6.4, color=INK if col == REF else col)
    else: ax.text((xs(t1) + xs(t2)) / 2, yb + (0.07 if above else -0.07), lab, ha='center', va='bottom' if above else 'top', fontsize=6.4, color=INK if col == REF else col)
bracket(block, mvt, 1.42, f'the label, y = take-off minus recorded off-block = {y:.0f} s', RECORD, side=True)
bracket(sched, mvt, -1.38, f'had the airport copied the schedule, y would read {mvt - sched:.0f} s', REF, above=False)

# band 2: members and mixture
b, h = bands[1]
ax = clear(fig.add_axes([L + 0.115, b + 0.035, 0.175, h - 0.05]))
yy = np.arange(len(wts))
ax.barh(yy, [w for _, w in wts], color=[MEMBER_NET if 'net' in n else MEMBER_TREE if 'tree' in n or 'per-airport' in n else MEMBER_LINEAR for n, _ in wts], height=0.66)
ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in wts], fontsize=6.2)
for i, (_, w) in enumerate(wts): ax.text(w + 0.006, i, f'{w:.3f}', va='center', fontsize=5.8, color=INK)
ax.set_xlim(0, 0.34); ax.set_xticks([0, 0.1, 0.2, 0.3]); ax.set_xlabel('weight in the blend', fontsize=6.4, labelpad=1.5)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax.text(-0.62, 1.07, 'nine regression members, each on y minus the network\'s taxi time', transform=ax.transAxes, fontsize=6.3, color=INK)
mx0 = L + 0.375
ax = clear(fig.add_axes([mx0, b + 0.055, R0 - mx0 - 0.03, h - 0.085]))
v = [bulk[k], vsd[k]]; wq = [Qg[k, 0], Qg[k, 1]]
ax.set_xlim(0, 9000); ax.set_ylim(-1.2, 1.5); ax.spines['left'].set_visible(False); ax.set_yticks([])
ax.plot(v[0], 0, 'o', color=MEASURED_MID, ms=4 + 16 * wq[0], mec='white', zorder=3)
ax.plot(v[1], 0, 'o', color=RECORD, ms=4 + 16 * wq[1], mec='white', zorder=3)
ax.text(v[0], 0.5, f'bulk, a taxi time\n{v[0]:.0f} s, weight {wq[0]:.2f}', ha='center', va='bottom', fontsize=6.2, color=MEASURED)
ax.text(v[1], 0.5, f'schedule copy\n{v[1]:.0f} s, weight {wq[1]:.2f}', ha='center', va='bottom', fontsize=6.2, color=RECORD)
for xv in v: ax.annotate('', xy=(pred[k] + (-110 if xv < pred[k] else 110), 0), xytext=(xv + (260 if xv < pred[k] else -260), 0),
                         arrowprops=dict(arrowstyle='->', color=REF, lw=0.6))
ax.plot(pred[k], 0, 'D', color=INK, ms=4.4, zorder=4)
ax.text(pred[k], -0.36, f'prediction, a hedge: {pred[k]:.0f} s', ha='center', va='top', fontsize=6.4, color=INK)
ax.text(pred[k], -0.80, f'this Fiumicino row turned out a copy ({yv[k]:.0f} s)', ha='center', va='top', fontsize=5.9, color=INK, style='italic')
ax.set_xlabel('seconds', fontsize=6.4, labelpad=1.5); ax.tick_params(axis='x', labelsize=6.2)
ax.text(0.0, 1.10, 'a classifier weights the bulk against values read from the\nrecord: the schedule, the flight plan, a one-day date error',
        transform=ax.transAxes, fontsize=6.3, color=INK, va='bottom')
farrow(L + 0.315, b + h * 0.42, mx0 - 0.01, b + h * 0.42, 'bulk')

# band 3: the trajectory stage
b, h = bands[2]
ax = clear(fig.add_axes([L + 0.0, b - 0.005, 0.345, h - 0.01]))
lat0 = np.deg2rad(52.31); kx = 111.32 * np.cos(lat0); ky = 110.57
gx, gy = (G.lon.values - 4.76) * kx, (G.lat.values - 52.31) * ky
ax.scatter(gx, gy, s=0.06, color=GROUND, alpha=0.35, lw=0, rasterized=True)
tx, ty, tt = (trk.lon.values - 4.76) * kx, (trk.lat.values - 52.31) * ky, trk.t.values
seg = np.stack([np.column_stack([tx[:-1], ty[:-1]]), np.column_stack([tx[1:], ty[1:]])], 1)
lc = LineCollection(seg, cmap=SEQ, array=(tt[:-1] - push) / 60, linewidths=1.8, zorder=3); ax.add_collection(lc)
ip = np.argmin(np.abs(tt - push))
ax.plot(tx[ip], ty[ip], 'o', color=MEASURED, ms=5.5, mec='white', mew=0.7, zorder=5)
ax.annotate(f'parked at G07, then pushback\n{pd.to_datetime(push, unit="s"):%H:%M:%S}', xy=(tx[ip], ty[ip]), xytext=(tx[ip] - 2.6, ty[ip] + 1.0),
            fontsize=6.0, color=MEASURED, arrowprops=dict(arrowstyle='-', color=MEASURED, lw=0.5), path_effects=HALO)
ax.plot(tx[-1], ty[-1], '^', color=INK, ms=5, zorder=5)
ax.annotate(f'take-off, runway 24\n{pd.to_datetime(mvt, unit="s"):%H:%M:%S}', xy=(tx[-1], ty[-1]), xytext=(tx[-1] - 3.2, ty[-1] - 0.9),
            fontsize=6.0, color=INK, arrowprops=dict(arrowstyle='-', color=INK, lw=0.5), path_effects=HALO)
cx, cy = np.median(tx), np.median(ty)
ax.set_aspect('equal'); ax.set_xlim(cx - 4.2, cx + 3.4); ax.set_ylim(cy - 3.1, cy + 3.0); ax.axis('off')
ax.text(0.0, 1.0, f'every adsb.lol ground sample at Amsterdam that day, and\nthis flight\'s track: taxi {mvt - push:.0f} s, recorded {y:.0f} s',
        transform=ax.transAxes, fontsize=6.3, va='top', color=INK, path_effects=HALO)
dx0 = L + 0.445
ax = clear(fig.add_axes([dx0, b + 0.03, R0 - dx0 - 0.075, h - 0.07]))
tab = [('EDDF', 175.0, 121.0, 60.6), ('EDDM', 168.4, 122.1, 96.9), ('EGLL', 234.1, 217.8, 89.8), ('EHAM', 166.5, 104.8, 96.9),
       ('LEBL', 212.4, 184.0, 94.7), ('LEMD', 167.8, 159.1, 78.0), ('LFPG', 237.3, 236.6, 7.3), ('LIRF', 496.5, 495.5, 29.8),
       ('LSZH', 172.2, 118.8, 95.0), ('LTFM', 229.8, 229.8, 0.0)]
tab = sorted(tab, key=lambda t: t[1] - t[2])
for i, (ap, b0, b1, cov) in enumerate(tab):
    ax.plot([b0, b1], [i, i], color=CONTEXT, lw=2.4, solid_capstyle='round', zorder=1)
    ax.plot(b0, i, 'o', color=REF, ms=3.4, zorder=2); ax.plot(b1, i, 'o', color=MEASURED, ms=3.8, zorder=3)
    ax.text(1.04, i, f'{cov:.0f}%', va='center', ha='left', fontsize=5.9, color=INK, transform=ax.get_yaxis_transform())
ax.text(1.04, len(tab) - 0.45, 'tracked\nin 2026', ha='left', va='bottom', fontsize=5.6, color=INK, linespacing=1.0,
        transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(tab))); ax.set_yticklabels([t[0] for t in tab], fontsize=6.2)
ax.set_xlim(90, 515); ax.set_ylim(-0.7, len(tab) + 0.3); ax.set_xticks([100, 200, 300, 400, 500])
ax.set_xlabel('ordinary-row window RMSE (s), base to stage', fontsize=6.3, labelpad=1.5)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
ax.text(-0.13, 1.10, 'three layers learn when to trust the track: large gains\nwhere receivers see it, almost none at Fiumicino', transform=ax.transAxes,
        fontsize=6.3, color=INK, va='bottom')
farrow(L + 0.335, b + h * 0.45, dx0 - 0.045, b + h * 0.45, 'pushback')

# band 4: feed rules
b, h = bands[3]
ax = clear(fig.add_axes([L + 0.10, b + 0.035, R0 - L - 0.13, h - 0.045]))
x0 = ts('2025-09-01'); dd = lambda s: (s - x0).dt.total_seconds() / 86400
ya, yd = 1.0, 0.0
def strip(df, yv0, seed, fill=True):
    jit = np.random.default_rng(seed).uniform(-0.2, 0.2, len(df))
    if fill:
        ax.scatter(dd(df.MVT_TIME_UTC_mvt), yv0 + jit, s=6, c=np.where(df.cp, RECORD, 'white'), edgecolors=np.where(df.cp, RECORD, REF),
                   linewidths=0.4, zorder=3)
    else:
        ax.scatter(dd(df.MVT_TIME_UTC_mvt), yv0 + jit, s=6, c=RECORD_TINT, edgecolors=RECORD, linewidths=0.3, zorder=2)
strip(arr, ya, 0); strip(dep[~jan], yd, 1); strip(dep[jan], yd, 2, fill=False)
j0 = (ts('2026-01-01') - x0).total_seconds() / 86400; j1 = (ts('2026-02-01') - x0).total_seconds() / 86400
ax.axvspan(j0 - 1.2, j1, color=SHADE, zorder=0)
for xa in (j0 + 6, (j0 + j1) / 2, j1 - 6):
    ax.annotate('', xy=(xa, yd + 0.26), xytext=(xa, ya - 0.26), arrowprops=dict(arrowstyle='-|>', color=RECORD, lw=1.0, mutation_scale=7))
ax.text(j0 + 0.5, -0.42, 'scored month: arrivals\nlabelled, departures\nblanked; the rule moves\nthem to the copy', ha='left', va='top',
        fontsize=5.8, color=INK, linespacing=1.05)
for m, lab in [('2025-09-01', 'Sep 2025'), ('2025-10-01', 'Oct'), ('2025-11-01', 'Nov'), ('2025-12-01', 'Dec'), ('2026-01-01', 'Jan 2026')]:
    xm = (ts(m) - x0).total_seconds() / 86400; ax.axvline(xm, color=CONTEXT, lw=0.5, zorder=0)
    ax.text(xm + 1, 1.45, lab, fontsize=6.2, color=INK, va='bottom')
ax.set_yticks([ya, yd]); ax.set_yticklabels(['THY early arrivals\nat Fiumicino', 'THY departures\nat Fiumicino'], fontsize=6.3)
ax.set_ylim(-1.35, 2.55); ax.set_xlim(-2, j1 + 1); ax.set_xticks([])
for sp in ('left', 'bottom'): ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.scatter([], [], s=10, c=RECORD, label='copied the schedule'); ax.scatter([], [], s=10, c='white', edgecolors=REF, linewidths=0.5, label='a real time')
ax.scatter([], [], s=10, c=RECORD_TINT, edgecolors=RECORD, linewidths=0.4, label='blanked; predicted as a copy')
ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.06), fontsize=6.0, frameon=False, ncol=3, handletextpad=0.2, columnspacing=1.0)
ax.text(-2, 2.05, 'a carrier\'s feed, read from its arrivals; with the flight-plan join (a plan attaches only within 3,606 s of the '
        'recorded off-block,\nat every airport) and a classifier for Fiumicino\'s coin-flip rows, these rules take 3.53 s', fontsize=6.0,
        color=INK, va='bottom')

# bottom band: what each part bought on the 2026 set, left to right
b, h = bands[4]
steps = [('base model', 254.5228, None, STEP[0]), ('trajectory stage', 226.5352, 254.5228, STEP[1]), ('more seeds', 226.3833, 226.5352, STEP[2]),
         ('feed rules', 225.9576, 226.3833, STEP[3]), ('join rules, classifier', 222.8507, 225.9576, STEP[4])]
xs5 = np.linspace(L + 0.07, R0 - 0.085, len(steps))
for k, ((lab, v1, v0, col), xc) in enumerate(zip(steps, xs5)):
    txt = f'{lab}\n{v1:.2f}' if v0 is None else f'{lab}\n{v1:.2f}  ($-${v0 - v1:.2f})'
    fig.text(xc, b + h * 0.70, txt, ha='center', va='center', fontsize=6.6, color=INK, linespacing=1.15,
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=col, lw=1.1))
    if k: fig.patches.append(FancyArrowPatch((xs5[k - 1] + 0.062, b + h * 0.70), (xc - 0.062, b + h * 0.70), transform=fig.transFigure,
                                             arrowstyle='-|>', mutation_scale=7, color=REF, lw=0.8))
ax = clear(fig.add_axes([L + 0.16, b + 0.016, R0 - L - 0.19, h * 0.16]))
for lab, v1, v0, col in steps[1:]:
    ax.add_patch(Rectangle((v1, 0), v0 - v1, 1, color=col, lw=0))
ax.set_xlim(258, 221); ax.set_ylim(0, 1); ax.set_yticks([])
for sp in ('left', 'right', 'top'): ax.spines[sp].set_visible(False)
ax.set_xticks([255, 250, 245, 240, 235, 230, 225]); ax.tick_params(axis='x', labelsize=6.0, length=2, pad=1)
ax.text(-0.015, 0.5, 'RMSE on the\n2026 set (s)', transform=ax.transAxes, ha='right', va='center', fontsize=6.2, color=INK, linespacing=1.05)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(OUT, f'model-architecture.{ext}'))
print('wrote analysis/figures/model-architecture.{png,pdf,svg}')
