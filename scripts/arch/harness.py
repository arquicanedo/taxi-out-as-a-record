"""Shared harness for the alternative-architecture programme (s230).

Every candidate trains on the SAME matrices the shipped MLP uses (gate_stage_a2.npz), so a
difference in result is a difference in architecture and nothing else. The shipped MLP is the
control and must reproduce 233.734.

The programme's brief, from s230's spec sheet: the payoff is bias INDEPENDENCE, not solo
accuracy. A model 20 s worse solo still pays +0.85 component-s if its bias is independent, and
`LGB` is the redundant component to be independent OF (slack +0.0314 against a floor of 0.9223,
where MLP x perAirport sits AT its floor at +0.0001).
"""
import os, math, time, warnings; warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn as nn
SP = os.environ['DC26_WORK']; THR = 7200
DATA = os.environ.get('DC26_DATA', os.environ['DC26_DATA'])

def _pick_device():
    """cuda > mps > cpu, overridable with DC26_DEV. Ported 2026-09-12 for an RTX 4000 Ada box."""
    d = os.environ.get('DC26_DEV')
    if d: return torch.device(d)
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')
DEV = _pick_device()

# DELL-PERF.md 3(b) said to turn TF32 on. IT WAS TRIED AND IT IS WRONG, AND IT DEFAULTS OFF.
#
# Measured on the Dell 2026-09-13, K=64, 3 seeds, everything else identical including the fp16
# window, B=8192 on both (which is also what Metal's batch_for returns, so this is the batch the
# recorded number used):
#
#     TF32 on    seeds 233.273  276.005  255.613   ->  solo 239.513  corr 0.9079
#     TF32 off   seeds 232.339  233.225  237.168   ->  solo 230.705  corr 0.9465
#     recorded                                         solo 230.289  corr 0.9486
#
# TF32 off reproduces the record to 0.42 s. TF32 on is 9.2 s adrift and its seed spread is ~21 s
# against a recorded 3.73 s. It gets worse with K, because the attention reduction is a sum over
# K terms: K=32 was unharmed, K=64 wobbled, and K=128 diverged outright to 1243.676 and 607.467
# on seeds 1 and 2. TF32's 10-bit mantissa against fp32's 23 is not enough for this reduction.
#
# It costs about 8 to 10% of throughput to leave it off (K=64 seed 480-509 s on, 529-559 s off),
# which is noise next to the 27x that batch sizing is worth. Do not turn this back on without
# a 3-seed A/B at K >= 64; a single seed will not show it, and that is exactly how it got in.
TF32 = os.environ.get('TF32', '0') == '1'
if DEV.type == 'cuda' and TF32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

# DELL-PERF.md 3(c). W's six per-neighbour columns are three exact 0/1 indicators (arr, has-taxi,
# same-runway) and three values clipped to +-8, +-8 and +-6 hours, where fp16 carries about three
# decimal digits. Storing fp16 halves the window: the K=256 train window goes 9.96 -> 4.98 GiB,
# which fits resident on a 12 GB card and retires the streaming path entirely. Applied at LOAD as
# well as at save, so the existing fp32 K=128 cache gets the benefit without a 176 s rebuild.
#
# GATED 2026-09-13 alongside TF32, and for the same reason: it is a suspect in the K=64/K=128
# divergence. Set WDT=32 to store and use the window in fp32 and A/B it against TF32=0.
#
# CUDA-ONLY, corrected 2026-09-13. This was global for most of a night, which meant the Mac would
# have silently started training on fp16 windows the next time it ran d2_window_wide.py. fp16 is
# here to fit a 12 GB card; Apple unified memory is 53 GB and does not need it, and "free" was
# measured on CUDA at K=64, not on Metal at any K. A cross-machine repo should not change the
# other machine's numerics as a side effect of a memory fix on this one.
# WDT=16 forces fp16 anywhere, WDT=32 forces fp32 anywhere.
_wdt = os.environ.get('WDT')
if _wdt is None:
    WDTYPE = np.float16 if DEV.type == 'cuda' else np.float32
else:
    WDTYPE = np.float16 if _wdt == '16' else np.float32

def as_wdtype(a):
    """Cast a window array to the storage dtype. The model casts back at point of use."""
    return np.asarray(a, dtype=WDTYPE)

# Activation cost per training row, GB, fitted on the Dell 2026-09-13 by
# scripts/arch/bench_window_mem.py. At fixed K the measured peak is LINEAR in batch to four
# figures (K=128: 5.84 / 7.47 / 10.74 / 17.27 / 30.34 GB at B=1024..16384, per-row deltas
# 1.592e-3 1.596e-3 1.594e-3 1.5955e-3), so peak = FIXED + slope(K)*B. Solving slope at K=64 and
# K=128 splits it into a term linear in K (the per-neighbour item MLP) and one quadratic in K
# (the attention matrix):
#     slope(K) = 7.059e-6 * K + 4.214e-8 * K^2      GB per row
# VALIDATED OUT OF SAMPLE before being used, per CLAUDE.md's estimator rule: it was fitted on
# K=64 and K=128 and then PREDICTED K=256 B=1024 at 11.4 GB against a measured 11.29 GB, 1% out.
_ROW_LIN, _ROW_QUAD = 7.059e-6, 4.214e-8
_BATCH_CACHE = {}

# WHY batch_for IS MEMOISED AND WHY IT CALLS empty_cache FIRST. Added 2026-09-13 after this
# function, as first written the same day, silently destroyed three experiments.
#
# Every caller invokes it INSIDE the per-seed loop. It read LIVE free VRAM, and after seed 0 the
# caching allocator still holds that seed's pool, so seed 1 measured a nearly full card and
# returned a fraction of the batch. Measured against real pressure:
#
#     free GiB   batch_for(32)   batch_for(128)
#        9.97           16384             4096
#        4.00            8192             1024
#        2.00            4096              512
#        0.40            1024              256
#
# During the D3 run the card sat at ~0.3 GiB free, so seeds 1 and 2 trained at 1024-2048 against
# seed 0's 16384. The signature is unmistakable in hindsight and I read it three times as
# something else: seed 0 clean, every later seed diverged, and later seeds ran ~1.8x LONGER
# because a smaller batch means more steps.
#
#     D3 K=32 contig   234.087  1418.495  3605.737
#     D3 K=32 log      235.027   867.152   882.519
#     D3 K=32 hybrid   234.086  2140.444  1151.156
#
# It also retracts the K=128 diagnosis in DELL-RESULTS. "K=128 needs a card that holds 4096" was
# wrong: batch_for(128) returns 2048 at 8 GiB free and 512 at 2 GiB, so K=128's seeds 1 and 2
# were collapsing for this reason, not because 12 GB cannot hold the recorded schedule.
#
# Memoising makes the batch a property of (K, base) rather than of allocator state, so it is
# constant across seeds and reproducible across runs. empty_cache() before measuring stops the
# retained pool from being counted as used on the FIRST call too. A fixed batch is worth more
# than an optimal one: a batch that varies per seed is not an experiment.

def row_gb(K):
    """Activation GB per training row at this K. See the fit above."""
    return _ROW_LIN*K + _ROW_QUAD*K*K

def batch_for(K, base=16384):
    """MPSGraph caps tensor dims at INT_MAX and the attention matrix is batch x K x K, so on
    Metal the batch must shrink as K grows -- K=128 crashed twice before this was found.

    REWRITTEN 2026-09-13 ON MEASUREMENT. Everything below the 'CORRECTED 2026-09-12' paragraph
    is kept because it is the reasoning this replaces, and it was wrong in a way worth recording:
    it scaled B from a K=128 anchor chosen to 'reproduce the benchmark exactly'. The benchmark it
    reproduced had never fitted in VRAM. On Windows WDDM the driver backs an oversubscribed
    allocation with system RAM over PCIe instead of raising OOM, so the job pages instead of
    crashing and nvidia-smi, which sees only the physical card, pins at ~11.9 GB and looks fine.
    Measured on the Dell at K=128:

        B=16384   peak 30.34 GB on an 11.99 GB card    806 rows/s    epoch 2159 s
        B= 2048   peak  7.47 GB, fits                21,989 rows/s   epoch   79 s

    27x, from batch size alone. The 36:03 that got the GPU written off as slower than an M5 Pro
    was one epoch of fourteen at 2159 s, not a seed. At B=2048 a full seed is 1144 s against the
    Mac's 2085 s, so the box is 1.82x FASTER, and its solo reproduces the control (233.053 against
    232.386, inside the 3.73 s seed spread).

    So: solve the measured model for the largest batch whose activations fit in free VRAM, with a
    headroom factor, rather than scaling from any anchor. Call this AFTER park(), so that
    mem_get_info already reflects whatever the window tensors took. BS/EVB still override.

    NOTE THIS CHANGES NUMERICS against every recorded CUDA run, because the batch changes. It
    cannot change Metal, and it cannot make anything worse than a configuration that paged.

    ---- superseded reasoning, kept so the error is legible ----

    CORRECTED 2026-09-12. This returned a flat `base` on CUDA, on the reasoning that CUDA has no
    INT_MAX cap. It has no cap and it does have 12 GB. The window path's activations are
    B x K x dh per layer, so they scale with K on any backend, and measured on the Dell during
    the K=128 Job 1 benchmark the card sat at 11,878 of 12,282 MiB (96.7%) with only 5.73 GB of
    resident tensors. The other 6.1 GB is activation workspace, at the K the brief treats as
    safe. So CUDA scales too, just from a higher anchor (K=128) than Metal's (K=32), which keeps
    K=128 at 16384 and reproduces the benchmark exactly while halving K=256.

    THIS DOES NOT BOUND THE DOMINANT ACTIVATION, added 2026-09-13 from analysis/DELL-PERF.md.
    4.00 of those 6.1 GB is ONE tensor: `nn.MultiheadAttention` with a key_padding_mask takes
    the math path in training mode and materialises B x heads x K x K in fp32, kept for
    backward. Returning `base*128//K` holds `B*K` constant, so that term is B*heads*K*K
    = LINEAR IN K under this very formula:

        K=128  B=16384   16384*4*128*128*4 = 4.00 GiB
        K=256  B= 8192    8192*4*256*256*4 = 8.00 GiB   <- doubles, it does not halve

    Holding the attention matrix constant needs B proportional to 1/K^2 (B=4096 at K=256), and
    that is not what this returns. Do not read a successful K=128 run as clearance for K=256.
    The real fix is F.scaled_dot_product_attention, which tiles and never materialises K x K;
    DELL-PERF.md section 3(a) has the patch and gates it on reproducing solo 232.386 at K=128.
    Until that lands, pass BS/EVB explicitly at K=256 rather than trusting this.

    RIGHT ON MECHANISM, 6x LOW ON MAGNITUDE, measured 2026-09-13. Peak alloc is 12.95 GB at
    K=64 and 31.16 GB at K=128, both at B=16384. Fitting peak = A*K^2 + C gives A*K^2 = 24.28 GB
    of attention term at K=128, not 4.00 GiB: roughly six K x K-sized tensors are live at once
    across the two layers, not one. So the quadratic diagnosis was right and every absolute
    number derived from it was low by about 6x. Separately, need_weights=False was tried and
    does NOT rescue it: it cuts peak from 30.34 to 21.48 GB at B=16384, still far over the card,
    and buys nothing once the batch is small enough to fit. SDPA remains the only route that
    removes the matrix, and it is no longer on the critical path now that batch size is."""
    if DEV.type == 'cuda':
        # MEMOISED, and measured after empty_cache(). Both parts are load-bearing; see below.
        hit = _BATCH_CACHE.get((K, base))
        if hit is not None:
            return hit
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info()
        budget = float(os.environ.get('VRAM_FRAC', '0.75')) * free / 2**30
        b = int(budget / row_gb(K))
        if b < 256:                       # nothing sane fits; let the caller see it fail loudly
            b = 256
        else:
            b = 1 << int(math.floor(math.log2(b)))      # power of two, so step counts stay tidy
        b = int(min(base, b))
        _BATCH_CACHE[(K, base)] = b
        return b
    return max(1024, base*32//max(K,32))

def vram():
    """(free, total) bytes, or None off CUDA."""
    return torch.cuda.mem_get_info() if DEV.type == 'cuda' else None

def park(t, reserve_gb=4.5):
    """Put a tensor on the device if it fits with `reserve_gb` left over for activations,
    otherwise pin it in host RAM for streaming. `take` then handles either placement.

    to_dev parks everything resident, which is right on 53 GB of Apple unified memory and wrong
    on a 12 GB card: at K=256 the train window alone is 1,740,323 x 256 x 6 x 4 B = 9.96 GiB
    (10.69 GB decimal) plus a 1.66 GiB mask, and cannot coexist with its own activations.
    Host-side gather plus an async copy costs ~7 s per seed over PCIe (DELL-BRIEF), which is
    under 0.5% of a K=128 seed.

    Streaming may be avoidable. DELL-PERF.md 3(c): three of W's six columns are exact 0/1
    indicators and the other three are clipped to +-8, +-8 and +-6, where fp16 carries about
    three decimal digits. An fp16 cache puts the K=256 train window at 4.98 GiB, which fits
    resident on a 12 GB card and retires this whole path. Not done, and it is the cheaper of
    the two fixes."""
    t = torch.as_tensor(t)
    if DEV.type != 'cuda': return t.to(DEV)
    free, _ = torch.cuda.mem_get_info()
    if t.element_size()*t.nelement() < free - reserve_gb*2**30: return t.to(DEV)
    return t.pin_memory()

def take(t, idx):
    """Index `t` by a device index tensor, whichever side of the PCIe bus `t` is parked on.

    idx stays on the device so the RNG stream is untouched: torch.randperm(N, device=DEV) must
    keep producing the same permutation it did before streaming existed, or a streamed run is
    not comparable with a resident one."""
    if t.device.type != 'cpu': return t[idx]
    return t[idx.to('cpu')].to(DEV, non_blocking=True)

def chunk(t, i, n):
    """Contiguous slice t[i:i+n], moved to the device if t is parked on the host. The eval path
    slices rather than gathers, so it needs its own accessor."""
    s = t[i:i+n]
    return s if s.device.type != 'cpu' else s.to(DEV, non_blocking=True)

def mem_report(tag=''):
    """Peak allocation actually measured, not an estimate. Call after the first epoch."""
    if DEV.type != 'cuda': return
    free, total = torch.cuda.mem_get_info()
    print(f'  vram{tag}: peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f} GB, '
          f'reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GB, '
          f'free {free/2**30:.2f} of {total/2**30:.2f} GB', flush=True)

def load():
    z = np.load(f'{SP}/gate_stage_a2.npz')
    zm = np.load(f'{SP}/gate_mlp.npz')
    d = dict(Xt=z['Xt'], Xv=z['Xv'], CT=z['CT'], CV=z['CV'],
             cards=[int(k) for k in z['cards']],
             yt=z['yt'], yv=z['yv'], bt=z['base_tr'], bv=z['base_va'],
             sd_t=z['sd_t'], sd_v=z['sd_v'], lb_t=z['lb_t'], lb_v=z['lb_v'],
             lgb=z['lgb_va'], pa=z['pa_va'], mlp=zm['mlp_va'])
    d['blend'] = 0.2*d['lgb'] + 0.4*d['mlp'] + 0.4*d['pa']
    # controls: the recorded component levels must reproduce before anything is compared to them
    ORD = d['yv'] < THR
    # recorded values are CLIPPED to [60, 200000], the house convention (s216). The UNCLIPPED
    # lgb is 232.930 and mixing the two is the s277 trap.
    for nm, rec in (('lgb',232.916), ('mlp',233.734), ('pa',236.380), ('blend',229.622)):
        got = float(np.sqrt(np.mean((d[nm][ORD]-d['yv'][ORD])**2)))
        assert abs(got-rec) < 1.0, f'{nm} reproduces {got:.3f}, record says {rec}'
    print(f'controls OK: lgb/mlp/perapt/blend reproduce to <1.0 s on {int(ORD.sum()):,} ordinary rows')
    return d

class Trunk(nn.Module):
    """The shipped MLP trunk, unchanged, so only the HEAD differs between candidates."""
    def __init__(s, nf, cards, h=768, out=1):
        super().__init__()
        s.emb = nn.ModuleList([nn.Embedding(int(k), min(24,(int(k)+1)//2)) for k in cards])
        e = sum(min(24,(int(k)+1)//2) for k in cards)
        s.body = nn.Sequential(
            nn.Linear(nf+e,h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(.15),
            nn.Linear(h,h//2), nn.GELU(), nn.BatchNorm1d(h//2), nn.Dropout(.15),
            nn.Linear(h//2,h//4), nn.GELU(), nn.BatchNorm1d(h//4))
        s.head = nn.Linear(h//4, out)
    def forward(s, x, c):
        return s.head(s.body(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(s.emb)],1)))

def to_dev(*ts):
    """Park tensors on the GPU ONCE. The training matrix is ~0.8 GB against 53 GB of unified
    memory, so per-batch host->device copies are pure waste (they dominated the shipped
    scripts, which index on CPU and .to(DEV) every batch)."""
    return [t.to(DEV) for t in ts]

def train(model, Xt, Ct, targets, loss_fn, epochs=14, bs=16384, seed=0, lr=2e-3, maxlr=3e-3):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    N = len(Xt); steps = epochs*(N//bs+1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=maxlr, total_steps=steps)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(N, device=DEV)
        for i in range(0, N, bs):
            j = perm[i:i+bs]
            if len(j) < 2: continue
            opt.zero_grad()
            loss_fn(model(Xt[j], Ct[j]), [t[j] for t in targets]).backward()
            opt.step(); sch.step()
    return model

@torch.no_grad()
def infer(model, Xv, Cv, bs=131072):
    model.eval()
    return torch.cat([model(Xv[i:i+bs], Cv[i:i+bs]).float().cpu()
                      for i in range(0, len(Xv), bs)])

def report(name, pred_va, d, s_irr=225.35):
    """Solo RMSE, independence against each component, and the optimal 2-way blend."""
    ORD = d['yv'] < THR; y = d['yv'][ORD]; p = np.asarray(pred_va)[ORD]
    e = p - y; sb_e = d['blend'][ORD] - y
    R = float(np.sqrt(np.mean(e**2))); sb = float(np.sqrt(np.mean(sb_e**2)))
    rho = float(np.corrcoef(e, sb_e)[0,1])
    floor = s_irr*s_irr/(sb*R) if sb*R > 0 else np.nan
    num = sb*sb - rho*sb*R; den = sb*sb + R*R - 2*rho*sb*R
    w = max(0.0, min(1.0, num/den)) if den > 0 else 0.0
    v = (1-w)**2*sb*sb + w*w*R*R + 2*w*(1-w)*rho*sb*R
    nb = float(np.sqrt(v))
    N = 344841; EXT = 5.9122e9
    board = float(np.sqrt((N*nb*nb+EXT)/N))
    print(f'\n=== {name} ===')
    print(f'  solo RMSE (ordinary rows)   {R:8.3f}   vs blend {sb:.3f}, lgb {np.sqrt(np.mean((d["lgb"][ORD]-y)**2)):.3f}')
    for k in ('lgb','mlp','pa'):
        print(f'  residual corr vs {k:<10} {np.corrcoef(e, d[k][ORD]-y)[0,1]:8.4f}')
    print(f'  residual corr vs BLEND      {rho:8.4f}   (floor {floor:.4f}, slack {rho-floor:+.4f})')
    print(f'  optimal 2-way blend         {nb:8.3f}   gain {sb-nb:+.3f} component-s at weight {w:.2f}')
    print(f'  implied board               {board:8.4f}   (v59 is 264.3314)')
    return dict(name=name, solo=R, rho=rho, floor=floor, blend=nb, gain=sb-nb, w=w, board=board)
