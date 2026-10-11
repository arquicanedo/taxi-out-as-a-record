"""Stage A2b (MPS, torch only): the MLP, the 0.4 component of the shipped bulk,
cross-fitted the same 2 folds as gate_a2.py so the gate never sees a leaked fit.

Matches v15_net.py: residual target y - naive_baseline, trained on non-extreme
rows only, h=768, 14 epochs, OneCycle, 8 seeds. Separate process from gate_a2
because LightGBM and PyTorch segfault together (CLAUDE.md).

Seeds: 8 on the validation side (matching shipped), 2 per training fold.
"""
import numpy as np, torch, torch.nn as nn, time, os, warnings; warnings.filterwarnings('ignore')
SP = os.environ["DC26_WORK"]
z = np.load(f'{SP}/gate_stage_a2.npz')
THR = 7200; OUT = f'{SP}/gate_mlp.npz'
Xt, Xv, CT, CV, cards = z['Xt'], z['Xv'], z['CT'], z['CV'], list(z['cards'])
yt, yv, bt_, bv_ = z['yt'], z['yv'], z['base_tr'], z['base_va']
in_fold1 = z['folds'][0]                       # True => row is in fold 1's holdout
nx = yt < THR
dev = torch.device(os.environ.get('DC26_DEV') or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')); T0 = time.time()

class Net(nn.Module):
    def __init__(s, nf, cards, h=768):
        super().__init__(); s.emb = nn.ModuleList([nn.Embedding(int(k), min(24, (int(k)+1)//2)) for k in cards])
        e = sum(min(24, (int(k)+1)//2) for k in cards)
        s.f = nn.Sequential(nn.Linear(nf+e, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(.15),
            nn.Linear(h, h//2), nn.GELU(), nn.BatchNorm1d(h//2), nn.Dropout(.15),
            nn.Linear(h//2, h//4), nn.GELU(), nn.BatchNorm1d(h//4), nn.Linear(h//4, 1))
    def forward(s, x, c): return s.f(torch.cat([x]+[e(c[:, i]) for i, e in enumerate(s.emb)], 1)).squeeze(1)

def run(fit_idx, pred_X, pred_C, seeds):
    """Fit on fit_idx (non-extreme rows only), return the mean residual prediction."""
    rt = (yt[fit_idx] - bt_[fit_idx]).astype(np.float32); rmn, rs = rt.mean(), rt.std()
    Xf = torch.tensor(Xt[fit_idx]); Cf = torch.tensor(CT[fit_idx], dtype=torch.long)
    rf = torch.tensor((rt - rmn) / rs)
    Xp = torch.tensor(pred_X).to(dev); Cp = torch.tensor(pred_C, dtype=torch.long).to(dev)
    N = len(Xf); B = 8192; EP = 14; ps = []
    for seed in seeds:
        torch.manual_seed(seed); net = Net(Xt.shape[1], cards).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EP*(N//B+1))
        for ep in range(EP):
            net.train(); perm = torch.randperm(N)
            for i in range(0, N, B):
                j = perm[i:i+B]; opt.zero_grad()
                nn.functional.mse_loss(net(Xf[j].to(dev), Cf[j].to(dev)), rf[j].to(dev)).backward()
                opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            r = torch.cat([net(Xp[i:i+65536], Cp[i:i+65536]) for i in range(0, len(Xp), 65536)]).cpu().numpy()
        ps.append(r*rs + rmn); print(f'    seed {seed} [{time.time()-T0:.0f}s]', flush=True)
    return np.mean(ps, 0)

mlp_tr = np.zeros(len(yt))
for k, hold in enumerate([in_fold1, ~in_fold1]):
    fit = np.where(~hold & nx)[0]; pred = np.where(hold)[0]
    print(f'  fold {k}: fit {len(fit):,} predict {len(pred):,}', flush=True)
    mlp_tr[pred] = bt_[pred] + run(fit, Xt[pred], CT[pred], range(2))
print(f'  full fit for validation', flush=True)
mlp_va = bv_ + run(np.where(nx)[0], Xv, CV, range(8))
nxv = yv < THR
print(f'MLP  val bulk {np.sqrt(np.mean((np.clip(mlp_va,60,200000)[nxv]-yv[nxv])**2)):7.2f}  [{time.time()-T0:.0f}s]', flush=True)
np.savez_compressed(OUT, mlp_tr=mlp_tr, mlp_va=mlp_va)
print(f'saved {OUT}', flush=True)
