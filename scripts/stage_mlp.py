"""s539: a PyTorch MLP stage on the exported ADS-B stage inputs (stage_mlp_export.py): residual target, inputs quantile-
clipped and standardised with missing indicators, airport embedding, Huber-free MSE, 5 seeds averaged, Jan|Jul cross-fit
(fit one month, predict the other) and a whole-window fit for the scored side. -> cand/stage_mlp.npz
  uv run --with numpy --with torch python scripts/stage_mlp.py
"""
import os, numpy as np, torch, torch.nn as nn
W = os.environ.get('DC26_CAND', os.path.join(os.environ['DC26_WORK'], 'adsblol/cand')); Z = np.load(f'{W}/stage_inputs.npz')
torch.set_num_threads(8); dev = os.environ.get('STAGE_MLP_DEV', 'cpu')   # CPU + deterministic algorithms: byte-identical reruns (MPS is not)
torch.use_deterministic_algorithms(True)
def prep(A, lo, hi, mu, sd):
    m = np.isnan(A).astype(np.float32); B = np.clip(np.nan_to_num(A, nan=0.0), lo, hi); return np.concatenate([(B - mu) / sd, m], 1).astype(np.float32)
class Net(nn.Module):
    def __init__(s, d, na):
        super().__init__(); s.e = nn.Embedding(na, 8)
        s.f = nn.Sequential(nn.Linear(d + 8, 256), nn.SiLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(s, x, a): return s.f(torch.cat([x, s.e(a)], 1)).squeeze(1)
def train_predict(fm, targets):
    A = Z['A']; lo, hi = np.nanquantile(A[fm], 0.005, 0), np.nanquantile(A[fm], 0.995, 0)
    B0 = np.clip(np.nan_to_num(A[fm], nan=0.0), lo, hi); mu, sd = B0.mean(0), B0.std(0) + 1e-6
    Xf = torch.tensor(prep(A[fm], lo, hi, mu, sd)); af = torch.tensor(Z['apt'][fm]); tf = torch.tensor((Z['t'][fm] / 300.0).astype(np.float32))
    outs = []
    for seed in range(5):
        torch.manual_seed(seed); net = Net(Xf.shape[1], 16).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        n = len(tf); steps = 0
        for ep in range(12):
            perm = torch.randperm(n)
            for i in range(0, n, 1024):
                b = perm[i:i + 1024]; opt.zero_grad()
                loss = ((net(Xf[b].to(dev), af[b].to(dev)) - tf[b].to(dev)) ** 2).mean(); loss.backward(); opt.step()
        net.eval(); preds = []
        with torch.no_grad():
            for Ax, ax in targets:
                Xt = torch.tensor(prep(Ax, lo, hi, mu, sd)); at = torch.tensor(ax)
                preds.append(np.concatenate([net(Xt[i:i + 8192].to(dev), at[i:i + 8192].to(dev)).cpu().numpy() for i in range(0, len(at), 8192)]) * 300.0)
        outs.append(preds)
    return [np.mean([o[k] for o in outs], 0) for k in range(len(targets))]
fit, app, J, L = Z['fit'], Z['app'], Z['J'], Z['L']; A, apt = Z['A'], Z['apt']
dM = np.zeros(len(fit))
for fm, am in ((fit & L, app & J), (fit & J, app & L)):
    dM[am] = train_predict(fm, [(A[am], apt[am])])[0]
dMr = np.zeros(len(Z['appr'])); ar = Z['appr']
dMr[ar] = train_predict(fit, [(Z['Ar'][ar], Z['aptr'][ar])])[0]
np.savez(os.environ.get('STAGE_MLP_OUT', f'{W}/stage_mlp.npz'), dM=dM, dMr=dMr)
print('MLP stage written; window rows', int(app.sum()), 'scored rows', int(ar.sum()), 'on', dev)
