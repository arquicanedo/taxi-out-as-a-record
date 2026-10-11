"""v15 stage 2: the MLP, rebuilt with block_dev in F_BULK (s77).

Identical to v8s_net.py except for the input paths. The blend gain from
block_dev was only -0.56 against -1.07 on the LightGBM alone, because the net
did not carry the feature; this is the run that closes that half.

Separate process from v15_trees.py: LightGBM and PyTorch segfault together
(OpenMP clash, exit 139).
"""
import pandas as pd, numpy as np, json, time, torch, torch.nn as nn, warnings; warnings.filterwarnings('ignore')
import features
tr=pd.read_parquet(features.wk('v15_tr')); rk=pd.read_parquet(features.wk('v15_rk'))
c=json.load(open(features.wk('v15_cols','json'))); F,FB,CATS=c['F'],c['F_BULK'],c['CATS']; NUMF=[x for x in FB if x not in CATS]
ext=(tr.y>=7200).values; trn=tr[~ext]
nt=trn.naive.values.astype(float); nr=rk.naive.values.astype(float)
bt=np.where(~np.isnan(nt),nt,trn.apt_hr_naive.values.astype(float)); br=np.where(~np.isnan(nr),nr,rk.apt_hr_naive.values.astype(float))
g=np.nanmedian(bt); bt=np.nan_to_num(bt,nan=g); br=np.nan_to_num(br,nan=g)
cards=[];CT=[];CR=[]
for x in CATS:
    u=pd.Index(trn[x].astype(str).unique()); cards.append(len(u)+1)
    CT.append(pd.Index(u).get_indexer(trn[x].astype(str))+1); CR.append(pd.Index(u).get_indexer(rk[x].astype(str))+1)
CT=np.clip(np.vstack(CT).T,0,None); CR=np.clip(np.vstack(CR).T,0,None)
Xt=trn[NUMF].to_numpy(np.float64); Xr=rk[NUMF].to_numpy(np.float64)
med=np.nanmedian(Xt,0); Mt=np.isnan(Xt).astype(np.float32); Mr=np.isnan(Xr).astype(np.float32)
Xt=np.where(np.isnan(Xt),med,Xt); Xr=np.where(np.isnan(Xr),med,Xr)
q1,q9=np.percentile(Xt,1,axis=0),np.percentile(Xt,99,axis=0); sc=np.where(q9-q1<1e-9,1,q9-q1)
Xt=np.hstack([np.clip((Xt-q1)/sc,-5,5),Mt]).astype(np.float32); Xr=np.hstack([np.clip((Xr-q1)/sc,-5,5),Mr]).astype(np.float32)
rt=(trn.y.values-bt).astype(np.float32); rmn,rs=rt.mean(),rt.std()
dev=torch.device('mps')
class Net(nn.Module):
    def __init__(s,nf,cards,h=768):
        super().__init__(); s.emb=nn.ModuleList([nn.Embedding(k,min(24,(k+1)//2)) for k in cards])
        e=sum(min(24,(k+1)//2) for k in cards)
        s.f=nn.Sequential(nn.Linear(nf+e,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(.15),
            nn.Linear(h,h//2),nn.GELU(),nn.BatchNorm1d(h//2),nn.Dropout(.15),
            nn.Linear(h//2,h//4),nn.GELU(),nn.BatchNorm1d(h//4),nn.Linear(h//4,1))
    def forward(s,x,c): return s.f(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(s.emb)],1)).squeeze(1)
Xt_t=torch.tensor(Xt);Ct_t=torch.tensor(CT,dtype=torch.long);rt_t=torch.tensor((rt-rmn)/rs)
Xr_t=torch.tensor(Xr).to(dev);Cr_t=torch.tensor(CR,dtype=torch.long).to(dev)
N=len(Xt_t);B=8192;EP=14;ps=[];t0=time.time()
for seed in range(8):
    torch.manual_seed(seed); net=Net(Xt.shape[1],cards).to(dev)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=EP*(N//B+1))
    for ep in range(EP):
        net.train(); perm=torch.randperm(N)
        for i in range(0,N,B):
            j=perm[i:i+B]; opt.zero_grad()
            nn.functional.mse_loss(net(Xt_t[j].to(dev),Ct_t[j].to(dev)),rt_t[j].to(dev)).backward(); opt.step(); sch.step()
    net.eval()
    with torch.no_grad():
        r=torch.cat([net(Xr_t[i:i+65536],Cr_t[i:i+65536]) for i in range(0,len(Xr_t),65536)]).cpu().numpy()*rs+rmn
    ps.append(br+r); print(f'  net seed {seed} [{time.time()-t0:.0f}s]',flush=True)
pd.DataFrame({'MVT_ID_mvt':rk.MVT_ID_mvt.values,'mlp':np.mean(ps,0)}).to_parquet(features.wk('v15_net'),index=False)
print('saved net',flush=True)
