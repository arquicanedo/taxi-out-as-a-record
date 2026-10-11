"""D3: MULTISCALE CONTEXT SAMPLING. Does the window need to be CONTIGUOUS?

K conflates four things -- how much information the window carries, how much elapsed time it
spans, the signal-to-noise ratio inside it, and cost. The K = 32 -> 64 -> 128 series grows all
four together, so it shows the bundle helps without identifying which part does. Idea from an
external consult, 2026-09-12; nowhere in the record before that.

This holds the TOKEN BUDGET FIXED and changes only WHICH neighbours fill it:

    SEL=contig   +-1..+-K/2                  the current window. CONTROL.
    SEL=log      log-spaced out to +-657     same K, 41x the horizon
    SEL=hybrid   half dense, half log-spaced

(The two log figures above read "+-~256" and "~16x" until 2026-09-13. The ladder actually gives
+-657 at K=32, which is 41x, and s270's published table already records +-657, so the docstring
was the wrong half. Corrected, finding unaffected.)

Identical architecture, identical K, identical features. Any difference is information
SELECTION, not information QUANTITY. Run at K=32 where a seed is 7.5 min.

NOT the closed traffic-and-queue result (~1.8 s): that tested aggregate counts as features.
This tests whether the existing window spends its capacity on redundant tokens.
Architecture, features and blend arithmetic are D2's, unchanged, so that the only difference
from the recorded K=32 D2 run is WHICH neighbours fill the window. (A second, verbatim copy of
D2's docstring sat here as a no-op string expression and described this file as "D2: the
movement-window transformer, widened". Removed 2026-09-13; it changed nothing and named the
wrong experiment.)
"""
import sys, os, time, glob, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn
import harness as H
SP=os.environ['DC26_WORK']; D=os.environ['DC26_DATA']
K=int(os.environ.get('K','32')); NF=6; SEEDS=int(os.environ.get('SEEDS','3'))
EPOCHS=int(os.environ.get('EPOCHS','14'))
# Was os.environ.get(chr(83)+chr(69)+chr(76),"contig"), i.e. 'SEL', here and at the np.save near
# the foot, while line ~55 spelled the same variable as a plain literal. Not load-bearing, and a
# live hazard: grep -rn "'SEL'" found one of the three, so a rename would have half-landed and
# every arm's cache path would have silently fallen back to "contig". Literals, 2026-09-13.
SEL=os.environ.get('SEL','contig')
CACHE=f'{SP}/arch_d3_{SEL}_K{K}.npz'
US=10**6
def build():
    cols=['MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt',
          'TAXITIME_SEC_mvt','RUNWAY_mvt','SCHED_TIME_UTC_mvt']
    raw=pd.concat([pq.read_table(f,columns=cols).to_pandas()
                   for f in sorted(glob.glob(f'{D}/training_2025-*.parquet'))],ignore_index=True)
    raw['t']=raw.MVT_TIME_UTC_mvt.astype('int64')//US
    assert 1.735e9<np.median(raw.t)<1.77e9, 'epoch not 2025'
    raw['apt']=raw.ADEP_mvt.where(raw.PHASE_mvt=='DEP',raw.ADES_mvt).astype(str)
    raw['arr']=(raw.PHASE_mvt=='ARR').astype(np.float32)
    raw['sdn']=(raw.t-raw.SCHED_TIME_UTC_mvt.astype('int64')//US).astype(np.float32)
    raw['rwy']=raw.RUNWAY_mvt.astype(str); raw['taxi']=raw.TAXITIME_SEC_mvt.astype(float)
    tr=pd.read_parquet(f'{SP}/night_tr_all.parquet',columns=['MVT_ID_mvt'])
    va=pd.read_parquet(f'{SP}/night_va_all.parquet',columns=['MVT_ID_mvt'])
    tgt=pd.concat([tr,va],ignore_index=True).MVT_ID_mvt.values
    W=np.zeros((len(tgt),K,NF),np.float32); M=np.zeros((len(tgt),K),np.float32)
    if SEL=='contig':
        off=np.array([o for o in range(-(K//2),K//2+1) if o!=0][:K])
    elif SEL=='log':
        # geometric ladder, mirrored: reaches 41x further at the same token count (+-657 at K=32)
        # geometric, but grow the ratio search until there are exactly K//2 UNIQUE magnitudes
        # (1.5**i dedupes to 14 at K=32, which silently produced a 28-token window)
        mag=[]; r=1.5
        while len(mag)<K//2:
            mag=sorted({int(round(r**i)) for i in range(0,60)})[:K//2]
            if len(mag)<K//2: r-=0.05
            if r<1.05: raise SystemExit('cannot build enough unique magnitudes')
        off=np.array([sg*m for m in mag for sg in (-1,1)][:K])
    elif SEL=='hybrid':
        h=K//4
        dense=[o for o in range(-h,h+1) if o!=0]
        mag=sorted({int(round(1.8**i)) for i in range(0,20) if round(1.8**i)>h})[:K//4]
        far=[s*m for m in mag for s in (-1,1)]
        off=np.array((dense+far)[:K])
    else: raise SystemExit(f'unknown SEL={SEL}')
    assert len(off)==K, f'{SEL} produced {len(off)} offsets, need {K}'
    # BOUNDS GUARD, added 2026-09-13 (dell-01 Q2 / mac-01 A2). The count assert above catches
    # hybrid at K=64, which builds 62 offsets and fails loudly. It does NOT catch log at K=64,
    # which builds exactly 64 and passes while spanning +-431,440 against a smallest airport of
    # 269,598 movements and a largest of 546,851: the far tokens fall out of range, get masked,
    # and the arm silently becomes a mostly-empty window compared against a full one. The whole
    # premise of this experiment is identical token count with different selection, so that is a
    # failure of the premise, not of the code, and it has to fail on the premise.
    nmin=int(raw.groupby('apt',sort=False).size().min())
    span=int(np.abs(off).max())
    assert span < 0.5*nmin, (f'{SEL} at K={K} spans +-{span:,} against a {nmin:,}-row smallest '
                             f'airport; far tokens would mask out and the token budget would not '
                             f'be matched. Reduce K or the ladder ratio.')
    print(f'  SEL={SEL}  offsets span {off.min()} .. {off.max()}  n={len(off)}  '
          f'guard ok ({span:,} < {0.5*nmin:,.0f})',flush=True)
    pos=pd.Series(np.arange(len(tgt)),index=tgt); t0=time.time()
    for apt,g in raw.groupby('apt',sort=False):
        g=g.sort_values('t'); n=len(g)
        tt=g.t.values.astype(np.float64); arr=g.arr.values.astype(np.float32)
        taxi=g.taxi.values; rwy=pd.factorize(g.rwy.values)[0]; sdn=g.sdn.values
        obs=np.where(arr>0,taxi,np.nan)
        r=pos.reindex(g.MVT_ID_mvt.values).values
        keep=np.where(~np.isnan(r))[0]
        if not len(keep): continue
        J=keep[:,None]+off[None,:]; ok=(J>=0)&(J<n); Jc=np.clip(J,0,n-1)
        ot=obs[Jc]
        feat=np.stack([np.clip((tt[Jc]-tt[keep][:,None])/1800.,-8,8), arr[Jc],
                       np.where(np.isnan(ot),0.,np.clip(ot/1800.,0,8)),
                       (~np.isnan(ot)).astype(np.float32),
                       (rwy[Jc]==rwy[keep][:,None]).astype(np.float32),
                       np.clip(sdn[Jc]/3600.,-6,6)],-1).astype(np.float32)*ok[...,None]
        rows=r[keep].astype(int); W[rows]=feat; M[rows]=ok.astype(np.float32)
    np.savez_compressed(CACHE,W=W,M=M,n_tr=len(tr))
    print(f'built K={K}: {W.shape} filled {M.mean():.1%} [{time.time()-t0:.0f}s]')
    return W,M,len(tr)
if os.path.exists(CACHE):
    z=np.load(CACHE); W,M,ntr=z['W'],z['M'],int(z['n_tr']); print(f'loaded {CACHE}')
else: W,M,ntr=build()
d=H.load(); THR=7200
Wt,Mt,Wv,Mv=W[:ntr],M[:ntr],W[ntr:],M[ntr:]
assert len(Wt)==len(d['Xt']) and len(Wv)==len(d['Xv'])
print(f'window filled: train {Mt.mean():.1%}  val {Mv.mean():.1%}')
class WNet2(nn.Module):
    """two attention layers, so neighbour-of-neighbour structure can compose."""
    def __init__(s,nf,cards,nfw,h=384,dh=128,heads=4):
        super().__init__()
        s.emb=nn.ModuleList([nn.Embedding(int(c),min(24,(int(c)+1)//2)) for c in cards])
        e=sum(min(24,(int(c)+1)//2) for c in cards)
        s.stem=nn.Sequential(nn.Linear(nf+e,h),nn.GELU(),nn.BatchNorm1d(h),nn.Dropout(.15))
        s.item=nn.Sequential(nn.Linear(nfw,dh),nn.GELU(),nn.Linear(dh,dh),nn.GELU())
        s.self_att=nn.MultiheadAttention(dh,heads,batch_first=True)
        s.ln=nn.LayerNorm(dh)
        s.q=nn.Linear(h,dh); s.att=nn.MultiheadAttention(dh,heads,batch_first=True)
        s.out=nn.Sequential(nn.Linear(h+dh,256),nn.GELU(),nn.BatchNorm1d(256),
                            nn.Dropout(.10),nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(s,x,c,w,m):
        z=s.stem(torch.cat([x]+[e(c[:,i]) for i,e in enumerate(s.emb)],1))
        it=s.item(w); pad=(m<0.5)
        pad=torch.where(pad.all(1,keepdim=True),torch.zeros_like(pad),pad)
        sa,_=s.self_att(it,it,it,key_padding_mask=pad); it=s.ln(it+sa)
        a,_=s.att(s.q(z).unsqueeze(1),it,it,key_padding_mask=pad)
        return s.out(torch.cat([z,a.squeeze(1)],1))
ORDt=d['yt']<THR; ORDv=d['yv']<THR
rt=(d['yt'][ORDt]-d['bt'][ORDt]).astype(np.float32); rmn,rs=rt.mean(),rt.std()
Xt,Ct,Wtt,Mtt,rr=H.to_dev(torch.tensor(d['Xt'][ORDt]),torch.tensor(d['CT'][ORDt],dtype=torch.long),
    torch.tensor(Wt[ORDt]),torch.tensor(Mt[ORDt]),torch.tensor((rt-rmn)/rs))
Xv,Cv,Wvv,Mvv=H.to_dev(torch.tensor(d['Xv']),torch.tensor(d['CV'],dtype=torch.long),
    torch.tensor(Wv),torch.tensor(Mv))
print(f'resident {sum(t.element_size()*t.nelement() for t in (Xt,Wtt))/2**30:.2f} GB on {H.DEV}')
EVB=int(os.environ.get('EVB', max(2048, 65536*32//max(K,32))))  # eval chunk must also scale
                                                                # with K: the attention matrix
                                                                # is chunk*K*K and MPSGraph
                                                                # caps tensor dims at INT_MAX
def run(seed):
    torch.manual_seed(seed)
    m=WNet2(d['Xt'].shape[1],d['cards'],NF).to(H.DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-5)
    # MPSGraph refuses tensors with dims over INT_MAX; the attention matrix is B*K*K, so the
    # batch must shrink as K grows. K=128 at B=16384 overflows. Scale it, overridable.
    N=len(Xt); B=int(os.environ.get('BS', H.batch_for(K)))
    # Print B PER SEED. It has to be identical on every seed; when it was not, seeds 1 and 2
    # trained at 1024-2048 against seed 0's 16384 and diverged, and nothing in the output said so.
    print(f'  seed {seed}: B={B} EVB={EVB}',flush=True)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=3e-3,total_steps=EPOCHS*(N//B+1))
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(N,device=H.DEV)
        for i in range(0,N,B):
            j=perm[i:i+B]
            if len(j)<2: continue
            opt.zero_grad()
            nn.functional.mse_loss(m(Xt[j],Ct[j],Wtt[j],Mtt[j]).squeeze(1),rr[j]).backward()
            opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        o=torch.cat([m(Xv[i:i+EVB],Cv[i:i+EVB],Wvv[i:i+EVB],Mvv[i:i+EVB]).squeeze(1).float().cpu()
                     for i in range(0,len(Xv),EVB)]).numpy().astype(np.float64)
    return np.clip(d['bv']+(o*rs+rmn),60,200000)
ps=[];t0=time.time()
for s in range(SEEDS):
    ps.append(run(s)); print(f'  seed {s}: solo {np.sqrt(np.mean((ps[-1][ORDv]-d["yv"][ORDv])**2)):.3f} [{time.time()-t0:.0f}s]',flush=True)
p=np.mean(ps,0)
H.report(f'D3 multiscale SEL={SEL} (K={K}, 2 attn layers, {SEEDS} seeds)',p,d)
# K-AWARE. This line used to hardcode arch_d2_wide_va.npy for every K, so the K=64 run
# silently overwrote D2's K=32 validation predictions and they had to be recomputed. The
# canonical name is kept as an alias for K=32 only, which is what the rest of the repo reads.
np.save(f'{SP}/arch_d3_{SEL}_va_K{K}.npy',p)

# and the joint blend against D.
# s244: scipy's SLSQP on ((A@w-y)**2).sum() terminated on its first step and returned the
# UNIFORM starting point, because the objective is ~1e10 and its default ftol is relative to
# it. Every "OOF-fitted" weight from s230 to s244 was an equal-weight blend. Solved here in
# closed form through the KKT system of the equality-constrained least squares, with an
# active-set clip for negatives -- no scipy, and nothing that can silently not converge.
y=d['yv']; ORD=y<THR; yy=y[ORD]
P={'LGB':d['lgb'][ORD],'MLP':d['mlp'][ORD],'perApt':d['pa'][ORD],
   'D':np.load(f'{SP}/arch_d_window_va.npy')[ORD],'D2':p[ORD]}
half=np.arange(len(yy))%2==0
def simp(keys,m):
    A=np.stack([P[k][m] for k in keys],1); n=len(keys)
    act=np.ones(n,bool)
    for _ in range(n):
        Aa=A[:,act]; k=int(act.sum())
        # minimise ||Aa w - y||^2 s.t. sum w = 1, via the KKT system
        K=np.zeros((k+1,k+1)); K[:k,:k]=Aa.T@Aa; K[:k,k]=1.; K[k,:k]=1.
        r=np.zeros(k+1); r[:k]=Aa.T@yy[m]; r[k]=1.
        w=np.linalg.solve(K,r)[:k]
        if (w>=-1e-12).all(): break
        idx=np.where(act)[0]; act[idx[int(np.argmin(w))]]=False   # drop the most negative, refit
    out=np.zeros(n); out[act]=np.clip(w,0,None); return out/out.sum()
def oof(keys):
    tot=0.
    for f,s in ((half,~half),(~half,half)):
        w=simp(keys,f); A=np.stack([P[k][s] for k in keys],1); tot+=float(((A@w-yy[s])**2).sum())
    return np.sqrt(tot/len(yy))
board=lambda r: float(np.sqrt((344841*r*r+5.9122e9)/344841))
print(f'\n{"blend":<34}{"OOF":>9}{"board":>10}')
for keys,lab in ((['LGB','MLP','perApt','D'],'shipped three + D'),
                 (['LGB','MLP','perApt','D2'],'shipped three + D2'),
                 (['LGB','MLP','perApt','D','D2'],'shipped three + D + D2')):
    r=oof(keys); w=simp(keys,np.ones(len(yy),bool))
    print(f'{lab:<34}{r:>9.3f}{board(r):>10.4f}   w {np.round(w,4).tolist()}')
