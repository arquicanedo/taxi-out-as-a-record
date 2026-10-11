"""D2: the movement-window transformer, widened. Slot-4 candidate attempt.

s241's lesson: solo accuracy and bias independence trade off along the FEATURE SET, so a fourth
component must differ in its INPUTS, not its head or its objective. D is the only component that
differs in its inputs -- it reads the neighbour movement window -- and the only one sitting at
its decorrelation floor (slack +0.0003). So the way to extend the blend is to give D MORE of the
thing nothing else sees, not to build another model on the same columns.

Changes from D: K=16 -> 32 neighbours, and a second attention layer so the model can compose
neighbour-of-neighbour structure rather than pooling once. Same six per-neighbour features, same
constraint that only ARRIVAL taxi-in is readable.
"""
import sys, os, time, glob, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch, torch.nn as nn
import harness as H
SP=os.environ['DC26_WORK']; D=os.environ['DC26_DATA']
K=int(os.environ.get('K','32')); NF=6; SEEDS=int(os.environ.get('SEEDS','3'))
# NEEDW=0 passes need_weights=False to both attention calls. Default 1 = unchanged behaviour.
# nn.MultiheadAttention defaults need_weights=True, and this script discards the weights with
# `sa,_=`, so the K x K averaged weight matrix is built and never read. With it off, PyTorch can
# take its fused path instead of the math path. NOT the SDPA rewrite (DELL-PERF 3(a)): the module,
# its parameters and their layout are untouched, so a checkpoint is still loadable either way.
# It CAN move the numerics by kernel choice, so it is gated on the solo control like anything else.
NEEDW=os.environ.get('NEEDW','1')=='1'
EPOCHS=int(os.environ.get('EPOCHS','14')); CACHE=f'{SP}/arch_d_window_K{K}.npz'
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
    off=np.array([o for o in range(-(K//2),K//2+1) if o!=0][:K])
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
    # DELL-PERF.md 3(d), applied 2026-09-13. Was savez_compressed: single-threaded zlib, which
    # took 176 s on the 7.47 GB K=128 cache (1.59 GB out, 4.6x) and would be ~14.94 GB at K=256,
    # so the brief's "rebuilds in under a minute" did not hold. Plain savez trades disk for wall
    # clock. W goes out fp16 per 3(c), which also halves what zlib was chewing on.
    np.savez(CACHE,W=H.as_wdtype(W),M=M,n_tr=len(tr))
    print(f'built K={K}: {W.shape} filled {M.mean():.1%} [{time.time()-t0:.0f}s]')
    return W,M,len(tr)
if os.path.exists(CACHE):
    z=np.load(CACHE); W,M,ntr=z['W'],z['M'],int(z['n_tr']); print(f'loaded {CACHE}')
else: W,M,ntr=build()
# Cast on LOAD too, so a pre-existing fp32 cache gets 3(c)'s halving without a rebuild.
W=H.as_wdtype(W); print(f'window dtype {W.dtype}, {W.nbytes/2**30:.2f} GiB')
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
        it=s.item(w.float()); pad=(m<0.5)   # w is parked fp16 (DELL-PERF 3(c)); cast at use
        pad=torch.where(pad.all(1,keepdim=True),torch.zeros_like(pad),pad)
        # THIS LINE IS THE ENTIRE MEMORY BUDGET (DELL-PERF.md s1, 2026-09-13). With a
        # key_padding_mask in training mode nn.MultiheadAttention takes the math path and
        # materialises B x heads x K x K in fp32, held for backward: 4.00 GiB at K=128 B=16384,
        # 8.00 GiB at K=256 B=8192. It is the layer D2 added over D, and it is a bandwidth
        # workload, which is why an RTX 4000 Ada (~360 GB/s) lost to the M5 Pro at 36:03 vs
        # 2085 s. F.scaled_dot_product_attention tiles it away; patch in DELL-PERF.md 3(a),
        # gated on reproducing solo 232.386 here at K=128.
        sa,_=s.self_att(it,it,it,key_padding_mask=pad,need_weights=NEEDW); it=s.ln(it+sa)
        # Query length 1, so B x heads x 1 x K. Negligible next to the line above.
        a,_=s.att(s.q(z).unsqueeze(1),it,it,key_padding_mask=pad,need_weights=NEEDW)
        return s.out(torch.cat([z,a.squeeze(1)],1))
ORDt=d['yt']<THR; ORDv=d['yv']<THR
rt=(d['yt'][ORDt]-d['bt'][ORDt]).astype(np.float32); rmn,rs=rt.mean(),rt.std()
Xt,Ct,rr=H.to_dev(torch.tensor(d['Xt'][ORDt]),torch.tensor(d['CT'][ORDt],dtype=torch.long),
    torch.tensor((rt-rmn)/rs))
Xv,Cv=H.to_dev(torch.tensor(d['Xv']),torch.tensor(d['CV'],dtype=torch.long))
# The windows are the only RESIDENT tensors large enough to force streaming: at K=256 the train
# window is 1,740,323 x 256 x 6 x 4 B = 9.96 GiB against a 12 GB card, and it has to coexist with
# its own activations. park() keeps them resident when they fit and pins them in host RAM when
# they do not; take()/chunk() read either placement, so the arithmetic is identical both ways.
# They are NOT the largest single consumer, though: see the self_att comment above. Streaming the
# window frees 9.96 GiB and the attention layer then asks for 8.00 GiB of it back.
Wtt,Mtt=H.park(torch.tensor(Wt[ORDt])),H.park(torch.tensor(Mt[ORDt]))
Wvv,Mvv=H.park(torch.tensor(Wv)),H.park(torch.tensor(Mv))
_gb=lambda t: t.element_size()*t.nelement()/2**30
print(f'windows {"STREAMED from pinned host" if Wtt.device.type=="cpu" else "resident"}: '
      f'train W {_gb(Wtt):.2f} GB + val W {_gb(Wvv):.2f} GB, X {_gb(Xt):.2f} GB on {H.DEV}')
EVB=int(os.environ.get('EVB', H.batch_for(K,16384)))  # eval chunk must also scale with K.
                                                      # Was max(2048, 65536*32//max(K,32)),
                                                      # which is 16384 at K=128; batch_for's
                                                      # CUDA anchor reproduces that and halves
                                                      # it at K=256, where a 32768 eval chunk
                                                      # would want 4.3 GB per activation.
                                                      # Eval runs under no_grad, so the K x K
                                                      # matrix is not kept for backward and this
                                                      # path is the safe one. TRAINING is where
                                                      # the anchor under-shrinks: see batch_for's
                                                      # docstring, B*K const means B*K*K grows.
def run(seed):
    torch.manual_seed(seed)
    m=WNet2(d['Xt'].shape[1],d['cards'],NF).to(H.DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=float(os.environ.get('LR','2e-3')),weight_decay=1e-5)
    # MPSGraph refuses tensors with dims over INT_MAX; the attention matrix is B*K*K, so the
    # batch must shrink as K grows. K=128 at B=16384 overflows. Scale it, overridable.
    N=len(Xt); B=int(os.environ.get('BS', H.batch_for(K)))
    # LR IS TIED TO BATCH AND THE BATCH CHANGED. Added 2026-09-13. 3e-3/2e-3 were tuned on Metal,
    # where batch_for gives 16384/8192/4096 at K=32/64/128. On a 12 GB card K=128 must drop to
    # 2048, half the Mac's, and at 3e-3 it diverges: TF32 off, 3 seeds, 233.852 / 796.867 /
    # 1066.282. K=64 is unaffected because its batch, 8192, is the SAME on both backends, which
    # is what proved TF32 rather than batch was the K=64 culprit. Linear scaling rule: scale LR
    # by B/B_ref. Override with MAXLR/LR to test it; default reproduces the recorded schedule.
    print(f'  seed {seed}: B={B} EVB={EVB} TF32={H.TF32}',flush=True)   # must not vary by seed
    mx=float(os.environ.get('MAXLR','3e-3'))
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=mx,total_steps=EPOCHS*(N//B+1))
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(N,device=H.DEV)   # stays on DEV: the RNG stream must not
        for i in range(0,N,B):                           # change when the windows stream
            j=perm[i:i+B]
            if len(j)<2: continue
            opt.zero_grad()
            nn.functional.mse_loss(m(Xt[j],Ct[j],H.take(Wtt,j),H.take(Mtt,j)).squeeze(1),
                                   rr[j]).backward()
            opt.step(); sch.step()
        if ep==0 and seed==0: H.mem_report(f' K={K} B={B}')
    m.eval()
    with torch.no_grad():
        o=torch.cat([m(Xv[i:i+EVB],Cv[i:i+EVB],H.chunk(Wvv,i,EVB),
                       H.chunk(Mvv,i,EVB)).squeeze(1).float().cpu()
                     for i in range(0,len(Xv),EVB)]).numpy().astype(np.float64)
    return np.clip(d['bv']+(o*rs+rmn),60,200000)
ps=[];t0=time.time()
for s in range(SEEDS):
    ps.append(run(s)); print(f'  seed {s}: solo {np.sqrt(np.mean((ps[-1][ORDv]-d["yv"][ORDv])**2)):.3f} [{time.time()-t0:.0f}s]',flush=True)
p=np.mean(ps,0)
H.report(f'D2 wide window (K={K}, 2 attn layers, {SEEDS} seeds)',p,d)
# K-AWARE. This line used to hardcode arch_d2_wide_va.npy for every K, so the K=64 run
# silently overwrote D2's K=32 validation predictions and they had to be recomputed. The
# canonical name is kept as an alias for K=32 only, which is what the rest of the repo reads.
np.save(f'{SP}/arch_d2_wide_va_K{K}.npy',p)
if K==32: np.save(f'{SP}/arch_d2_wide_va.npy',p)
# Everything below is a diagnostic weight solve against the K=16 window net's predictions, which no release script
# produces. The output above is final; stop cleanly rather than fail the driver when that file is absent (2026-09-23).
if not os.path.exists(f'{SP}/arch_d_window_va.npy'):
    print(f'saved {SP}/arch_d2_wide_va_K{K}.npy; blend diagnostic skipped (no arch_d_window_va.npy)'); sys.exit(0)
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
