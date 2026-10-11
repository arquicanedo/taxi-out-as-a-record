"""RETIRED. This is probe-and-invert code, published deliberately.

WHY IT IS HERE. This script implements the method our paper discloses and retires: submit a
known perturbation on a known row set, read the public score, solve `dSSE = 2T + Q` for the
inner product of the perturbation with the true errors, and ship the optimum. We used it, it
contributed a measured +2.7501 RMSE-seconds or more to our standing, and we stopped on
2026-09-11 after an organiser wrote that reverse-engineering the ranking is unwelcome.

**We publish it rather than describe it** because a disclosure a reader cannot check is not a
disclosure. `scripts/v76_build.py` builds the same model with this content removed and ships
alongside the model that keeps it, so the difference can be priced without taking our word for
anything. Every constant is itemised in REPRODUCE.md section 9.

**It is not run by any shipped build**, and some of these scripts depend on intermediate state
that no longer exists, so several will not execute at all. They are here to be read. Do not run
them: the method measures the evaluation set rather than the problem, which is why we stopped,
independent of any rule.

---

Does v54 repeat v55's error? v55 measured a gain on a window whose COVERAGE differed from
the scored set's (89.2% vs 72.2%) and lost 2.96 s. The analogous question for v54:

  does the cell's COMPOSITION on validation match its composition on the scored set?

The scored cell is 32 rows, all LIRF. If the validation cell is spread across airports, then
alpha* = +0.6813 was measured on a population the scored cell does not resemble, and it is the
same mistake wearing different clothes."""
import glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq
# Paths are env defaults so this frozen script names no directory only one machine has.
# DC26_TRAIN_LEGACY defaults to the PRE-REISSUE extract these scripts were written
# against; its twelve training_*.parquet are identical to the current extract's, which
# is why nothing here ever failed loudly. Behaviour is unchanged with the vars unset.
W=os.environ['DC26_WORK']; TRAIN=os.environ.get('DC26_TRAIN_LEGACY', os.environ['DC26_DATA'])
DATA=os.environ['DC26_DATA']; THR,FLOOR,CAP,BS=7200,60,200000,1.842e8
z=np.load(f'{W}/gate_stage_a2.npz'); zm=np.load(f'{W}/gate_mlp.npz')
va=pd.read_parquet(f'{W}/night_va_all.parquet')
assert np.allclose(va.TAXITIME_SEC_mvt.values.astype(float),z['yv'])
yv,Pc,sdv,lb=z['yv'],z['Pc'],z['sd_v'],z['lb_v']
bulk=0.2*z['lgb_va']+0.4*zm['mlp_va']+0.4*z['pa_va']
sdc=np.clip(sdv,FLOOR,CAP); lbc=np.clip(np.where(np.isnan(lb),sdv,lb),FLOOR,CAP)
hit=(lb>=THR)&~np.isnan(lb); copy=np.where(hit,lbc,sdc); P=np.maximum(Pc,hit*0.9)
apt=va.apt.astype(str).values; mon=va.month.values
b=np.clip(P*copy+(1-P)*bulk,FLOOR,CAP)
V34=(apt=='LIRF')&(~np.isfinite(lb))&(sdv>=50000.)
p=np.where(V34,np.clip(np.rint(0.40*np.clip(bulk+86400.,FLOOR,CAP)+0.60*b),FLOOR,CAP),b)
p=np.clip(np.rint(p+np.where(mon==1,-8.778,-3.861)),FLOOR,CAP)
C49=(Pc>=.10)&(Pc<.95)&~V34; p=np.where(C49,np.clip(np.rint((1-.3376)*p+.3376*copy),FLOOR,CAP),p)
C51=(~np.isfinite(lb))&(sdv>=7200)&(sdv<14400)&(Pc>=.01)&~C49&~V34
p=np.where(C51,np.clip(np.rint((1-.2922)*p+.2922*bulk),FLOOR,CAP),p)
r=(p-yv)**2
allf=pd.concat([pq.read_table(f,columns=['PHASE_mvt','FLIGHT_mvt']).to_pandas()
                for f in sorted(glob.glob(f'{TRAIN}/training_*.parquet'))],ignore_index=True)
allf=allf[allf.PHASE_mvt=='DEP'].FLIGHT_mvt.astype(str)
KNOWN=frozenset(allf[allf.str.match(r'^[A-Z]{3}\d')].str[:3].value_counts()[lambda s:s>=50].index)
def dup(x):
    x=str(x)
    if len(x)<4 or not x[:3].isalpha() or x[:3] not in KNOWN: return False
    return any(len(x)>=3+k and x[3:3+k]==x[:3][-k:] for k in (1,2,3))
raw=pd.concat([pq.read_table(f,columns=['MVT_ID_mvt','FLIGHT_mvt','CALLSIGN_flt']).to_pandas()
               for f in sorted(glob.glob(f'{TRAIN}/training_*.parquet'))],
              ignore_index=True).drop_duplicates('MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(va.MVT_ID_mvt)
CELL=np.array([dup(x) for x in raw.FLIGHT_mvt.astype(str)])&raw.CALLSIGN_flt.isna().values
EX=CELL&~V34
print(f'VALIDATION cell: {int(EX.sum())} rows')
print('  by airport: '+', '.join(f'{a}:{int(((apt==a)&EX).sum())}' for a in sorted(set(apt[EX]))))
print(f'  extreme {int((yv[EX]>=THR).sum())} ({(yv[EX]>=THR).mean():.1%})   mean P_used {P[EX].mean():.4f}')
rk=pq.read_table(f'{DATA}/ranking.parquet',columns=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','FLIGHT_mvt',
    'CALLSIGN_flt','LOBT_flt']).to_pandas()
v51=pq.read_table('submissions/quick-boat_v51.parquet').to_pandas()
rd=rk[rk.PHASE_mvt=='DEP'].set_index('MVT_ID_mvt').reindex(v51.MVT_ID_mvt)
t=pd.read_parquet(f'{W}/v15_trees_jul19.parquet'); nn=pd.read_parquet(f'{W}/v15_net_jul19.parquet')
mm=t.merge(nn,on='MVT_ID_mvt').set_index('MVT_ID_mvt').reindex(v51.MVT_ID_mvt)
sdr=mm.sched_delay.values.astype(float); lbr=mm.mvt_minus_lobt.values.astype(float); Pcr=mm.Pc.values.astype(float)
hitr=np.isfinite(lbr)&(lbr>=THR); Pr=np.maximum(Pcr,hitr*0.9)
R34=(rd.ADEP_mvt.astype(str).values=='LIRF')&rd.LOBT_flt.isna().values&(sdr>=50000.)
RC=np.array([dup(x) for x in rd.FLIGHT_mvt.astype(str)])&rd.CALLSIGN_flt.isna().values&~R34
print(f'\nSCORED cell: {int(RC.sum())} rows')
print('  by airport: '+', '.join(f'{a}:{int(((rd.ADEP_mvt.astype(str).values==a)&RC).sum())}' for a in sorted(set(rd.ADEP_mvt.astype(str).values[RC]))))
print(f'  mean P_used {Pr[RC].mean():.4f}')
print('\n=== the v55-shaped question: is alpha* stable when restricted to LIRF? ===')
D=np.clip(copy,FLOOR,CAP)-p
for lab,msk in (('validation cell, ALL airports',EX),('validation cell, LIRF only',EX&(apt=='LIRF')),
                ('validation cell, non-LIRF',EX&(apt!='LIRF'))):
    ok=msk&(D!=0); n=int(ok.sum())
    if n<3: print(f'  {lab:32s} n={n} -- too few'); continue
    Q=float((D[ok]**2).sum()); T=float((D[ok]*(p[ok]-yv[ok])).sum()); a=-T/Q
    q=p.copy(); q[ok]=np.clip(np.rint(p[ok]+0.60*D[ok]),FLOOR,CAP)   # what we SHIPPED
    r2=(q-yv)**2
    print(f'  {lab:32s} n={n:>3}  ex {int((yv[ok]>=THR).sum()):>2} ({(yv[ok]>=THR).mean():>5.1%})  '
          f'meanP {P[ok].mean():.3f}  alpha* {a:+.4f}  gain@0.60 {(r.sum()-r2.sum())/BS:+.4f}')

print('\n=== THE v55 LESSON APPLIED: re-measure on rows matching the SCORED P distribution ===')
print('v55 failed because a quantity was measured where a condition (coverage) did not match')
print('the scored set. Here the mismatched condition is P_used: 0.5458 val vs 0.7152 scored.')
print(f'\n  P_used distribution')
for lab,v in (('validation cell',P[EX]),('scored cell',Pr[RC])):
    print(f'    {lab:16s} '+' '.join(f'{q:.0%}={np.quantile(v,q):.3f}' for q in (.1,.25,.5,.75,.9))+
          f'   frac P>=0.7: {(v>=0.7).mean():.1%}')
print(f'\n  alpha* and the SHIPPED 0.60, on validation subsets chosen to match the scored P')
for lab,sub in (('all 33 rows (what was used)',EX),
                ('P_used >= 0.50',EX&(P>=0.50)),
                ('P_used >= 0.70  <- scored mean',EX&(P>=0.70)),
                ('P_used >= 0.85',EX&(P>=0.85))):
    ok=sub&(D!=0); n=int(ok.sum())
    if n<3: print(f'    {lab:32s} n={n} -- too few to price'); continue
    Q=float((D[ok]**2).sum()); T=float((D[ok]*(p[ok]-yv[ok])).sum()); a=-T/Q
    q=p.copy(); q[ok]=np.clip(np.rint(p[ok]+0.60*D[ok]),FLOOR,CAP)
    r2=(q-yv)**2; o=np.argsort(-np.maximum(r,r2))
    k20=(r[o[20:]].sum()-r2[o[20:]].sum())/BS
    print(f'    {lab:32s} n={n:>3}  ex {(yv[ok]>=THR).mean():>5.1%}  meanP {P[ok].mean():.3f}  '
          f'alpha* {a:+.4f}  gain@0.60 {(r.sum()-r2.sum())/BS:+.4f}  k20 {k20:+.4f}')
print('\n  reweight the validation cell to the scored P distribution (importance weights)')
w=np.ones(len(P))
b_edges=np.array([0,.3,.5,.7,.85,.95,1.01])
vb=np.digitize(P[EX],b_edges); sb=np.digitize(Pr[RC],b_edges)
for i in range(1,len(b_edges)+1):
    nv=(vb==i).sum(); ns=(sb==i).sum()
    if nv: w[np.flatnonzero(EX)[vb==i]]=ns/nv
ok=EX&(D!=0)
Qw=float((w[ok]*D[ok]**2).sum()); Tw=float((w[ok]*D[ok]*(p[ok]-yv[ok])).sum())
print(f'    reweighted alpha* {-Tw/Qw:+.4f}   (unweighted {+0.6813:+.4f})')
print(f'    reweighted gain at 0.60: {(2*0.60*(-Tw/Qw)-0.60**2)*Qw/BS:+.4f} board-s')
