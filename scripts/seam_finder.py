"""SEAM FINDER: the end-to-end methodology, reusable on any airport or population.

Built 2026-09-20 after the carrier seam paid +0.979 (v96) and KMM cost -2.31 (v97). Every guard
below was bought with a specific failure in this repo; the citations are load-bearing, not decoration.

    uv run --quiet --with pandas --with numpy --with pyarrow python scripts/seam_finder.py \
        --airport LIRF --join fail --ms-min 600 --ms-max 7200 [--perm 200]

WHAT IT DOES

 1. SLICE SEARCH.  Every *_mvt categorical and quantile-binned numeric, singly and pairwise, scoring
    the best LEVEL rather than the column (CLAUDE.md rule 5: a flat marginal hides a sharp joint --
    EGLL looked undetectable at 4.1% and the joint reached AUC 0.9539).

 2. D^2-WEIGHTED RATE, NOT AUC (s237).  The event probability is not what squared loss wants. Route
    reached AUC 0.88 and lost money; rank statistics have died five times here.

 3. OUT OF FOLD BY MONTH (s389, s391).  A rate measured in-sample is not a rate. Four leads died to
    the Jan+Jul window alone, which is the hardest of the six two-month folds.

 4. PERMUTATION NULL (s393).  The ENTIRE search is re-run on shuffled labels and the best-found lift
    recorded. On 627 rows a null search routinely finds 0.42 lift; without this the pipeline is a
    finding generator. Report only what beats the null's MAXIMUM.

 5. THE COMMIT/FEATURE SPLIT (s396).  THIS IS THE ONE MOST RECENTLY PAID FOR.
       rate >= 0.99 AND no month below 0.95  ->  COMMIT-GRADE. A structurally broken feed: the
            system never records an off-block, and a thing that never happens keeps never happening.
            v96's eleven carriers were 99.11% in 2025 and delivered 100% in 2026, dSSE == -Q.
       anything below that                   ->  FEATURE-GRADE ONLY. A mixture whose weight is a
            free parameter of the year. KMM was 81.7% across twelve months and delivered 3.2%,
            because its feed was repaired between years. Do not commit a mixture.

 6. FOR COMMIT-GRADE ONLY, the remaining guards:
       v94  check what the non-copy alternative IS, per predicate, not pooled. v94 assumed an
            ordinary taxi where the truth was a date error and lost 4.13.
       s305 per-row break-even against the incumbent, and a single-row mass cap. v80 put 58.7% of
            its bet on one row and lost 8.0989.
       s390 every join asserted on MVT_ID, never on len. Two scripts here joined positionally and
            agreed with reality on 0.0003% of rows.
       s398 SUPPORT-RANGE CHECK. No fired row may sit outside the range of the labelled rows that
            justify it. AEZ2617 is 92/92 copies with max ms 2,884; a 2026 row at ms 12,905 is 4.5x
            beyond anything the predicate was ever verified on, and it carried 94.2% of a candidate
            submission. v94 died of exactly this at a different scale. The rule is: for each fired
            row, its predictor must lie inside [q01, q99] of the labelled support, or it is dropped.
"""
def arrival_2026_gate(carriers, airport, D=os.environ['DC26_DATA']):
    """s413: THE THIRD GATE, and the only LABELLED 2026 instrument on this mechanism.

    ranking.parquet holds 344,693 ARRIVALS with BLOCK_TIME and the target PRESENT, in both scored
    months. A handling agent that does not feed the off-block system is a property of the HANDLER,
    not of the direction, so a departure-side total-fallback carrier is checkable on 2026 rows.

    THE FOOTING IS NOT OBVIOUS AND GETTING IT WRONG MEASURES A FILTERED POPULATION. For an arrival
    y = BLOCK - MVT, so a schedule copy gives y = SCHED - MVT, which is positive only when the
    aircraft lands EARLY (measured: arrival copies are early on 99.97% of 2025 rows). The mirror of
    v96's "departs >600 s late" is therefore "lands >600 s EARLY". On late arrivals the eleven
    carriers score 0.000, which is the footing artefact and not a contradiction.

    NECESSARY, NOT SUFFICIENT. Measured on 2025 where both directions are labelled: departure-HIGH
    implies arrival-HIGH on 7 of 8 pairs, but arrival-HIGH does NOT imply departure-HIGH --
    LIRF/MSR is 0.955 on 154 arrivals against 0.195 on 524 departures, every month in [0.11,0.33].
    Use this to KILL a candidate whose 2026 arrival rate has collapsed (the test v97 never had:
    KMM's feed was repaired between years and no 2025 evidence could say so). Never use it to
    LICENSE a carrier known only from the arrival side.
    """
    import glob as _g
    C=['ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','FLIGHT_mvt']
    d=pq.read_table(f'{D}/ranking.parquet',columns=C).to_pandas()
    for c in ('MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt'):
        d[c]=pd.to_datetime(d[c],errors='coerce',utc=True)
    d=d[d.PHASE_mvt.eq('ARR')&d.ADES_mvt.astype(str).eq(airport)]     # ADES for an ARRIVAL (s220)
    early=(d.SCHED_TIME_UTC_mvt-d.MVT_TIME_UTC_mvt).dt.total_seconds()
    d=d[(early>600)&d.FLIGHT_mvt.astype(str).str[:3].isin(list(carriers))].copy()
    d['cp']=((d.BLOCK_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()<=60)
    d['mon']=d.MVT_TIME_UTC_mvt.dt.month
    out={'n':len(d),'rate':float(d.cp.mean()) if len(d) else float('nan'),
         'by_month':{int(m):(int(g.cp.sum()),len(g)) for m,g in d.groupby('mon')},
         'by_carrier':{c:(int(g.cp.sum()),len(g)) for c,g in d.groupby(d.FLIGHT_mvt.astype(str).str[:3])}}
    return out


def support_range_ok(fired_ms, labelled_ms, lo=0.01, hi=0.99):
    """s398: a fired row must sit inside the labelled support that justifies it."""
    if len(labelled_ms)<5: return np.zeros(len(fired_ms),bool)
    a,b=np.quantile(labelled_ms,lo),np.quantile(labelled_ms,hi)
    return (fired_ms>=a)&(fired_ms<=b)
import os, glob, re, argparse, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, pyarrow.parquet as pq
D=os.environ['DC26_DATA']
TYP,PER=1021.,1.842e8
ap=argparse.ArgumentParser()
ap.add_argument('--airport',default='LIRF'); ap.add_argument('--join',default='fail',choices=['fail','present','any'])
ap.add_argument('--ms-min',type=float,default=600.); ap.add_argument('--ms-max',type=float,default=1e9)
ap.add_argument('--perm',type=int,default=0); ap.add_argument('--top',type=int,default=10)
a=ap.parse_args()
sch=pq.read_schema(f'{D}/ranking.parquet'); MVT=[c for c in sch.names if c.endswith('_mvt')]
L=pd.concat([pq.read_table(f,columns=MVT+['LOBT_flt']).to_pandas()
             for f in sorted(glob.glob(f'{D}/training_*.parquet'))],ignore_index=True)
for c in ['MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']: L[c]=pd.to_datetime(L[c],errors='coerce',utc=True)
d=L[L.PHASE_mvt.eq('DEP')].copy(); d['y']=d.TAXITIME_SEC_mvt.astype(float)
d['ms']=(d.MVT_TIME_UTC_mvt-d.SCHED_TIME_UTC_mvt).dt.total_seconds()
d['mon']=d.MVT_TIME_UTC_mvt.dt.month
TEN=set(d.ADEP_mvt.unique())
m=d.ADEP_mvt.eq(a.airport)&d.ms.between(a.ms_min,a.ms_max)
if a.join=='fail': m&=d.LOBT_flt.isna()
elif a.join=='present': m&=d.LOBT_flt.notna()
c=d[m].copy(); c['cp']=(np.abs(c.y-c.ms)<=60).astype(int); c['D2']=(c.ms-TYP)**2
print(f'population: {a.airport} join={a.join} ms in [{a.ms_min:,.0f},{a.ms_max:,.0f}]')
print(f'  {len(c):,} labelled rows, copy {c.cp.mean():.2%}, D^2-weighted {float((c.D2*c.cp).sum()/c.D2.sum()):.2%}')
if len(c)<150: raise SystemExit('population too small to search')
def feats(x):
    F={'carrier':x.FLIGHT_mvt.astype(str).str[:3],'ades_reg':x.ADES_mvt.astype(str).str[:2],
       'ades_ten':x.ADES_mvt.isin(TEN).map({True:'T',False:'O'}),'runway':x.RUNWAY_mvt.astype(str),
       'stand_pre':x.STAND_mvt.astype(str).str[:1],'actype':x.AIRCRAFT_TYPE_mvt.astype(str).str[:2],
       'garb':x.FLIGHT_mvt.astype(str).str.match(r'^([A-Z]{3})(\1|[A-Z]{1,3})\d').map({True:'G',False:'C'}),
       'flt_len':x.FLIGHT_mvt.astype(str).str.len().astype(str)}
    for nm,src,bins in (('ms_b',x.ms,5),('stand_n',pd.to_numeric(x.STAND_mvt.astype(str).str.extract(r'(\d+)')[0],errors='coerce'),4)):
        try: F[nm]=pd.qcut(src,bins,labels=list('abcde')[:bins],duplicates='drop')
        except Exception: F[nm]=pd.Series('NA',index=x.index)
    F['hour_b']=pd.cut(x.MVT_TIME_UTC_mvt.dt.hour,[-1,5,11,17,23],labels=['n','m','a','e'])
    F['dow_b']=pd.cut(x.MVT_TIME_UTC_mvt.dt.dayofweek,[-1,0,3,5,6],labels=['mo','mi','fr','we'])
    return {k:v.astype(str).fillna('NA') for k,v in F.items()}
F=feats(c); keys=list(F)
labs=[(k,F[k].values) for k in keys]+[(f'{x} x {y}',(F[x]+'|'+F[y]).values) for i,x in enumerate(keys) for y in keys[i+1:]]
D2=c.D2.values; mon=c.mon.values; MO=np.unique(mon); MINN=max(25,int(0.01*len(c)))
def best(cpv,collect=False):
    base=float((D2*cpv).sum()/D2.sum()); out=[]; bb=0.
    for nm,lab in labs:
        s=pd.Series(lab); vc=s.value_counts()
        for v in vc[vc>=MINN].index:
            sel=(lab==v); num=0.;den=0.
            for mm in MO:
                te=sel&(mon==mm); tr=sel&(mon!=mm)
                if not te.any() or tr.sum()<10: continue
                r=float((D2[tr]*cpv[tr]).sum()/max(D2[tr].sum(),1e-9))
                num+=r*float(D2[te].sum()); den+=float(D2[te].sum())
            if den<=0: continue
            l=abs(num/den-base); bb=max(bb,l)
            if collect: out.append((l,nm,str(v),int(sel.sum()),num/den))
    return (sorted(out,reverse=True),bb) if collect else bb
hits,rb=best(c.cp.values,collect=True)
print(f'  {len(labs)} slice families, min level size {MINN}')
thr=None
if a.perm:
    rng=np.random.default_rng(0)
    nl=np.array([best(rng.permutation(c.cp.values)) for _ in range(a.perm)])
    thr=nl.max()
    print(f'\n  PERMUTATION NULL over {a.perm} shuffles: median {np.median(nl):.4f}  '
          f'p99 {np.percentile(nl,99):.4f}  MAX {thr:.4f}')
print(f'\n  {"lift":>7} {"n":>6} {"rate":>7} {"grade":>14}  slice')
for l,nm,v,n,rate in hits[:a.top]:
    sub=c[np.array([lab for k,lab in labs if k==nm][0])==v]
    pm=[float(x.cp.mean()) for _,x in sub.groupby('mon') if len(x)>=3]
    worst=min(pm) if pm else 0.
    hi=max(pm) if pm else 1.
    # s401: THE TEMPORAL SPLIT. The permutation null above answers "did this happen by chance".
    # It cannot answer "will this happen again", and those are different questions -- bias_sweep
    # found a slice 4.4x the null max that LOST 5.04 board-s when corrected out of sample. KMM
    # (v97) passed the null and drifted 84.6% -> 77.5% across the halves: -2.31 board-s.
    h1=sub[sub.mon<=6].cp; h2=sub[sub.mon>=7].cp
    r1=float(h1.mean()) if len(h1)>=10 else np.nan
    r2=float(h2.mean()) if len(h2)>=10 else np.nan
    gap=abs(r1-r2) if np.isfinite(r1) and np.isfinite(r2) else np.nan
    stable=bool(np.isfinite(gap) and gap<=0.05)
    # DETERMINISTIC MEANS NEAR 0 OR NEAR 1. v95 shipped stand-4xx at 0/28 and won; the first
    # version of this grader called that "feature-only" because s396 was written from the v96
    # side only. A slice that NEVER copies is as structural as one that ALWAYS does.
    if   rate>=0.99 and worst>=0.95 and stable: grade='COMMIT +'
    elif rate<=0.01 and hi<=0.05    and stable: grade='COMMIT -'
    elif (rate>=0.99 and worst>=0.95) or (rate<=0.01 and hi<=0.05): grade='DRIFTS'
    elif thr is not None and l<=thr: grade='below null'
    else: grade='feature-only'
    g=f'{gap:.1%}' if np.isfinite(gap) else '  n/a'
    print(f'  {l:>7.4f} {n:>6} {rate:>7.1%} {grade:>14}  {nm} = {v}   month range '
          f'[{worst:.2f},{hi:.2f}]  H1/H2 {r1:.2f}/{r2:.2f} gap {g}')
print(f'\n  GRADING (s396, corrected for direction):')
print(f'    COMMIT +  rate>=0.99 and every month >=0.95  -- the feed NEVER reports an off-block')
print(f'    COMMIT -  rate<=0.01 and every month <=0.05  -- the feed ALWAYS does (v95, stand 4xx)')
print(f'    DRIFTS    deterministic overall but H1 and H2 disagree by >5pp -- do not commit')
print(f'    feature   anything in between: a MIXTURE, and a mixture weight is a free parameter of')
print(f'              the year. KMM was 81.7% over twelve months and delivered 3.2%: -2.31 board-s.')
print(f'\n  AND THE SECOND GATE, which the split does NOT see (s398): every fired row must sit')
print(f'  inside its predicate\'s labelled ms support. v98 was 100.0% in BOTH halves, 100% in Jan')
print(f'  and Jul, and still lost 0.96 because ONE fired row sat at ms 12,905 against a labelled')
print(f'  range of 600-2,884. Call support_range_ok() on the fired set before shipping; the two')
print(f'  gates are independent and each caught exactly the loss the other could not see.')
print(f'\n  AND THE THIRD GATE (s413), the only LABELLED 2026 instrument: call arrival_2026_gate()')
print(f'  on a carrier predicate. v96\'s eleven score 138/138 on 2026 LIRF arrivals landing >600s')
print(f'  EARLY, Jan 43/43 and Jul 95/95, which is the test v97 never had. It is NECESSARY and not')
print(f'  sufficient: LIRF/MSR is 0.955 on arrivals and 0.195 on departures. Kill with it, never license.')
