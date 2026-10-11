"""s564: the scored-side bulk members averaged over every seed trained so far, for v118 = v117 with less seed noise.

Adding seeds fits nothing: each member's recipe, data and blend weight stay as shipped, and only the average over seeds
gets longer. The expected squared error removed is the seed-noise variance the longer average takes out (s563, s564).

  K128   the shipped 8-seed mean (d2_window_rk_K128.parquet, via p2_members_rk.parquet) combined with every extra seed
         in $SEEDS_ROOT/k128/k128rk_s{n}.npy (scripts/arch/d2_rank.py SEED0=8 SEED_DIR=..., v15_rk_jul19 order)
  CL     every p2P/clean/rank_clean_s{n}.npy (scripts/p2p_clean.py MODE=rank; submitting order)
  ID, RQ every p2P/arms/rank_{idcat,rwyq}_s{n}.npy (scripts/p2p_arm.py MODE=rank; v15_rk_jul19 order)
  L55A, P55A  not here: the pricers read p2P/trees_adsb/rank_stored_{pooled,perapt}_s{n}.npy through ADSB_SEEDS_POOLED and
         ADSB_SEEDS_PERAPT; this script prints the seed lists present and writes them to $SEEDS_OUT/seeds.env
Writes $SEEDS_OUT/p2_members_rk.parquet. CONTROL=1 uses only the shipped seeds and asserts the file equals the shipped one.
  SEEDS_OUT=... uv run --quiet --with pandas --with pyarrow --with numpy python scripts/seeds_members.py
"""
import glob, os, re
import numpy as np, pandas as pd, pyarrow.parquet as pq
W = os.environ['DC26_WORK']; D = os.environ['DC26_DATA']
P2 = os.environ.get('DC26_SESSION', f'{W}/session') + '/p2P'; ROOT = os.environ.get('SEEDS_ROOT', f'{W}/seeds')
OUT = os.environ['SEEDS_OUT']; os.makedirs(OUT, exist_ok=True); CONTROL = os.environ.get('CONTROL') == '1'
# the scored rows in the members' order: from stage 2's v15_rk_jul19.parquet, or, without it, from the archived K128
# seed files, which carry the same ids in the same order (the assert below checks the two agree when both exist)
_rk = f'{W}/v15_rk_jul19.parquet'
rk_ids = (pd.read_parquet(_rk, columns=['MVT_ID_mvt']).MVT_ID_mvt.values if os.path.exists(_rk)
          else np.load(f'{ROOT}/k128/k128rk_ids.npy', allow_pickle=True))
sub_ids = pq.read_table(f'{D}/submitting.parquet').to_pandas().MVT_ID_mvt.values
r2s = pd.Series(np.arange(len(rk_ids)), index=rk_ids).reindex(sub_ids).values.astype(int); assert (rk_ids[r2s] == sub_ids).all()
ship = pd.read_parquet(f'{W}/p2_members_rk.parquet').set_index('MVT_ID_mvt').reindex(sub_ids); M = ship.copy()
num = lambda f: int(re.search(r'_s(\d+)\.npy$', f).group(1))
def present(pat, shipped):
    fs = sorted(glob.glob(pat.format(s='*')), key=num); have = [num(f) for f in fs]
    assert all(s in have for s in shipped), f'shipped seeds missing for {pat}'
    return list(shipped) if CONTROL else have
mean_of = lambda pat, ss: np.mean([np.load(pat.format(s=s)) for s in ss], 0)
log = []
kf = [] if CONTROL else sorted(glob.glob(f'{ROOT}/k128/k128rk_s*.npy'), key=num)
if kf:
    assert (np.load(f'{ROOT}/k128/k128rk_ids.npy', allow_pickle=True) == rk_ids).all(), 'K128 seed files are not in v15_rk_jul19 order'
    new = np.stack([np.load(f) for f in kf])[:, r2s]
    M['K128'] = (8 * ship.K128.values + new.sum(0)) / (8 + len(kf))
log.append(f'K128 shipped 8 + extra {[num(f) for f in kf]}')
s = present(f'{P2}/clean/rank_clean_s{{s}}.npy', (0, 1)); M['CL'] = mean_of(f'{P2}/clean/rank_clean_s{{s}}.npy', s); log.append(f'CL seeds {s}')
for k, arm in (('ID', 'idcat'), ('RQ', 'rwyq')):
    s = present(f'{P2}/arms/rank_{arm}_s{{s}}.npy', (0, 1)); M[k] = mean_of(f'{P2}/arms/rank_{arm}_s{{s}}.npy', s)[r2s]; log.append(f'{k} seeds {s}')
sp = present(f'{P2}/trees_adsb/rank_stored_pooled_s{{s}}.npy', (0, 1, 2)); sq = present(f'{P2}/trees_adsb/rank_stored_perapt_s{{s}}.npy', (0, 1, 2))
log.append(f'L55A seeds {sp}'); log.append(f'P55A seeds {sq}')
assert M.notna().all().all()
if CONTROL:
    assert np.array_equal(M.to_numpy(float), ship.to_numpy(float)), 'CONTROL: rebuilt members differ from the shipped file'
M.reset_index().to_parquet(f'{OUT}/p2_members_rk.parquet', index=False)
open(f'{OUT}/seeds.env', 'w').write(f"ADSB_SEEDS_POOLED={','.join(map(str, sp))}\nADSB_SEEDS_PERAPT={','.join(map(str, sq))}\n")
d = M.to_numpy(float) - ship.to_numpy(float)
print('\n'.join(log)); print(f'members changed vs shipped: ' + '  '.join(f'{c} RMS {np.sqrt(np.mean(d[:, i] ** 2)):.2f}' for i, c in enumerate(M.columns)))
print(f'wrote {OUT}/p2_members_rk.parquet and {OUT}/seeds.env' + ('  (CONTROL: equals the shipped file)' if CONTROL else ''))
