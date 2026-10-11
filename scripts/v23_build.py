"""Regenerate quick-boat_v23.parquet: everything that measured positive, combined.

v23 is the v19 rung: v18's features plus k_q15, with the pooled model at raised
capacity. It is the best candidate we can construct from what has been measured.

  k_q15              s82 rejected it at +0.11 on the j3 window as "too fine". The
                     adversarial run re-measured -0.52 on the all window and
                     confirmed -0.38 on disjoint seeds. The flip is mechanism-
                     consistent: the ATFM slot compliance window is 15 minutes and
                     July added the regulation-heavy airports.
  pooled capacity    2000x255x20 is worth -1.82 on the pooled model. It is applied
                     via DC26_POOLED_CAP to the POOLED MODEL ONLY: the same config
                     costs +0.82 to +2.17 on the per-airport models, which train on
                     ~1/10 the data each and are already at the right capacity.

Bulk-only over v19/v21 by construction: F stays at 49, so Pc is bit-identical on
all 344,841 rows and the extreme side is provably unchanged.

NOT SUBMITTED. Its board value projects to 278.0-279.7 on a transfer ratio that
two measurements put between 0.15 and 0.87, which is why the range is that wide.
"""
import hashlib, os, pathlib, subprocess, sys

SHA = '3f3c616e1cd88c07772d5560e7ed0d5356c4cb69bd0e076d967f94139370bafa'
OUT = 'submissions/quick-boat_v23.parquet'
DATA = os.environ['DC26_DATA']
ENV = {'DC26_DATA': DATA, 'DC26_TAG': '_jul19', 'DC26_RECIPE': 'v19',
       'DC26_POOLED_CAP': '2000,255,20'}
UV = ['uv', 'run', '--quiet', '--with', 'pandas', '--with', 'numpy', '--with', 'pyarrow']
STAGES = [
    # v8s is recipe-independent; reuse the _jul build via symlink rather than rerun
    (['sh', '-c', 'T="${DC26_TMP:-/tmp}"; for f in v8s_tr v8s_rk; do ln -sf $T/${f}_jul.parquet $T/${f}_jul19.parquet; done; '
                  'ln -sf $T/v8s_cols_jul.json $T/v8s_cols_jul19.json'], None),
    (UV + ['python', '-u', 'scripts/v15_augment.py'], None),
    (UV + ['--with', 'scikit-learn', '--with', 'lightgbm', 'python', '-u', 'scripts/v15_trees.py'], None),
    (UV + ['--with', 'torch', 'python', '-u', 'scripts/v15_net.py'], None),
    (UV + ['python', '-u', 'scripts/v15_combine.py',
           f"{os.environ.get('DC26_TMP', '/tmp')}/v15_trees_jul19.parquet",
           f"{os.environ.get('DC26_TMP', '/tmp')}/v15_net_jul19.parquet", OUT], None),
]

def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

if '--run' in sys.argv:
    env = dict(os.environ, **ENV)
    for cmd, _ in STAGES:
        print('\n$ ' + ' '.join(f'{k}={v}' for k, v in ENV.items()) + ' ' + ' '.join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
else:
    print(__doc__)
    print('commands:')
    for cmd, _ in STAGES:
        print('  ' + ' '.join(f'{k}={v}' for k, v in ENV.items()) + ' ' + ' '.join(cmd))
    print()

assert pathlib.Path(OUT).exists(), f'{OUT} absent; rerun with --run'
got = sha256(OUT)
assert got == SHA, f'sha256 mismatch\n  expected {SHA}\n  got      {got}'
print(f'OK  {OUT}  sha256 {got}')
