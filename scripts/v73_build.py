"""v73 = v64's chain with the K=128 movement window in the bulk. Model-level, no board arithmetic.

    v64 bulk  LGB .2473  perApt .2217  D .0849  D2 .4462              OOF 226.710
    v73 bulk  LGB .2230  perApt .2030  D .0000  D2 .0880  K128 .4860  OOF 225.737
    +0.974 OOF-s = +0.501 board-s at the measured 1.944x year gap (s247)

Every weight was fitted OUT OF FOLD on labelled 2025 rows with a KKT simplex solve, never read
off a leaderboard score. K=128 beat K=64 on both solo accuracy (229.463 vs 230.289) AND
independence (residual corr vs blend 0.9473 vs 0.9486), dropping the implied Bayes floor to
223.41. The fitted weights zero out D and cut D2 to 0.088: K=128 subsumes both predecessors.

The extreme chain is UNCHANGED. v64_build.py's control still asserts byte-identity with v59 at
the shipped bulk before anything else runs.
"""
import os, subprocess, sys
W=os.environ['DC26_WORK']
RK=f'{W}/d2_window_rk_K128.parquet'
if not os.path.exists(RK): sys.exit(f'ranking-side K=128 not built yet: {RK}')
env=dict(os.environ, W_LGB='0.2230', W_MLP='0.0000', W_PA='0.2030',
         W_D='0.0000', W_D2='0.0880', W_K128='0.4860',
         V64_OUT=os.environ.get('V73_OUT','submissions/quick-boat_v73.parquet'))
sys.exit(subprocess.call([sys.executable,'scripts/v64_build.py'],env=env))
