#!/usr/bin/env bash
# v99 -> v100, the final submission (254.5228 on the public board, 2026-09-23).
#
# v100 = v99 with the bulk blend refitted as the s492 package: nine members, fixed out-of-fold weights
# embedded in scripts/v100_build.py (fitted on labelled 2025 data only), applied on v99's unfired rows.
# This driver rebuilds the five NEW ranking-side members from the features and applies the weights.
# The four shipped members (v55 trees, K32 and K128 windows) come from regen2.sh stages 7-9.
#
#   DRY=1 scripts/regen_v100.sh      print the plan
#   scripts/regen_v100.sh            run it: about 1 to 2.5 hours on 18 cores
#
# Exact path (seconds): with the archived members ($DC26_WORK/p2_members_rk.parquet) and the
# archived v99, `python scripts/v100_build.py` reproduces v100 BYTE-IDENTICALLY (verified 2026-09-23).
# From scratch some LightGBM members are re-drawn (LightGBM times its histogram strategy at run time and CL's early
# stopping amplifies the difference; findings s513b), so v100 reproduces NUMERICALLY: measured 2026-09-23, RMS 3.334 s
# on 262,824 rows with every CPU member rebuilt. The last step reports the difference from the shipped file.
set -euo pipefail
: "${DC26_WORK:?set DC26_WORK to a work directory}"
S="$(cd "$(dirname "$0")" && pwd)"
R="uv run --quiet --with pandas --with numpy --with scikit-learn --with scipy --with pyarrow --with lightgbm python -u"
export P2P_OUT="${P2P_OUT:-$DC26_WORK/regen_v100/p2P}"      # a fresh folder: the archived members are never overwritten
export P2M_OUT="${P2M_OUT:-$DC26_WORK/regen_v100/p2_members_rk.parquet}"
mkdir -p "$P2P_OUT"
run() { if [ -n "${DRY:-}" ]; then echo "  would run: $*"; else echo "+ $*"; "$@"; fi; }
echo "[1] target encodings: ten-month, twelve-month and out-of-fold (5 calendar-day folds)"
run $R "$S/p2p_te.py"
echo "[2] L55o, P55o: the v55 trees on out-of-fold encodings, ranking side, seeds 0,1"
run env MODE=rank TE=oof SEEDS=0,1 $R "$S/p2p_v55.py"
echo "[3] ID: airport x operator / stand / carrier identity categoricals;  RQ: same-runway neighbours' naive"
run env MODE=rank ARM=idcat SEEDS=0,1 $R "$S/p2p_arm.py"
run env MODE=rank ARM=rwyq  SEEDS=0,1 $R "$S/p2p_arm.py"
echo "[4] CL: the independently built model, raw frame -> features (2025 and 2026) -> ranking side"
run $R "$S/p1c_load.py"
for sel in train rank; do for k in 1 2 3; do run env SEL=$sel $R "$S/p2p_clean_feat$k.py"; done; done
run env MODE=rank SEEDS=0,1 $R "$S/p2p_clean.py"
echo "[5] members -> $P2M_OUT, then v100"
run env P2P_ALLOW_DRIFT=1 $R "$S/p2p_save_members.py"
run env V100_MEMBERS="$P2M_OUT" V100_OUT="${V100_OUT:-$DC26_WORK/rebuilt/quick-boat_v100.parquet}" V100_ALLOW_DRIFT=1 $R "$S/v100_build.py"
