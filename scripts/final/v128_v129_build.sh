#!/bin/bash
# The two interim reads of 2026-10-08 (analysis/FINAL-PLAN.md, "Pre-registration of reads 1 and 2").
#   v128  v127 unchanged: does the leaderboard still score against the truth v127 was scored on (247.7412)?
#   v129  S: v127 with the copy weight from the chain's classifier (alpha 1.00, fitted on the labelled window) on non-hit
#         rows, the tail re-run as v127's (scripts/final/exp_copyswap.py; inputs in $DC26_WORK, outputs in
#         $EXP_DIR). The decision rule is fixed in the plan before the upload.
set -euo pipefail; cd "$(dirname "$0")/../.."
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
[ "$(sha submissions/quick-boat_v128.parquet)" = 74fcf296d1ee90b08cfa6ce7b7356a8b82143ccf51dd2b87e1754925938f4d12 ]
cmp -s submissions/quick-boat_v127.parquet submissions/quick-boat_v128.parquet
[ "${SKIP_S:-0}" = 1 ] || uv run --quiet --with pandas --with numpy --with pyarrow --with lightgbm --with scikit-learn python scripts/final/exp_copyswap.py all
[ "$(sha "${EXP_DIR:?set EXP_DIR}/S.parquet")" = 11fe7fc77ad067aeed80e77a3542846620c73eb49ab0e7f8f77f91d29711405c ]
cmp -s "${EXP_DIR:?set EXP_DIR}/S.parquet" submissions/quick-boat_v129.parquet
echo "v128 74fcf296 (= v127), v129 11fe7fc7 (= S): both reproduce"
