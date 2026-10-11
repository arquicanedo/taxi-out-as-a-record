#!/bin/bash
# s554: month-fold out-of-fold predictions of the full-year ADS-B members on 2025 rows (seed 0), the base a full-year ADS-B
# stage is trained against. NEST=win (five folds inside the ten non-window months: the window stays honest) and NEST=all
# (six folds over twelve: production). P folds in parallel, NJ threads each. Resumable (finished member files are skipped).
#   ./scripts/oof_folds_v112.sh [P] [NJ]
cd "$(dirname "$0")/.."; P="${1:-3}"; NJ="${2:-6}"; L="${DC26_WORK:?set DC26_WORK to a work directory}/adsblol/oof"; mkdir -p "$L"
{ for f in 2,8 3,9 4,10 5,11 6,12; do echo "win $f"; done; for f in 1,7 2,8 3,9 4,10 5,11 6,12; do echo "all $f"; done; } |
  xargs -P "$P" -L 1 bash -c 'MODE=oof NEST=$0 FOLD=$1 SEEDS=0 NJ='"$NJ"' uv run --quiet --with pandas --with pyarrow --with numpy \
    --with scikit-learn --with lightgbm python scripts/p2p_v55_adsb.py > "'"$L"'/oof_$0_${1/,/-}.log" 2>&1; echo "$0 $1 exit $? $(tail -1 "'"$L"'/oof_$0_${1/,/-}.log")"'
