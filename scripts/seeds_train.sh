#!/bin/bash
# s564: extra scored-side seeds for the CPU bulk members, in order of expected gain per hour (s563 pricing). Each script
# writes one file per seed under a new seed number, so nothing shipped is touched, and every finished seed is usable by
# scripts/seeds_members.py at once. K128 runs separately on the GPU (scripts/arch/d2_rank.py SEED0=8 SEED_DIR=...).
#   nohup ./scripts/seeds_train.sh > $DC26_WORK/seeds/train.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
NJ="${NJ:-16}"
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with scikit-learn python -u "$@"; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
say "CL seeds 2-9";        MODE=rank SEEDS=2,3,4,5,6,7,8,9 NJ=$NJ PY scripts/p2p_clean.py
say "L55A seeds 3-14";     MODE=rank TE=stored MODELS=pooled SEEDS=3,4,5,6,7,8,9,10,11,12,13,14 NJ=$NJ PY scripts/p2p_v55_adsb.py
say "ID seeds 2-9";        MODE=rank ARM=idcat SEEDS=2,3,4,5,6,7,8,9 NJ=$NJ PY scripts/p2p_arm.py
say "RQ seeds 2-9";        MODE=rank ARM=rwyq SEEDS=2,3,4,5,6,7,8,9 NJ=$NJ PY scripts/p2p_arm.py
say "P55A seeds 3-14";     MODE=rank TE=stored MODELS=perapt SEEDS=3,4,5,6,7,8,9,10,11,12,13,14 NJ=$NJ PY scripts/p2p_v55_adsb.py
say "CPU SEEDS DONE"
