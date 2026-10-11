#!/bin/bash
# v118 = v117 with less seed noise in the scored-side bulk members (s564, s568): K128 shipped 8 seeds + extra 8..17,
# CL seeds 0..9, L55A seeds 0..9; ID, RQ, P55A as shipped. Window side untouched, so every stage model, weight, alpha and
# rule is v117's. Rebuilds with scripts/regen_seeds.sh from the saved seed files, checks the hash, then copies.
#   ./scripts/v118_build.sh          SKIP_REGEN=1 reuses an existing $DC26_WORK/seeds/chain
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; C="${SEEDS_DIR:-$WK/seeds/chain}"
[ "${SKIP_REGEN:-0}" = 1 ] || SEEDS_DIR="$C" ./scripts/regen_seeds.sh
SHA118=2f2079f70ba9526ac7ee93f131711e64ece209266f4d6cf6378b364e365428de
h=$(shasum -a 256 "$C/submissions/quick-boat_v117.parquet" | cut -d' ' -f1)
[ "$h" = "$SHA118" ] || { echo "SHA MISMATCH: $h != $SHA118"; exit 1; }
cp "$C/submissions/quick-boat_v117.parquet" submissions/quick-boat_v118.parquet
echo "  submissions/quick-boat_v118.parquet  sha256 $h  matches"
