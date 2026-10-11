#!/bin/bash
# v115 and v116: the CLEAN model (REPRODUCE.md section 12), submitted once to measure its score on the 2026 set.
#   v115  the clean base: learned five-class gate and labelled month offsets. Its nine-member bulk weights are
#         full_package.npz's, fitted against y minus the leaderboard offsets (s561); WF=refit removes that trace
#   v116  the clean final: the trajectory stage re-priced and rebuilt on v115 by the same rules
# Not a search: both files are fixed by labelled 2025 data before either is scored, and the paper reports whatever
# they score. Regenerates with scripts/regen_clean.sh, copies under the submission names, asserts the recorded sha256.
#   ./scripts/v115_v116_build.sh              SKIP_REGEN=1 reuses an existing $DC26_WORK/clean
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; C="${CLEAN_DIR:-$WK/clean}"
[ "${SKIP_REGEN:-0}" = 1 ] || WF=archived CLEAN_DIR="$C" ./scripts/regen_clean.sh   # the weights as submitted
check() { h=$(shasum -a 256 "$1" | cut -d' ' -f1); [ "$h" = "$2" ] || { echo "SHA MISMATCH $1: $h"; exit 1; }; echo "  $1  sha256 $h  matches"; }
check "$C/submissions/quick-boat_v100.parquet" 00bf453efc80d6d5805332ef81c3becde0d28da0e8ad7f35a2ad622009ebf8e4   # checked before
check "$C/submissions/quick-boat_v114.parquet" a460c540c9d36e7930b2695f8b1c30533b70bcba5c21de14a8baa1884257a9c7   # anything is copied
cp "$C/submissions/quick-boat_v100.parquet" submissions/quick-boat_v115.parquet
cp "$C/submissions/quick-boat_v114.parquet" submissions/quick-boat_v116.parquet
echo "  submissions/quick-boat_v115.parquet and quick-boat_v116.parquet rebuilt byte-identically"
