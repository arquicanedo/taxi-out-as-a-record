#!/bin/bash
# The final model, v126, in one command once the v114 inputs are in place (README "Reproduce", REPRODUCE.md sections
# 11 and 13): the adsb.lol feature tables rebuilt by scripts/regen_adsb.sh and v15_rk_jul19.parquet in $DC26_WORK.
#   1  CONTROL=1 scripts/regen_seeds.sh   v100 -> v117 with the shipped seeds; must equal the shipped v117 byte for byte
#   2  scripts/v118_build.sh              the same chain with every archived seed -> submissions/quick-boat_v118.parquet
#   3  scripts/v120_build.py ... v126     the rules, each asserting its base and its own sha256
#   DRY=1 ./scripts/reproduce_final.sh    prints the plan        SKIP_CONTROL=1 skips step 1
#   TAIL=1 ./scripts/reproduce_final.sh   v120 to v126 only, from the tracked v118 and the archived members and seeds:
#                                         seconds, and no adsb.lol tables needed
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
: "${DC26_DATA:?set DC26_DATA to the challenge extract directory}"
: "${DC26_WORK:?set DC26_WORK to a work directory}"
export DC26_SESSION="${DC26_SESSION:-$DC26_WORK/session}"
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }
PY=(uv run --quiet --with pandas --with pyarrow --with numpy python)
[ -n "${UV_EXCLUDE_NEWER:-}" ] || echo "note: UV_EXCLUDE_NEWER is not set; REPRODUCE.md section 0 gives the date that pins the versions"
if [ "${TAIL:-0}" = 1 ]; then   # from the tracked v118 and the archived members and seeds: seconds, no adsb.lol tables
  run env RESTORE_ONLY=1 scripts/regen_adsb.sh
  run env RESTORE_ONLY=1 scripts/regen_seeds.sh
  run env SEEDS_OUT="$DC26_WORK/seeds/chain" "${PY[@]}" scripts/seeds_members.py
else
  [ "${SKIP_CONTROL:-0}" = 1 ] || run env CONTROL=1 scripts/regen_seeds.sh
  run scripts/v118_build.sh
fi
for v in 120 121 122 123 124 125 126; do run "${PY[@]}" "scripts/v${v}_build.py"; done
[ -n "${DRY:-}" ] || echo "v126 sha256 $(shasum -a 256 submissions/quick-boat_v126.parquet | cut -c1-64)  (expected d66768fe3504ddcc...)"
