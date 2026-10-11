#!/bin/bash
# v127 = the CLEAN FINAL: v116 (v114's architecture and trajectory stage rebuilt without the constants fitted to leaderboard
# scores, REPRODUCE.md section 12) with the rules of v117 and v120 to v126 re-applied unchanged, so that a final model with
# no leaderboard-fitted constant has a score beside v126's. Nothing here is selected or fitted anew:
#   v117        the blend weight refitted outside LIRF, priced by its own pre-stated rule on the clean chain
#   v120-v126   the same rule scripts in CLEAN mode (their base and output passed in; shas not checked against the
#               shipped ones); the bulk v120/v121 hedge toward uses the clean chain's labelled month offsets
#               (offsets.json) in place of SHIFT_1/SHIFT_7, and the shipped v100 members
# Not re-applied: v118's extra seeds (seed noise on the scored side, not a rule) and v119 (it lost).
# On the first run (2026-10-05) v117's rule did NOT pass on the clean chain (G/SE +0.97 against its bar of 1), so v127 is
# v116 plus the rules of v120 to v126. PRE-REGISTERED before upload, from each rule's -dSSE/Q on the submitted chain times
# its Q on the clean one: 246.41; if every rule pays exactly its Q, 250.79; the two guaranteed parts alone (v122's clip
# onto the join interval, 1.8250e9, and v124's floor, 3.3815e7, both projections onto intervals that contain y) give
# 257.89, so a score above 257.89 would mean the rules that are not guaranteed lost.
#   ./scripts/v127_build.sh     needs $CLEAN_DIR from scripts/regen_clean.sh with WF=archived (the v116 run);
#                               works in $V127_DIR (default $DC26_WORK/clean127), writes submissions/quick-boat_v127.parquet
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; SRC="${CLEAN_DIR:-$WK/clean}"; C="${V127_DIR:-$WK/clean127}"
SHA116=a460c540c9d36e7930b2695f8b1c30533b70bcba5c21de14a8baa1884257a9c7
SHA127=74fcf296d1ee90b08cfa6ce7b7356a8b82143ccf51dd2b87e1754925938f4d12   # recorded from the first run
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python "$@"; }
PYX() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn "$@"; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
[ "$(sha "$SRC/submissions/quick-boat_v114.parquet")" = "$SHA116" ] || { echo "$SRC is not the v116 clean run"; exit 1; }
rm -rf "$C"; mkdir -p "$C/logs"; cp -R "$SRC/cand" "$SRC/submissions" "$SRC/wF.npz" "$SRC/offsets.json" "$C/"
export SHIFT_1="$(python3 -c "import json;print(json.load(open('$C/offsets.json'))['SHIFT_1'])")"
export SHIFT_7="$(python3 -c "import json;print(json.load(open('$C/offsets.json'))['SHIFT_7'])")"
export DC26_SUBDIR="$C/submissions" DC26_CAND="$C/cand" DC26_WF="$C/wF.npz" CLEAN=1 STAGE_MLP_FILE="$C/cand/stage_mlp.npz"
export DC26_MEMBERS_RK="$REPO/artifacts/p2_members_rk.parquet"
S="$C/submissions"; say "clean offsets SHIFT_1=$SHIFT_1 SHIFT_7=$SHIFT_7; base v116 $SHA116"

say "[v117] the blend weight refitted outside LIRF, by its pre-stated rule"
SAVE=1 PYX python -u scripts/adsb_price_v117.py > "$C/logs/price_v117.log" 2>&1
if grep -q 'saved cand/v117_adsb.npz' "$C/logs/price_v117.log"; then PY scripts/v117_build.py 2>&1 | tail -2
else say "  the rule did not pass on the clean chain: v117 = v114"; cp "$S/quick-boat_v114.parquet" "$S/quick-boat_v117.parquet"; fi

prev=117
for v in 120 121 122 123 124 125 126; do
  say "[v$v]"
  env "V${v}_BASE=$S/quick-boat_v$prev.parquet" "V${v}_OUT=$S/quick-boat_v$v.parquet" V121_REF="$S/quick-boat_v121.parquet" \
    uv run --quiet --with pandas --with pyarrow --with numpy python scripts/v${v}_build.py 2>&1 | grep -v "^  wrote\|^  if \|^  floor alone" | tail -3
  prev=$v
done
cp "$S/quick-boat_v126.parquet" submissions/quick-boat_v127.parquet
h=$(sha submissions/quick-boat_v127.parquet); say "wrote submissions/quick-boat_v127.parquet sha256 $h"
[ -z "$SHA127" ] || { [ "$h" = "$SHA127" ] && say "sha256 matches the recorded v127" || { echo "v127 does not reproduce: $h"; exit 1; }; }
