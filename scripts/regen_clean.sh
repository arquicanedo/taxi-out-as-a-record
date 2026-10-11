#!/bin/bash
# The CLEAN model: v100's architecture and the trajectory stage (v102 -> v114) without the constants fitted to scores returned by
# the 2026 set. The base on the scored rows uses the learned five-class gate and labelled month offsets, which is the
# construction the labelled validation window has always used (scripts/clean_base_build.py); the trajectory stage is
# then re-priced and rebuilt on it with the ordinary pricers in CLEAN mode (DC26_SUBDIR, DC26_CAND, SHIFT_1/7, CLEAN=1).
# The side branches v105, v106 and v109 are not part of the clean chain (v106 priced positive on the window and was
# dropped on its leaderboard score, s535: a selection, disclosed in REPRODUCE.md section 12).
#   ./scripts/regen_clean.sh              WF=refit (default): the bulk weights refitted without any leaderboard offset
#   WF=archived ./scripts/regen_clean.sh  the weights as submitted in v115/v116 (scripts/v115_v116_build.sh uses this)
# Writes $CLEAN_DIR (default $DC26_WORK/clean) {submissions,cand,logs}; the repository's submissions/ is never touched.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; C="${CLEAN_DIR:-$WK/clean}"; mkdir -p "$C/submissions" "$C/cand" "$C/logs"
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python "$@"; }
PYX() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn "$@"; }
PYC() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost "$@"; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
# The learned five-class gate is a torch model (scripts/arch/b4_rank.py, eight seeds) that re-draws like the other nets.
# v115 and v116 were built from the archived copy (2026-09-09); it is restored when the work directory has none.
[ -f "$WK/b4_gate_rk.parquet" ] || cp "$REPO/artifacts/b4_gate_rk.parquet" "$WK/b4_gate_rk.parquet"
[ "$(shasum -a 256 "$WK/b4_gate_rk.parquet" | cut -c1-64)" = 2374a269a107afd88b72d50a5d09aba30adc8775ec0a018ac09193b6c66dcfd6 ] \
  || say "NOTE: b4_gate_rk.parquet is not the archived copy; v115/v116 will not reproduce byte for byte"
if [ "${SKIP_BASE:-0}" != 1 ]; then say "[0] clean base, WF=${WF:-refit}"; CLEAN_DIR="$C" PY scripts/clean_base_build.py 2>&1 | tee "$C/logs/base.log" | grep -v "^CONTROL\|^  -2T"; fi
rm -f "$C"/cand/v1*_adsb.npz "$C"/cand/stage_inputs.npz "$C"/cand/stage_mlp.npz   # a step that saves nothing then fails loudly
export SHIFT_1="$(python3 -c "import json;print(json.load(open('$C/offsets.json'))['SHIFT_1'])")"
export SHIFT_7="$(python3 -c "import json;print(json.load(open('$C/offsets.json'))['SHIFT_7'])")"
for m in 2 6; do v="$(python3 -c "import json;print(json.load(open('$C/offsets.json')).get('SHIFT_$m',''))")"; [ -z "$v" ] || export SHIFT_$m="$v"; done   # final phase: Feb, Jun
export DC26_SUBDIR="$C/submissions" DC26_CAND="$C/cand" DC26_WF="$C/wF.npz" CLEAN=1
say "offsets SHIFT_1=$SHIFT_1 SHIFT_7=$SHIFT_7"
step() { v=$1; shift; say "[$v] pricing"; "$@" > "$C/logs/price_$v.log" 2>&1 || { say "PRICING $v FAILED"; tail -20 "$C/logs/price_$v.log"; exit 1; }
         PY scripts/${v}_build.py 2>&1 | tail -2 | tee -a "$C/logs/build.log"; }
export ADSB_FEAT="${ADSB_FEAT_V102:-artifacts/adsb_push_feat_v102.parquet}"; step v102 PY scripts/adsb_price.py
export ADSB_FEAT="${ADSB_FEAT_V103:-artifacts/adsb_push_feat_v103.parquet}"; step v103 PY scripts/adsb_price_v103.py
export ADSB_FEAT="${ADSB_FEAT_V104:-artifacts/adsb_push_feat_v104.parquet}"; step v104 PY scripts/adsb_price_v104.py; unset ADSB_FEAT
step v107 PYX python scripts/adsb_price_v107.py
say "[v108] the MLP stage on the clean v107 (CPU, deterministic)"
PY scripts/stage_mlp_export.py > "$C/logs/mlp_export.log" 2>&1
STAGE_MLP_OUT="$C/cand/stage_mlp.npz" uv run --quiet --with numpy --with torch python scripts/stage_mlp.py > "$C/logs/mlp.log" 2>&1
export STAGE_MLP_FILE="$C/cand/stage_mlp.npz"
step v108 PYC python scripts/adsb_price_v108.py
say "[v110] full-year bulk members (archived predictions; no retraining)"
PY scripts/adsb_price_v110.py > "$C/logs/price_v110_prereg.log" 2>&1 || true
SAVE=Cfam PY scripts/adsb_v110_diag.py > "$C/logs/price_v110.log" 2>&1; grep -q 'saved cand/v110_adsb.npz' "$C/logs/price_v110.log"
PY scripts/v110_build.py 2>&1 | tail -2 | tee -a "$C/logs/build.log"
export SAVE=1; step v112 PYX python scripts/adsb_stage_full.py
step v114 PYX python scripts/adsb_price_v114.py; unset SAVE
say "CLEAN CHAIN DONE: $C/submissions/quick-boat_v114.parquet"
