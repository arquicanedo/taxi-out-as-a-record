#!/bin/bash
# v118 = v117 with less seed noise (s564): the scored-side bulk members averaged over more seeds, then the submitted chain
# rebuilt on its scored side. The window side is untouched, so every stage model, weight, alpha and selection rule is the
# shipped one; only the 2026 predictions move. Uses the CLEAN-mode plumbing (own submissions/ and cand/ directories, every
# env set) with the SUBMITTED chain's offsets and weights; nothing in it is a clean rebuild.
#   ./scripts/regen_seeds.sh             every seed trained so far -> $SEEDS_DIR/submissions/quick-boat_v117.parquet
#   CONTROL=1 ./scripts/regen_seeds.sh   the shipped seeds only; must rebuild v117 byte for byte (sha 47c5f030...)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; SS="${DC26_SESSION:-$WK/session}"
C="${SEEDS_DIR:-$WK/seeds/chain$([ "${CONTROL:-0}" = 1 ] && echo _control || true)}"
# The seed predictions v118 adds (artifacts/seeds, 32 files: K128 seeds 8-17, CL 0-9, ID and RQ 0-1, L55A 3-9; the
# K128 net re-draws on Metal, so they are archived rather than retrained): restore any missing from the work
# directories, and refuse to run if one that is present differs from the archived copy. seeds_members.py averages every
# seed file it finds, so a work directory holding further seeds builds a different model.
while read -r sum f; do f="${f#./}"; src="artifacts/seeds/$f"
  [ -f "$src" ] && [ "$(shasum -a 256 "$src" | cut -c1-64)" = "$sum" ] || { echo "archived seed file $f missing or corrupt"; exit 1; }
  case "$f" in session-20260922/*) dst="$SS/${f#session-20260922/}" ;; *) dst="$WK/$f" ;; esac
  if [ ! -f "$dst" ]; then mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; echo "  restored $f"
  elif [ "$(shasum -a 256 "$dst" | cut -c1-64)" != "$sum" ]; then echo "work-dir $f differs from artifacts/seeds"; exit 1; fi
done < artifacts/seeds/SHA256SUMS
[ "${RESTORE_ONLY:-0}" = 1 ] && { echo "  seed files in place"; exit 0; }
rm -rf "$C/submissions" "$C/cand"; mkdir -p "$C/submissions" "$C/cand" "$C/logs"
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python "$@"; }
PYX() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn "$@"; }
PYC() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost "$@"; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
say "[0] members (CONTROL=${CONTROL:-0})"; SEEDS_OUT="$C" PY scripts/seeds_members.py | tee "$C/logs/members.log"
set -a; . "$C/seeds.env"; set +a
V100_MEMBERS="$C/p2_members_rk.parquet" V100_MEMBERS0="$WK/p2_members_rk.parquet" V100_OUT="$C/submissions/quick-boat_v100.parquet" \
  V100_ALLOW_DRIFT=1 PY scripts/v100_build.py 2>&1 | tail -3 | tee "$C/logs/v100.log"
export DC26_SUBDIR="$C/submissions" DC26_CAND="$C/cand" DC26_WF="$SS/p2P/full_package.npz" SHIFT_1=-8.778 SHIFT_7=-3.861 CLEAN=1
export DC26_MEMBERS_RK="$C/p2_members_rk.parquet"
step() { v=$1; shift; say "[$v] pricing"; "$@" > "$C/logs/price_$v.log" 2>&1 || { say "PRICING $v FAILED"; tail -20 "$C/logs/price_$v.log"; exit 1; }
         PY scripts/${v}_build.py 2>&1 | tail -2 | tee -a "$C/logs/build.log"; }
export ADSB_FEAT=artifacts/adsb_push_feat_v102.parquet; step v102 PY scripts/adsb_price.py
export ADSB_FEAT=artifacts/adsb_push_feat_v103.parquet; step v103 PY scripts/adsb_price_v103.py
export ADSB_FEAT=artifacts/adsb_push_feat_v104.parquet; step v104 PY scripts/adsb_price_v104.py; unset ADSB_FEAT
step v107 PYX python scripts/adsb_price_v107.py
say "[v108] the MLP stage (CPU, deterministic)"
PY scripts/stage_mlp_export.py > "$C/logs/mlp_export.log" 2>&1
STAGE_MLP_OUT="$C/cand/stage_mlp.npz" uv run --quiet --with numpy --with torch python scripts/stage_mlp.py > "$C/logs/mlp.log" 2>&1
export STAGE_MLP_FILE="$C/cand/stage_mlp.npz"
step v108 PYC python scripts/adsb_price_v108.py
say "[v110] full-year bulk members (seeds pooled ${ADSB_SEEDS_POOLED}, per-airport ${ADSB_SEEDS_PERAPT})"
PY scripts/adsb_price_v110.py > "$C/logs/price_v110_prereg.log" 2>&1 || true
SAVE=Cfam PY scripts/adsb_v110_diag.py > "$C/logs/price_v110.log" 2>&1; grep -q 'saved cand/v110_adsb.npz as variant "C alpha"' "$C/logs/price_v110.log"
PY scripts/v110_build.py 2>&1 | tail -2 | tee -a "$C/logs/build.log"
export SAVE=1; step v112 PYX python scripts/adsb_stage_full.py
grep -q 'saved cand/v112_adsb.npz as "C2 alpha"' "$C/logs/price_v112.log"
step v114 PYX python scripts/adsb_price_v114.py
grep -q 'saved cand/v114_adsb.npz as "no LIRF"' "$C/logs/price_v114.log"
step v117 PYX python scripts/adsb_price_v117.py; unset SAVE
grep -q 'saved cand/v117_adsb.npz as "alpha refit outside LIRF"' "$C/logs/price_v117.log"
h=$(shasum -a 256 "$C/submissions/quick-boat_v117.parquet" | cut -d' ' -f1)
if [ "${CONTROL:-0}" = 1 ]; then
  [ "$h" = 47c5f0309c6a415617c615125c8160435cb38aa3ba918e6c4fba51b13406ee11 ] || { say "CONTROL FAILED: $h"; exit 1; }
  say "CONTROL: the shipped seeds rebuild v117 byte for byte"
else
  say "SEEDS CHAIN DONE: $C/submissions/quick-boat_v117.parquet sha256 $h (v118 candidate)"
fi
