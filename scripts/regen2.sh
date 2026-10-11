#!/bin/sh
# Regenerate BOTH shipped models from a fresh challenge extract. Supersedes regen.sh.
#
# WHY THIS EXISTS. regen.sh (2026-09-08) stops at v45, which scored 266.8355 against the shipped
# v74's 261.9189 -- it reproduces a model 4.9166 board-s behind our entry, and contains none of
# the window stack that now carries 0.574 of the blend weight. This driver goes all the way to
# v74 (best) and v76 (robust), which are the only two models the public repository ships.
#
# USAGE
#   DRY=1 sh scripts/regen2.sh              validate the graph, run nothing. DO THIS FIRST.
#   sh scripts/regen2.sh                    full run against the CURRENT extract, ~10-14 h
#   DC26_DATA=/path/to/new sh scripts/regen2.sh     a fresh extract; see FRESH EXTRACT below
#
# FRESH EXTRACT. Stages marked [THIS-EXTRACT] assert byte-identity against a shipped parquet and
# CANNOT pass on new rows -- that is their purpose. On a fresh extract:
#   - set FRESH=1, which skips those asserts and skips the v29 archive in favour of
#     v23 -> v25 -> the 0.35 blend (scripts/v29_build.py, exact there: see s284)
#   - every board-fitted constant must be RE-DERIVED or dropped. v76 is the variant that
#     drops them; see MODELS.md and s280.
set -e
: "${DC26_DATA:?set DC26_DATA to the challenge extract directory}"
: "${DC26_WORK:?set DC26_WORK to a work directory}"
export DC26_DATA DC26_WORK
export DC26_TAG="${DC26_TAG:-_jul19}"        # the shipped chain's tag; every v15_*_jul19 artefact
# The _jul19 tables were built by v23_build.py's ENV, not by this driver's: recipe v19 (+k_q15) and the
# pooled capacity override. Without these two, stages 2-3 build the v18 rung at 600x127x100, a different
# bulk model from the shipped v15_trees_jul19.parquet. Missing here until 2026-09-23 (REPRODUCE.md s3 trap).
export DC26_RECIPE="${DC26_RECIPE:-v19}"
export DC26_POOLED_CAP="${DC26_POOLED_CAP:-2000,255,20}"
R="uv run --quiet --with pandas --with numpy --with scikit-learn --with pyarrow --with lightgbm"
RT="uv run --quiet --with pandas --with numpy --with pyarrow --with torch"
W="$DC26_WORK"; S="$(cd "$(dirname "$0")" && pwd)"
DRY="${DRY:-}"; FRESH="${FRESH:-}"; FROM="${FROM:-1}"; ONLY="${ONLY:-}"
# FROM=N resumes at stage N so a failure 9 hours in does not restart from zero.
# ONLY=N runs exactly one stage. Both respect DRY.
STAGE=0

say() { printf '\n== %s ==\n' "$1"; }
skip() { STAGE=$((STAGE+1))
  if [ -n "$ONLY" ]; then [ "$STAGE" != "$ONLY" ]; else [ "$STAGE" -lt "$FROM" ]; fi; }
need() { [ -e "$1" ] || { echo "  MISSING INPUT: $1"; MISS=$((MISS+1)); }; }
prod() { [ -f "$S/$1" ] || { echo "  MISSING PRODUCER: scripts/$1"; MISS=$((MISS+1)); }; }
run() { if [ -n "$DRY" ]; then echo "  would run: $*"; else "$@"; fi; }
# pin FILE: the neural components re-draw on every run (MPS) and the chain cannot absorb it (findings s513, s514:
# a re-drawn MLP moves v74 by RMS 30 s, a re-drawn K128 differs from the shipped one by RMS 41 s). So the chain reads
# the ARCHIVED predictions from artifacts/, and this run's are kept beside them as *_redrawn. USE_REDRAWN=1 opts out.
pin() { A="$S/../artifacts/$1"
  if [ -n "$DRY" ]; then echo "  would pin $1 from artifacts/"; return; fi
  if [ -f "$A" ] && [ -z "$USE_REDRAWN" ]; then
    [ -f "$W/$1" ] && mv "$W/$1" "$W/${1%.*}_redrawn.${1##*.}"; cp "$A" "$W/$1"; echo "  pinned $1 from artifacts/ (USE_REDRAWN=1 keeps this run's)"
  fi; }

MISS=0
say "0. preflight"
echo "  DC26_DATA=$DC26_DATA"
echo "  DC26_WORK=$W"
echo "  DC26_TAG=$DC26_TAG   FRESH=${FRESH:-0}"
need "$DC26_DATA/submitting.parquet"; need "$DC26_DATA/ranking.parquet"
# the extract names these training_2025-MM-01_2025-(MM+1)-01.parquet, so count rather than
# enumerate: an assumed filename pattern was wrong on the first dry run.
NTR=$(ls "$DC26_DATA"/training_*.parquet 2>/dev/null | wc -l | tr -d ' ')
[ "$NTR" = "12" ] || { echo "  EXPECTED 12 training_*.parquet, found $NTR"; MISS=$((MISS+1)); }
echo "  training files: $NTR"
# OPDI is read from stage 6 on; ONLY=1..5 runs without it
if [ -z "$ONLY" ] || [ "$ONLY" -ge 6 ]; then need "$S/../data/opdi/flight_list_202501.parquet"; fi
need "$S/../reference/pru/ATFM_Slot_Adherence.parquet"
need "$S/../reference/pru/Airport_Arrival_ATFM_Delay.parquet"
for p in v8s_build.py v15_augment.py v15_trees.py v15_net.py opdi_features.py v55_trees.py \
         arch/d_rank.py arch/d2_rank.py v23_build.py v25_build.py v29_build.py \
         v64_build.py v73_build.py v74_build.py v76_build.py; do prod "$p"; done
# v66_build.py and v67_build.py implement the RETIRED probe-and-invert method and are published as disclosure code
# (MANIFEST.md s2). Stage 13 composes their archived outputs from submissions/ by default. RERUN_RETIRED=1 re-runs them
# instead, which only re-applies constants already fitted: neither reads the leaderboard. See REPRODUCE.md section 9.
V66V67="${RERUN_RETIRED:-0}"
if [ "$V66V67" = "1" ]; then for p in v66_build.py v67_build.py; do [ -f "$S/$p" ] || { echo "  RERUN_RETIRED=1 but $p is absent"; exit 1; }; done; fi
[ -n "$FRESH" ] || { need "$S/../submissions/quick-boat_v29.parquet"
                     need "$S/../submissions/quick-boat_v59.parquet"; }
mkdir -p "$W"
if [ "$MISS" -gt 0 ]; then echo "\n$MISS missing. external data: scripts/pull_pru.py, scripts/pull_opdi.py"; exit 1; fi
echo "  preflight OK"

# SCRATCH ISOLATION. Every producer reads DC26_WORK, DC26_DATA and DC26_TMP from the environment and has no
# default, so nothing can write outside them. (On 2026-09-14 two scripts with a hardcoded work path overwrote a
# neural component that could not be rebuilt; that is why there are no defaults.)

if skip; then echo "-- stage 1 skipped"; else
say "1. features -> v8s tables"
run $R python -u "$S/v8s_build.py"
fi
if skip; then echo "-- stage 2 skipped"; else
say "2. + block_dev / sched_vs_hr -> v15 tables"
run $R python -u "$S/v15_augment.py"
fi
if skip; then echo "-- stage 3 skipped"; else
say "3. bulk LightGBM + per-airport + classifier Pc"
# SEPARATE PROCESS from stage 4: LightGBM and PyTorch segfault together (exit 139, OpenMP clash).
run $R python -u "$S/v15_trees.py"
fi
if skip; then echo "-- stage 4 skipped"; else
say "4. the MLP component (8 nets, torch)"
run $RT python -u "$S/v15_net.py"
fi
if skip; then echo "-- stage 5 skipped"; else
say "5. stage /tmp intermediates where the build scripts look"
if [ -z "$DRY" ]; then
  T="${DC26_TMP:-/tmp}"
  for f in v15_tr v15_rk v15_trees v15_net; do cp "$T/${f}${DC26_TAG}.parquet" "$W/"; done
  cp "$T/v15_cols${DC26_TAG}.json" "$W/"
  # the MLP re-draws (RMS ~6 s) and v64's reconstruction of v29 amplifies it ~5x (s513): pin the archive
  pin "v15_net${DC26_TAG}.parquet"
else echo "  would copy ${DC26_TMP:-/tmp}/v15_{tr,rk,trees,net}${DC26_TAG}.parquet -> $W"; fi

fi
if skip; then echo "-- stage 6 skipped"; else
say "6. OPDI airframe features   [external: OPDI flight_list]"
run $R python -u "$S/opdi_features.py"
fi
if skip; then echo "-- stage 7 skipped"; else
say "7. v55 trees: both components retrained with the airframe features"
run $R python -u "$S/v55_trees.py"

fi
if skip; then echo "-- stage 8 skipped"; else
say "8. window component D  (K=16) -> d_window_rk.parquet   [GPU, ~1 h]"
run $RT python -u "$S/arch/d_rank.py"
pin d_window_rk.parquet        # the 2026-09-14 re-draw; the original K=16 was overwritten that day
fi
if skip; then echo "-- stage 9 skipped"; else
say "9. window component D2 (K=32) -> d2_window_rk.parquet  [GPU, ~2 h]"
run env K=32 $RT python -u "$S/arch/d2_rank.py"
pin d2_window_rk.parquet
fi
if skip; then echo "-- stage 10 skipped"; else
say "10. window component K128     -> d2_window_rk_K128.parquet  [GPU, ~5 h]"
run env K=128 $RT python -u "$S/arch/d2_rank.py"
pin d2_window_rk_K128.parquet

fi
if skip; then echo "-- stage 11 skipped"; else
say "11. the calibration base v29"
if [ -n "$FRESH" ]; then
  run $R python -u "$S/v23_build.py"
  run $R python -u "$S/v25_build.py"
  run env MODE=emit V29_OUT="$W/quick-boat_v29_reconstructed.parquet" $R python -u "$S/v29_build.py"
  export V29_SRC="$W/quick-boat_v29_reconstructed.parquet"
  echo "  FRESH: using the reconstruction. Exact on a fresh extract (s284); 0.00069 board-s here."
else
  run env MODE=verify $R python -u "$S/v29_build.py"
  echo "  [THIS-EXTRACT] the archived v29 is authoritative and its sha is asserted by v64_build.py"
fi

fi
if skip; then echo "-- stage 12 skipped"; else
say "12. chain control: the UNMODIFIED chain must reproduce v59 on 0 rows   [THIS-EXTRACT]"
# V64_CONTROL=1 ends in SystemExit WITH A MESSAGE, so its exit code is 1 on SUCCESS. Under
# `set -e` that aborts the driver, so gate on the control's own output instead of its status.
# Same trap v76_build.py hit on 2026-09-13; caught here by reading the script, not by DRY.
if [ -n "$FRESH" ]; then echo "  SKIPPED on a fresh extract: v59 is a different row set by construction"
elif [ -n "$DRY" ]; then echo "  would run: V64_CONTROL=1 v64_build.py (gated on output, not status)"
else
  OUT_CTL="$(env V64_CONTROL=1 $R python -u "$S/v64_build.py" 2>&1 || true)"
  printf '%s\n' "$OUT_CTL"
  case "$OUT_CTL" in *"vs v59: 0 rows differ"*) echo "  control OK" ;;
    *) echo "  CONTROL FAILED -- the chain no longer reproduces v59. Stopping."; exit 1 ;; esac
fi

fi
if skip; then echo "-- stage 13 skipped"; else
say "13. the two shipped models -> $W/rebuilt, NEVER into submissions/"
# Until 2026-09-14 these six lines wrote straight into submissions/, so this driver would have
# overwritten the standing entry with a fresh build before anything compared them -- and four of
# the six assert no sha by default. A regeneration driver must not mutate the artefacts it
# verifies. Everything now lands in $W/rebuilt and stage VERIFY diffs it. See s289.
RB="$W/rebuilt"; mkdir -p "$RB"
run env V64_OUT="$RB/quick-boat_v64.parquet" $R python -u "$S/v64_build.py"
if [ "$V66V67" = "1" ]; then
  run env V66_BASE="$RB/quick-boat_v64.parquet" V66_OUT="$RB/quick-boat_v66.parquet" \
          $R python -u "$S/v66_build.py"
  run env V67_BASE="$RB/quick-boat_v66.parquet" V67_REF="$RB/quick-boat_v64.parquet" \
          V67_OUT="$RB/quick-boat_v67.parquet" $R python -u "$S/v67_build.py"
else
  echo "  v66/v67: using the archived parquets (RERUN_RETIRED=1 re-runs the retired scripts instead)"
  [ -n "$DRY" ] || for v in v66 v67; do cp "$S/../submissions/quick-boat_$v.parquet" "$RB/"; done
fi
run env V73_OUT="$RB/quick-boat_v73.parquet" $R python -u "$S/v73_build.py"
run env V74_SUB="$RB" V74_ALLOW_DRIFT=1 $R python -u "$S/v74_build.py"      # BEST
run env V76_SUB="$RB" V76_OUT="$RB/quick-boat_v76.parquet" SHA76= \
        $R python -u "$S/v76_build.py"                                      # ROBUST

fi

say "VERIFY: rebuilt vs shipped, row by row"
if [ -z "$DRY" ]; then
  for v in v64 v66 v67 v73 v74 v76; do
    A="$S/../submissions/quick-boat_$v.parquet"; B="$W/rebuilt/quick-boat_$v.parquet"
    [ -f "$B" ] || { echo "  $v: not rebuilt"; continue; }
    $R python - "$A" "$B" "$v" <<'PYEOF'
import sys, hashlib, numpy as np, pyarrow.parquet as pq
a_p, b_p, v = sys.argv[1:4]
rd = lambda f: pq.read_table(f).column('TAXITIME_SEC_mvt').to_numpy().astype(float)
a, b = rd(a_p), rd(b_p)
ha = hashlib.sha256(open(a_p,'rb').read()).hexdigest()[:16]
hb = hashlib.sha256(open(b_p,'rb').read()).hexdigest()[:16]
d = b - a; nz = d != 0
if not nz.any():
    print(f'  {v}: BYTE-IDENTICAL  {ha}'); raise SystemExit
q = float((d**2).sum()); n = len(a)
print(f'  {v}: {int(nz.sum()):,} of {n:,} rows differ  mean|d| {np.abs(d[nz]).mean():.2f}s  '
      f'max {np.abs(d).max():.0f}s  sum d^2 {q:.4g}  shipped {ha} rebuilt {hb}')
PYEOF
  done
  echo
  echo "  A NON-ZERO DIFF IS EXPECTED AND IS NOT A FAILURE. The window components are neural and"
  echo "  MPS attention is not deterministic across runs even under torch.manual_seed, so the"
  echo "  chain reproduces NUMERICALLY, not bit-for-bit. Measured 2026-09-14 on this extract:"
  echo "    v64/v66/v67  ~121,000 rows move, mean 1.3 s, sum d^2 0.0017% of SSE  = ~0.002 board-s"
  echo "    v73          BYTE-IDENTICAL: W_D=0.0000 multiplies the re-drawn component by zero"
  echo "    v74 / v76    661 rows, sum d^2 1338  = +0.000007 board-s. THE TWO DELIVERABLES."
  echo "  Against a 0.144 OOF-s seed floor. The v59 control at stage 12 is byte-identical. s289."
  echo "  on a FRESH extract the diff will be far larger, and that is the point: every"
  echo "  board-fitted constant was fitted to a scored set that no longer exists. MODELS.md, s280."
  echo
  echo "  submissions/ must be untouched by this run:"
  git -C "$S/.." status --porcelain submissions/ | sed 's/^/    /'
  echo "    changed: $(git -C "$S/.." status --porcelain submissions/ | wc -l | tr -d ' ') files"
else echo "  would diff each rebuilt parquet against its shipped counterpart"; fi
