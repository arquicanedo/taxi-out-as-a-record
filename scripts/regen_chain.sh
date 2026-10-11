#!/usr/bin/env bash
# v79 -> v99: the chain between scripts/regen2.sh (raw extract -> v76) and scripts/regen_v100.sh (v99 -> v100).
#
# Each build reads its ARCHIVED base from submissions/ (sha-asserted) and writes to $DC26_WORK/rebuilt, so every
# stage is checked against its shipped file on its own. DRY=1 prints the plan. FULL=1 also retrains v99's bulk
# components (~45 min; they rebuild IDENTICALLY, colsample included, clean room 2026-09-23). Measured 2026-09-23 from the archived bases:
#   v84  every value identical to the shipped file (the bytes differ only in parquet metadata)
#   v90  162 of 344,841 rows differ, mean 14.6 s, sum d^2 7.1e4 = 0.0004 s of RMSE (its 5-seed classifier re-draws)
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"; W="${DC26_WORK:?set DC26_WORK to a work directory}"; RB="$W/rebuilt"; mkdir -p "$RB"
R="uv run --quiet --with pandas --with numpy --with scikit-learn --with scipy --with pyarrow --with lightgbm python -u"
RT="uv run --quiet --with pandas --with numpy --with pyarrow --with torch python -u"
run() { if [ -n "${DRY:-}" ]; then echo "  would run: $*"; else echo "+ $*"; "$@"; fi; }
echo "[1] the validation-window artefacts v84 reads: night matrices, the stage-A2 gate, the window nets on its matrix"
# newcols -> derived_ops -> night_prep pass their frames through ${DC26_TMP:-/tmp}; these two producers were missing
# from this driver until 2026-09-23, so step [1] could only run where an old /tmp copy survived
run $R "$S/newcols.py"
run $R "$S/derived_ops.py"
run env DC26_SPLIT=all DC26_SP="$W" $R "$S/night_prep.py"
run $R "$S/gate_a2.py"
run $RT "$S/gate_mlp.py"      # the MLP half of the stage-A2 gate; arch/harness.load() reads gate_mlp.npz
for K in 32 128; do run env K=$K $RT "$S/arch/d2_window_wide.py"
  # the window nets re-draw on every run (MPS); v84 reads the ARCHIVED ones unless USE_REDRAWN=1 (findings s514)
  A="$S/../artifacts/arch_d2_wide_va_K$K.npy"
  if [ -z "${DRY:-}" ] && [ -f "$A" ] && [ -z "${USE_REDRAWN:-}" ]; then
    mv "$W/arch_d2_wide_va_K$K.npy" "$W/arch_d2_wide_va_K${K}_redrawn.npy" 2>/dev/null || true; cp "$A" "$W/"; echo "  pinned arch_d2_wide_va_K$K.npy"
  fi
done
echo "[2] v79 -> v99, each from its archived base"
for v in 79 84 90 91 95 96; do run env "V${v}_OUT=$RB/quick-boat_v${v}.parquet" $R "$S/v${v}_build.py"; done
if [ -n "${FULL:-}" ]; then run $R "$S/v99_components.py"; fi
run env V99_OUT="$RB/quick-boat_v99.parquet" $R "$S/v99_build.py"
echo "[3] rebuilt against shipped, by value"
[ -n "${DRY:-}" ] && { echo "  would compare v79 v84 v90 v91 v95 v96 v99"; exit 0; }
for v in 79 84 90 91 95 96 99; do
  $R - "$S/../submissions/quick-boat_v$v.parquet" "$RB/quick-boat_v$v.parquet" "v$v" <<'PYEOF'
import sys, numpy as np, pyarrow.parquet as pq
a, b, v = sys.argv[1:4]
x = pq.read_table(a).column('TAXITIME_SEC_mvt').to_numpy().astype(float); y = pq.read_table(b).column('TAXITIME_SEC_mvt').to_numpy().astype(float)
d = y - x; nz = d != 0
print(f'  {v}: {int(nz.sum()):,} rows differ' + (f', mean |d| {np.abs(d[nz]).mean():.1f} s, sum d^2 {float((d*d).sum()):.3e}' if nz.any() else ', identical values'))
PYEOF
done
