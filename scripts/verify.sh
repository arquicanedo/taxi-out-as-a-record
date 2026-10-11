#!/bin/bash
# The two-minute check, with only the challenge data (DC26_DATA): every shipped submission and archived input against its
# SHA-256, then the builds that need nothing else, each run twice: FIRE=0 must write its base back byte for byte, and
# FIRE=1 must write its own submission byte for byte: v100, and v120 to v126 from the archived members and seeds.
# Nothing is written inside the repository.
#   DC26_DATA=/path/to/extract ./scripts/verify.sh
set -uo pipefail
# The library versions every shipped file was built with (pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1, lightgbm 4.7.0,
# scikit-learn 1.9.1, xgboost 3.4.1, catboost 1.2.10, torch 2.14.1, scipy 1.18.1; the 2026-10-08 build resolved the
# same set). pyarrow 26.0.0 (2026-10-09) writes parquet bytes differently, so an unpinned run fails the byte checks.
export UV_EXCLUDE_NEWER="${UV_EXCLUDE_NEWER:-2026-10-02T00:00:00Z}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
: "${DC26_DATA:?set DC26_DATA to the challenge extract directory}"
T="$(mktemp -d)"; export DC26_WORK="$T/work" DC26_SESSION="$T/work/session"; mkdir -p "$DC26_WORK"
PY=(uv run --quiet --with pandas --with pyarrow --with numpy python)
fail=0
check() { if "$@" > "$T/out" 2>&1; then echo "PASS  $LABEL"; else echo "FAIL  $LABEL"; tail -3 "$T/out" | sed 's/^/      /'; fail=1; fi; }
LABEL="submissions/SHA256SUMS (every shipped submission)"; check sh -c 'cd submissions && shasum -a 256 -c --quiet SHA256SUMS'
for d in artifacts/seeds artifacts/fable; do LABEL="$d/SHA256SUMS"; check sh -c "cd $d && shasum -a 256 -c --quiet SHA256SUMS"; done
LABEL="v100  FIRE=0 -> v99"; check env FIRE=0 V100_MEMBERS=artifacts/p2_members_rk.parquet "${PY[@]}" scripts/v100_build.py
LABEL="v100  FIRE=1 -> its own sha256"; check env V100_MEMBERS=artifacts/p2_members_rk.parquet V100_OUT="$T/v100.parquet" "${PY[@]}" scripts/v100_build.py
LABEL="restore the archived members and seeds into a scratch directory"; check sh -c 'RESTORE_ONLY=1 scripts/regen_adsb.sh && RESTORE_ONLY=1 scripts/regen_seeds.sh'
LABEL="the scored-side members with every archived seed (v118's)"; check env SEEDS_OUT="$DC26_WORK/seeds/chain" "${PY[@]}" scripts/seeds_members.py
for v in 120 121 122 123 124 125 126; do
  LABEL="v$v  FIRE=0 -> its base"; check env FIRE=0 "${PY[@]}" "scripts/v${v}_build.py"
  LABEL="v$v  FIRE=1 -> its own sha256"; check env "V${v}_OUT=$T/v$v.parquet" "${PY[@]}" "scripts/v${v}_build.py"
done
rm -rf "$T"
[ "$fail" = 0 ] && echo "ALL PASS" || { echo "SOME CHECKS FAILED"; exit 1; }
