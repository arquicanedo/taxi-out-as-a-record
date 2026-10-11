#!/bin/bash
# Rebuild v102-v114 (the ADS-B stages, the weather stage and the full-year ADS-B members, findings s516-s559) from the
# repository and assert their sha256.
#   ./scripts/regen_adsb.sh            from the feature tables (archived, or regenerated once with FROM_DAYS=1) and the
#                                      member predictions in artifacts/ (minutes)
#   FROM=110 ./scripts/regen_adsb.sh   only v110, v112 and v114 (needs v108's candidate, $DC26_WORK/adsblol/cand/v108_adsb.npz)
#   FROM_DAYS=1 ./scripts/regen_adsb.sh   also regenerate the feature tables from the parsed adsb.lol days in
#                                          $DC26_WORK/adsblol/days and check them against the archived tables, or, where
#                                          a table is not archived (the public release ships none: ODbL), against
#                                          artifacts/adsb_push_feat.content.sha256, then place it in artifacts/
#   PULL=1 FROM_DAYS=1 ./scripts/regen_adsb.sh   first pull and parse the 427 days from adsb.lol (the 124 Jan+Jul days of
#                                          2025 and 2026, then the ten other 2025 months; streamed, network-bound, hours)
# v110/v112 RETRAIN from scratch (hours; LightGBM multithreaded, so predictions may drift and the builds then need
# V110_ALLOW_DRIFT=1 / V112_ALLOW_DRIFT=1): the full year of days via adsb_pull_year.sh (streamed, nothing raw stored),
# then FROM=train scripts/overnight_v110.sh (full-year members) and scripts/oof_folds_v112.sh (month-fold OOF members), which
# overwrite the session directory's p2P/trees_adsb ($DC26_SESSION); the archived copies are in artifacts/adsb_window.
# Writes only to a temp directory; submissions/ is never touched. adsb.lol data: ODbL 1.0, scripts/adsb/ATTRIBUTION.md.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
WK="${DC26_WORK:?set DC26_WORK to a work directory}"; SS="${DC26_SESSION:-$DC26_WORK/session}"
T="$(mktemp -d)"; echo "work dir $T"
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python "$@"; }
# v100-chain window inputs the ADS-B pricers read (artifacts/adsb_window, 26 files): restore any that are missing from the
# work directory, and refuse to run if one that is present differs from the archived copy.
# A file absent from artifacts/adsb_window is taken from artifacts/ under its base name (the release keeps one copy of the
# v100 members and weights there), and must still match the recorded hash.
while read -r sum f; do f="${f#./}"; src="artifacts/adsb_window/$f"; [ -f "$src" ] || src="artifacts/${f##*/}"
  [ -f "$src" ] && [ "$(shasum -a 256 "$src" | cut -c1-64)" = "$sum" ] || { echo "archived window input $f missing or corrupt"; exit 1; }
  case "$f" in session-20260922/*) dst="$SS/${f#session-20260922/}" ;; *) dst="$WK/$f" ;; esac
  if [ ! -f "$dst" ]; then mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; echo "  restored $f"
  elif [ "$(shasum -a 256 "$dst" | cut -c1-64)" != "$sum" ]; then echo "work-dir $f differs from artifacts/adsb_window"; exit 1; fi
done < artifacts/adsb_window/SHA256SUMS
[ "${RESTORE_ONLY:-0}" = 1 ] && { echo "  archived window inputs in place"; exit 0; }
for f in "$WK/v15_rk_jul19.parquet" "$WK/b4ab_ctx.npz" "$WK/b4ab_P_u.npy" "$SS/trackA/window_ids.npz"; do
  [ -f "$f" ] || { echo "missing $f (v15_rk_jul19.parquet: ONLY=1 then ONLY=2 regen2.sh, then copy it from $DC26_TMP; REPRODUCE.md section 11)"; exit 1; }; done
if [ "${PULL:-0}" = 1 ]; then ./scripts/adsb_pull_all.sh 3 5; ADSB_PREF_FILE=artifacts/adsb_pull/PREFERRED_RELEASES_2025.txt ./scripts/adsb_pull_year.sh 3 5; fi
for v in v102 v103 v104 v105 full; do [ -f "artifacts/adsb_push_feat_$v.parquet" ] || [ "${FROM_DAYS:-0}" = 1 ] || {
  echo "artifacts/adsb_push_feat_$v.parquet is not in this tree: regenerate it with FROM_DAYS=1 (and PULL=1 to download the days)"; exit 1; }; done
if [ "${FROM_DAYS:-0}" = 1 ]; then
  for v in 102 103 104 105 full; do
    case $v in
      102) ADSB_TIME_MATCH=0 ADSB_PARK=0 PY scripts/adsb_push_features.py > /dev/null; new=$WK/adsblol/push_feat.parquet ;;
      103) ADSB_TIME_MATCH=1 ADSB_PARK=0 PY scripts/adsb_push_features.py > /dev/null; new=$WK/adsblol/push_feat_tm.parquet ;;
      104) ADSB_TIME_MATCH=1 ADSB_PARK=1 PY scripts/adsb_push_features.py > /dev/null; new=$WK/adsblol/push_feat_tm_pk.parquet ;;
      105) ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_XY=1 PY scripts/adsb_push_features.py > /dev/null; new=$WK/adsblol/push_feat_tm_pk_xy.parquet ;;
      full) ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_ALL_MONTHS=1 PY scripts/adsb_push_features.py > /dev/null; new=$WK/adsblol/push_feat_tm_pk_all.parquet ;;
    esac
    ref=artifacts/adsb_push_feat_v$v.parquet; [ "$v" = full ] && ref=artifacts/adsb_push_feat_full.parquet
    if [ ! -f "$ref" ]; then PY scripts/adsb_feat_hash.py --place "$v" "$new" "$ref" || { echo "feature table $v differs from the archive"; exit 1; }
      continue; fi
    PY - "$ref" "$new" <<'PY'
import sys, pandas as pd
a = pd.read_parquet(sys.argv[1]).set_index('MVT_ID_mvt').sort_index(); b = pd.read_parquet(sys.argv[2]).set_index('MVT_ID_mvt').sort_index()
assert a.index.equals(b.index), 'row sets differ'
bad = [c for c in a.columns if not a[c].equals(b[c])]; assert not bad, f'columns differ: {bad}'
print(f'  feature table regenerated == {sys.argv[1]} ({len(a):,} rows)')
PY
  done
fi
if [ "${FROM:-102}" -le 109 ]; then
ADSB_FEAT=artifacts/adsb_push_feat_v102.parquet PY scripts/adsb_price.py > "$T/price_v102.log"
FIRE=0 PY scripts/v102_build.py | tail -1
V102_OUT="$T/v102.parquet" PY scripts/v102_build.py | tail -1
ADSB_FEAT=artifacts/adsb_push_feat_v103.parquet PY scripts/adsb_price_v103.py > "$T/price_v103.log"
FIRE=0 PY scripts/v103_build.py | tail -1
V103_OUT="$T/v103.parquet" PY scripts/v103_build.py | tail -1
ADSB_FEAT=artifacts/adsb_push_feat_v104.parquet PY scripts/adsb_price_v104.py > "$T/price_v104.log"
FIRE=0 PY scripts/v104_build.py | tail -1
V104_OUT="$T/v104.parquet" PY scripts/v104_build.py | tail -1
PY scripts/adsb_price_v105.py > "$T/price_v105.log"
FIRE=0 PY scripts/v105_build.py | tail -1
V105_OUT="$T/v105.parquet" PY scripts/v105_build.py | tail -1
PY scripts/adsb_price_v106.py > "$T/price_v106.log"
FIRE=0 PY scripts/v106_build.py | tail -1
V106_OUT="$T/v106.parquet" PY scripts/v106_build.py | tail -1
uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_price_v107.py > "$T/price_v107.log"
FIRE=0 PY scripts/v107_build.py | tail -1
V107_OUT="$T/v107.parquet" PY scripts/v107_build.py | tail -1
if [ "${FROM_DAYS:-0}" = 1 ]; then   # the MLP stage: retrain on CPU (deterministic) and require the archived predictions
  PY scripts/stage_mlp_export.py > /dev/null && STAGE_MLP_OUT="$T/mlp.npz" uv run --quiet --with numpy --with torch python scripts/stage_mlp.py > /dev/null
  uv run --quiet --with numpy python -c "import numpy as np,sys; a=np.load('$T/mlp.npz'); b=np.load('artifacts/v108_stage_mlp.npz'); sys.exit(0 if all(np.array_equal(a[k],b[k]) for k in ('dM','dMr')) else 1)" \
    && echo "  MLP stage retrained == artifacts/v108_stage_mlp.npz" || { echo "MLP stage retrain differs"; exit 1; }
fi
uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost python scripts/adsb_price_v108.py > "$T/price_v108.log"
FIRE=0 PY scripts/v108_build.py | tail -1
V108_OUT="$T/v108.parquet" PY scripts/v108_build.py | tail -1
uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn --with catboost python scripts/adsb_price_v109.py > "$T/price_v109.log"
FIRE=0 PY scripts/v109_build.py | tail -1
V109_OUT="$T/v109.parquet" PY scripts/v109_build.py | tail -1
fi
# v110 (s550-s552): the pre-registered pricer (its two variants lost; logged for the record), then the corrected C-alpha family
PY scripts/adsb_price_v110.py > "$T/price_v110_prereg.log"
SAVE=Cfam PY scripts/adsb_v110_diag.py > "$T/price_v110.log"; grep -q 'saved cand/v110_adsb.npz as variant "C alpha"' "$T/price_v110.log"
FIRE=0 PY scripts/v110_build.py | tail -1
V110_OUT="$T/v110.parquet" PY scripts/v110_build.py | tail -1
# v112 (s554-s556): the full-year stage on the month-fold out-of-fold members
SAVE=1 uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_stage_full.py > "$T/price_v112.log"
grep -q 'saved cand/v112_adsb.npz as "C2 alpha"' "$T/price_v112.log"
FIRE=0 PY scripts/v112_build.py | tail -1
V112_OUT="$T/v112.parquet" PY scripts/v112_build.py | tail -1
# v114 (s558, s559): v112 with LIRF left at v108
SAVE=1 uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn python scripts/adsb_price_v114.py > "$T/price_v114.log"
grep -q 'saved cand/v114_adsb.npz as "no LIRF"' "$T/price_v114.log"
FIRE=0 PY scripts/v114_build.py | tail -1
V114_OUT="$T/v114.parquet" PY scripts/v114_build.py | tail -1
echo "REGEN OK: v${FROM:-102}-v114 reproduce byte-identically (logs in $T)"
