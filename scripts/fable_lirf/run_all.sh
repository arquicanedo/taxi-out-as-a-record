#!/bin/sh
# The pipeline of the independent agent session (s586): the flight-plan join rule, the coin-flip classifier and the
# local-time class at LIRF, from the challenge files only. Runs the scripts from this directory with the working
# directory FABLE_DIR, where every cache and output is written; also reads DC26_DATA, DC26_WORK (for v121's window
# vector, cand/v121_window.npz) and the repository's submissions/quick-boat_v121.parquet.
#   FABLE_DIR=$DC26_WORK/fable_lirf ./scripts/fable_lirf/run_all.sh
# The 2026 outputs the builds read (l5_2026.parquet for v123, l3_2026.parquet for v125) are archived in artifacts/fable/;
# a rerun from an empty FABLE_DIR on 2026-10-03 reproduced both byte for byte (multithreaded LightGBM could move them).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "${FABLE_DIR:?set FABLE_DIR}"; cd "$FABLE_DIR"
PY="uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm python"
$PY "$HERE/build_cache.py"      # all.parquet, lirf.parquet
$PY "$HERE/window.py"           # win_lirf.parquet: LIRF window rows joined to newW; top-50 residual listings
$PY "$HERE/join_rule.py"        # proves |BLOCK - LOBT| <= 3606 s on every join-present departure (the organiser's join)
$PY "$HERE/arr_rule.py"         # arrival join is |MVT - ARVT_3| <= ~3600 s, same in 2025 and 2026
$PY "$HERE/dest_arr.py"         # dep_destarr.parquet: LIRF departure matched to its own arrival row at the ten airports
$PY "$HERE/classes.py"          # lirf_train_cls.parquet: copy / roll / tz / real / other classes
$PY "$HERE/dayfeat.py"          # lirf_dep_feat.parquet: day-level disruption and carrier-day DLA features
$PY "$HERE/clip.py"             # L1 lever, estate-wide, plus 2026 guaranteed floor
$PY "$HERE/recover.py"          # L2 lever (destination-arrival recovery)
$PY "$HERE/tz_lever.py"         # L3 lever (timezone-offset cell) and cell B tables
$PY "$HERE/roll_rule.py"        # 12h date-roll rule and L4 cap
$PY "$HERE/cf_clf.py"           # coin-flip copy classifier, leave-one-month-out; cf_clf_win/2026.parquet
$PY "$HERE/cf_clf2.py"          # win_cf_pred.parquet; blend variants
$PY "$HERE/cf_clf3.py"          # band-restricted blends
$PY "$HERE/jp_clf2.py"; $PY "$HERE/jp_clf3.py"   # join-present classifier (closed, negative)
$PY "$HERE/bayes_cf.py"; $PY "$HERE/stand_rot.py"; $PY "$HERE/arr2026.py"; $PY "$HERE/arr_tz.py"; $PY "$HERE/far_check.py"   # ruled-out / supporting
$PY "$HERE/levers.py"           # FINAL: L1-L5 sequential on the LIRF window with bootstrap; 2026 application -> rank_levers_lirf.parquet
$PY "$HERE/l5_export.py"     # v123's input: l5_2026.parquet
$PY "$HERE/l3_export.py"     # v125's input: l3_2026.parquet
