#!/bin/bash
# The final phase (2026-10-08): the clean model, run once on the four scored months of final_ranking.parquet
# (January, February, June, July 2026), trained on the 2025 files only. No constant fitted to the public leaderboard
# enters it (REPRODUCE.md section 9 lists them; FINAL.md says what replaces each).
#
#   DC26_DATA   a folder exposing final_ranking.parquet as ranking.parquet, final_submitting.parquet as
#               submitting.parquet, and the twelve training_2025-*.parquet files (symlinks are fine)
#   DC26_WORK, DC26_TMP, DC26_SESSION   empty folders for this run (nothing is read from or written to any other)
#   ADSB_DAYS   the parsed adsb.lol days (scripts/adsb_stream_day.py output): all of 2025 and the four 2026 months
#               (ADSB_MONTHS="2026.01 2026.02 2026.06 2026.07" scripts/adsb_pull_all.sh; adsb_pull_year.sh for 2025)
#
#   FROM=N / ONLY=N   resume at, or run only, step N.   DRY=1   print the steps.
#
# Steps
#   1  features and the base members' inputs: regen2.sh stages 1-7 (FRESH=1, USE_REDRAWN=1)
#   2  the window members K128 and K32 (GPU): regen2.sh stages 10, 9
#   3  the chain's copy model, rebuilt: regen2.sh stage 11 (v23 -> v25 -> v29 reconstruction)
#   4  the nine bulk members: target encodings, OOF trees, ID / RQ arms, CL, then p2_members_rk.parquet
#   5  the learned five-class gate (GPU): arch/b4_rank.py -> b4_gate_rk.parquet
#   6  the 2025 window inputs of the trajectory stage, restored from artifacts/adsb_window (window side only)
#   7  the four ADS-B feature tables on four months (v102, v103, v104, full)
#   8  the full-year ADS-B members on the scored rows (p2p_v55_adsb.py MODE=rank, seeds 0-2)
#   9  the clean base (learned gate), the chain's seam rules v95 and v96 at their labelled rates (final/seam_rules.py),
#      then the trajectory stage v102 -> v114 (regen_clean.sh SKIP_BASE=1), every variant pinned to the clean chain's
#  10  v117 (the window-fitted blend weight outside LIRF, as in the submitted design), the Fable LIRF inputs on four
#      months, rules v120 -> v126 in CLEAN mode
#  11  the LIRF day-scale rule at its labelled 2025 rates -> quick-boat_final.parquet, THE FINAL FILE: the not-fitted
#      model on all four months
#  12  the competition model for comparison, quick-boat_final_with_v126.parquet: January and July from v126 (with the
#      constants fitted to the leaderboard, REPRODUCE.md section 9), February and June from step 11, then every upload check
#      (OLD_DATA: the 2026-09-04 extract v126 was built from, read only to identify v126's take-off-anchored rows)
set -euo pipefail
# The library versions every shipped file was built with (pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1, lightgbm 4.7.0,
# scikit-learn 1.9.1, xgboost 3.4.1, catboost 1.2.10, torch 2.14.1, scipy 1.18.1; the 2026-10-08 build resolved the
# same set). pyarrow 26.0.0 (2026-10-09) writes parquet bytes differently, so an unpinned run fails the byte checks.
export UV_EXCLUDE_NEWER="${UV_EXCLUDE_NEWER:-2026-10-02T00:00:00Z}"
: "${DC26_DATA:?}" "${DC26_WORK:?}" "${DC26_TMP:?}" "${DC26_SESSION:?}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"; S=scripts
export DC26_DATA DC26_WORK DC26_TMP DC26_SESSION FRESH=1 USE_REDRAWN=1
ADSB_DAYS="${ADSB_DAYS:-$DC26_WORK/adsblol/days}"; export ADSB_DAYS
# February and June have no window in the 2025 cross-fit; their offsets are the twelve-month cross-fit's mean residual
# on ordinary rows (scripts/xfit12.py; SE 0.64 and 0.60). January and July keep the window's labelled offsets.
export SHIFT_2="${SHIFT_2:-3.800}" SHIFT_6="${SHIFT_6:-2.727}"
P2="$DC26_SESSION/p2P"; C="$DC26_WORK/clean"; L="$DC26_WORK/logs"; mkdir -p "$L" "$P2"
R="uv run --quiet --with pandas --with numpy --with scikit-learn --with scipy --with pyarrow --with lightgbm python -u"
RT="uv run --quiet --with pandas --with numpy --with pyarrow --with torch python -u"
FROM="${FROM:-1}"; ONLY="${ONLY:-}"; DRY="${DRY:-}"
want() { if [ -n "$ONLY" ]; then [ "$1" = "$ONLY" ]; else [ "$1" -ge "$FROM" ]; fi; }
say() { echo "$(date -u +%H:%M:%SZ) $*"; }
run() { if [ -n "$DRY" ]; then echo "  would run: $*"; else "$@"; fi; }
N_SUB=$(uv run --quiet --with pyarrow python -c "import pyarrow.parquet as pq,sys;print(pq.read_metadata(sys.argv[1]).num_rows)" "$DC26_DATA/submitting.parquet")
say "scored rows $N_SUB   work $DC26_WORK   days $ADSB_DAYS   SHIFT_2 $SHIFT_2 SHIFT_6 $SHIFT_6"

if want 1; then say "[1] regen2.sh stages 1-7"; for s in 1 2 3 4 5 6 7; do run env ONLY=$s sh $S/regen2.sh > "$L/regen2_s$s.log" 2>&1; done; fi
if want 2; then say "[2] regen2.sh stages 10, 9 (GPU)"; for s in 10 9; do run env ONLY=$s sh $S/regen2.sh > "$L/regen2_s$s.log" 2>&1; done; fi
if want 3; then say "[3] regen2.sh stage 11, the chain's copy model"; run env ONLY=11 sh $S/regen2.sh > "$L/regen2_s11.log" 2>&1; fi
if want 4; then
  say "[4] the nine bulk members"
  [ -f "$P2/full_package.npz" ] || cp artifacts/full_package.npz "$P2/full_package.npz"   # member names and archived weights
  export P2P_OUT="$P2" P2M_OUT="$DC26_WORK/p2_members_rk.parquet"
  if [ "${SAVE_ONLY:-0}" != 1 ]; then
  run $R $S/p2p_te.py > "$L/p2p_te.log" 2>&1
  run env MODE=rank TE=oof SEEDS=0,1 $R $S/p2p_v55.py > "$L/p2p_v55.log" 2>&1
  run env MODE=rank ARM=idcat SEEDS=0,1 $R $S/p2p_arm.py > "$L/p2p_arm_id.log" 2>&1
  run env MODE=rank ARM=rwyq SEEDS=0,1 $R $S/p2p_arm.py > "$L/p2p_arm_rq.log" 2>&1
  run $R $S/p1c_load.py > "$L/p1c_load.log" 2>&1
  for sel in train rank; do for k in 1 2 3; do run env SEL=$sel $R $S/p2p_clean_feat$k.py > "$L/clean_feat_${sel}_$k.log" 2>&1; done; done
  run env MODE=rank SEEDS=0,1 $R $S/p2p_clean.py > "$L/p2p_clean.log" 2>&1
  fi
  # the assembly needs the GPU window members K32/K128 (step 2); SKIP_SAVE=1 trains the rest while they run, SAVE_ONLY=1 assembles
  [ "${SKIP_SAVE:-0}" = 1 ] || run env P2P_ALLOW_DRIFT=1 $R $S/p2p_save_members.py > "$L/p2p_save_members.log" 2>&1
fi
if want 5; then say "[5] the learned gate (GPU)"; run $RT $S/arch/b4_rank.py > "$L/b4_rank.log" 2>&1; fi
if want 6; then
  say "[6] restore the 2025 window inputs (scored-side files are rebuilt, never restored)"
  [ -n "$DRY" ] || while read -r sum f; do f="${f#./}"
    case "$f" in p2_members_rk.parquet|v15_rk_jul19.parquet|*/rank_stored_*) continue ;; esac
    src="artifacts/adsb_window/$f"; [ -f "$src" ] || src="artifacts/${f##*/}"
    [ "$(shasum -a 256 "$src" | cut -c1-64)" = "$sum" ] || { echo "archived $f corrupt"; exit 1; }
    case "$f" in session-20260922/*) dst="$DC26_SESSION/${f#session-20260922/}" ;; *) dst="$DC26_WORK/$f" ;; esac
    if [ -f "$dst" ]; then [ "$(shasum -a 256 "$dst" | cut -c1-64)" = "$sum" ] || { echo "$dst differs from the archive"; exit 1; }
    else mkdir -p "$(dirname "$dst")"; cp "$src" "$dst"; fi
  done < artifacts/adsb_window/SHA256SUMS
fi
if want 7; then
  say "[7] ADS-B feature tables on four months"
  export ADSB_SCORED_MONTHS=01,02,06,07; A="$DC26_WORK/adsblol"; mkdir -p "$A"
  run env ADSB_TIME_MATCH=0 ADSB_PARK=0 $R $S/adsb_push_features.py > "$L/adsb_feat_v102.log" 2>&1
  run env ADSB_TIME_MATCH=1 ADSB_PARK=0 $R $S/adsb_push_features.py > "$L/adsb_feat_v103.log" 2>&1
  run env ADSB_TIME_MATCH=1 ADSB_PARK=1 $R $S/adsb_push_features.py > "$L/adsb_feat_v104.log" 2>&1
  run env ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_ALL_MONTHS=1 $R $S/adsb_push_features.py > "$L/adsb_feat_full.log" 2>&1
fi
export ADSB_FEAT_V102="$DC26_WORK/adsblol/push_feat.parquet" ADSB_FEAT_V103="$DC26_WORK/adsblol/push_feat_tm.parquet" \
       ADSB_FEAT_V104="$DC26_WORK/adsblol/push_feat_tm_pk.parquet" ADSB_FEAT_FULL="$DC26_WORK/adsblol/push_feat_tm_pk_all.parquet"
if want 8; then
  say "[8] full-year ADS-B members on the scored rows"
  run env MODE=rank TE=stored MODELS=pooled,perapt SEEDS=0,1,2 NJ=8 P2P_OUT="$P2" ADSB_FEAT_ALL="$ADSB_FEAT_FULL" \
    $R $S/p2p_v55_adsb.py > "$L/adsb_members_rank.log" 2>&1
fi
export DC26_MEMBERS_RK="$DC26_WORK/p2_members_rk.parquet" ADSB_RK_IDS="$DC26_WORK/v15_rk_jul19.parquet" ADSB_RANK_DIR="$P2/trees_adsb" \
       PIN_PICKS=1 WF="${WF:-refit}" CLEAN_DIR="$C"
if want 9; then
  say "[9] clean base (WF=$WF), seam rules v95/v96, trajectory stage v102 -> v114 (picks pinned)"
  mkdir -p "$C/submissions" "$C/cand" "$C/logs"
  run $R $S/clean_base_build.py > "$L/clean_base.log" 2>&1
  [ -n "$DRY" ] || mv "$C/submissions/quick-boat_v100.parquet" "$C/submissions/quick-boat_v100_gate.parquet"
  run $R $S/final/seam_rules.py --data "$DC26_DATA" --base "$C/submissions/quick-boat_v100_gate.parquet" \
    --out "$C/submissions/quick-boat_v100.parquet" > "$L/seam_rules.log" 2>&1
  run env SKIP_BASE=1 bash $S/regen_clean.sh > "$L/regen_clean.log" 2>&1
fi
if want 10; then
  say "[10] v117, Fable on four months, v120 -> v126"
  SUB="$C/submissions"
  export DC26_SUBDIR="$SUB" DC26_CAND="$C/cand" DC26_WF="$C/wF.npz" CLEAN=1 STAGE_MLP_FILE="$C/cand/stage_mlp.npz"
  for m in 1 7; do export SHIFT_$m="$(python3 -c "import json;print(json.load(open('$C/offsets.json'))['SHIFT_$m'])")"; done
  # v117: the blend weight that moves covered rows toward layers 2 and 3, refitted on the window's covered rows outside
  # LIRF (a window-only fit). v127 left it out on a selection rule computed on the scored rows (G/SE 0.97); the
  # submitted design carries it, so it is built here (PIN_V117=1) and the rule's own verdict is printed beside it.
  run env SAVE=1 PIN_V117=1 uv run --quiet --with pandas --with pyarrow --with numpy --with lightgbm --with xgboost --with scikit-learn \
    python -u $S/adsb_price_v117.py > "$L/price_v117.log" 2>&1
  run $R $S/v117_build.py > "$L/v117.log" 2>&1
  # v120 and v121 run first: the Fable pipeline reads v121's predictions on the scored rows
  prev=117
  for v in 120 121; do run env "V${v}_BASE=$SUB/quick-boat_v$prev.parquet" "V${v}_OUT=$SUB/quick-boat_v$v.parquet" V121_REF="$SUB/quick-boat_v121.parquet" \
      $R $S/v${v}_build.py > "$L/v$v.log" 2>&1; prev=$v; done
  # The Fable pipeline trains its LIRF classifier on the 2025 window and needs that window's prediction vector, which
  # paper_final_window.py writes from the submitted chain (FABLE_WINDOW; the same file v127 and the clean room used).
  # On the scored rows it reads this run's v121.
  export FABLE_DIR="$DC26_WORK/fable_lirf" FABLE_V121="$SUB/quick-boat_v121.parquet"
  mkdir -p "$DC26_WORK/adsblol/cand"; [ -n "$DRY" ] || cp "${FABLE_WINDOW:?set FABLE_WINDOW to v121_window.npz}" "$DC26_WORK/adsblol/cand/v121_window.npz"
  run sh $S/fable_lirf/run_all.sh > "$L/fable.log" 2>&1
  export SHA_L5= SHA_L3=   # the Fable outputs are this run's, not the archived Jan/Jul ones
  for v in 122 123 124 125 126; do run env "V${v}_BASE=$SUB/quick-boat_v$prev.parquet" "V${v}_OUT=$SUB/quick-boat_v$v.parquet" V121_REF="$SUB/quick-boat_v121.parquet" \
      $R $S/v${v}_build.py > "$L/v$v.log" 2>&1; prev=$v; done
  say "four-month clean model: $SUB/quick-boat_v126.parquet"
fi
if want 11; then
  say "[11] the LIRF day-scale rule -> quick-boat_final.parquet (the not-fitted model, four months)"
  run $R $S/final/lirf_dayscale.py --data "$DC26_DATA" --base "$C/submissions/quick-boat_v126.parquet" \
    --out "$DC26_WORK/quick-boat_final.parquet" > "$L/lirf_dayscale.log" 2>&1
fi
if want 12; then
  say "[12] the competition model for comparison: Jan/Jul from v126, Feb/Jun from step 11"
  run $R $S/final/splice_final.py --data "$DC26_DATA" --old-data "${OLD_DATA:?set OLD_DATA to the 2026-09-04 extract}" \
    --uniform "$DC26_WORK/quick-boat_final.parquet" --v126 submissions/quick-boat_v126.parquet \
    --out "$DC26_WORK/quick-boat_final_with_v126.parquet" 2>&1 | tee "$L/splice.log"
fi
