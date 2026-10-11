#!/bin/bash
# s550 overnight chain: wait for the full-year pull, verify completeness, build and archive the full-year feature table,
# train the ADS-B tree members (window and scored side in parallel), price v110. Each step is logged with a timestamp.
cd "$(dirname "$0")/.."; WK="${DC26_WORK:?set DC26_WORK to a work directory}"; LOG="$WK/adsblol/overnight.log"; P2="${DC26_SESSION:-$DC26_WORK/session}/p2P"
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
PY() { uv run --quiet --with pandas --with pyarrow --with numpy --with scikit-learn --with lightgbm python "$@"; }   # lightgbm.sklearn needs scikit-learn
# FROM=train resumes at training, on the archived full-year table (the first run of 2026-09-28 stopped there)
if [ "${FROM:-}" != train ]; then
say "waiting for the full-year pull"
while pgrep -f "adsb_pull_year.sh|adsb_stream_day.py 2025" > /dev/null; do sleep 60; done
say "pull finished"; python3 scripts/adsb_pull_check.py --final >> "$LOG" 2>&1 || {
  say "incomplete: re-pulling missing days on the archived replica list"
  ADSB_PREF_FILE=artifacts/adsb_pull/PREFERRED_RELEASES_2025.txt ./scripts/adsb_pull_year.sh 3 5 >> "$LOG" 2>&1
  python3 scripts/adsb_pull_check.py --final >> "$LOG" 2>&1 || say "STILL INCOMPLETE after the re-pull (continuing; see list above)"; }
python3 scripts/adsb_day_manifest.py >> "$LOG" 2>&1
say "building the full-year feature table"
ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_ALL_MONTHS=1 PY scripts/adsb_push_features.py >> "$LOG" 2>&1 || { say "FEATURE BUILD FAILED"; exit 1; }
cp "$WK/adsblol/push_feat_tm_pk_all.parquet" artifacts/adsb_push_feat_full.parquet; say "archived artifacts/adsb_push_feat_full.parquet"
fi
say "training ADS-B members: window and scored side in parallel"
MODE=win TE=lf MODELS=pooled,perapt SEEDS=0,1,2 NJ=8 PY scripts/p2p_v55_adsb.py > "$WK/adsblol/train_win.log" 2>&1 &
MODE=rank TE=stored MODELS=pooled,perapt SEEDS=0,1,2 NJ=8 PY scripts/p2p_v55_adsb.py > "$WK/adsblol/train_rank.log" 2>&1 &
wait; tail -4 "$WK/adsblol/train_win.log" >> "$LOG"; tail -3 "$WK/adsblol/train_rank.log" >> "$LOG"
for m in win_lf rank_stored; do for t in pooled perapt; do for s in 0 1 2; do [ -f "$P2/trees_adsb/${m}_${t}_s$s.npy" ] || { say "MISSING member file ${m}_${t}_s$s"; exit 1; }; done; done; done
say "pricing v110"; PY scripts/adsb_price_v110.py > "$WK/adsblol/price_v110.log" 2>&1 || { say "PRICING FAILED"; exit 1; }
tail -14 "$WK/adsblol/price_v110.log" >> "$LOG"; say "CHAIN DONE"
