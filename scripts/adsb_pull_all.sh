#!/bin/bash
# Pull and parse the 124 adsb.lol days the ADS-B pushback feature needs: Jan+Jul 2025 (labelled, the selection
# window) then Jan+Jul 2026 (scored), months interleaved so a partial pull already spans both. Idempotent: days with a
# .done marker are skipped, so re-running retries only failures.   ./scripts/adsb_pull_all.sh [parallel] [workers]
# ADSB_MONTHS="2026.02 2026.06" pulls those months instead (the final phase's Feb+Jun 2026); days past a month's end are
# dropped by date validation, so 31 candidates per month are safe.
P="${1:-3}"; W="${2:-5}"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${DC26_WORK:?set DC26_WORK to a work directory}/adsblol/days"; mkdir -p "$OUT"
days() { if [ -n "$ADSB_MONTHS" ]; then for i in $(seq -w 1 31); do for m in $ADSB_MONTHS; do echo "$m.$i"; done; done
         else for y in 2025 2026; do for i in $(seq -w 1 31); do echo "$y.07.$i"; echo "$y.01.$i"; done; done; fi; }
days | while read d; do date -j -f %Y.%m.%d "$d" +%Y.%m.%d 2>/dev/null | grep -qx "$d" && echo "$d"; done \
  | while read d; do [ -f "$OUT/d${d//./}.done" ] || echo "$d"; done \
  | xargs -P "$P" -I{} sh -c "uv run --quiet --with pandas --with pyarrow python '$HERE/adsb_stream_day.py' {} $W 2>&1 | tail -1"
