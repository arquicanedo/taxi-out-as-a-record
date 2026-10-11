#!/bin/bash
# s548: pull and parse the ten 2025 months outside the window (Feb-Jun, Aug-Dec) from adsb.lol, STREAMED straight into
# scripts/adsb_stream_day.py (no tar and no extract ever written; only the parsed per-day points, ~5 MB a day). Each day
# uses the replica adsb.lol names in PREFERRED_RELEASES.txt (including -0tmp tags); if that replica fails (truncated or
# corrupt), the other of prod-0 / staging-0 is tried. Idempotent: days with a .done marker are skipped.
#   ./scripts/adsb_pull_year.sh [parallel] [workers]
P="${1:-3}"; W="${2:-5}"; HERE="$(cd "$(dirname "$0")" && pwd)"; OUT="${DC26_WORK:?set DC26_WORK to a work directory}/adsblol/days"; mkdir -p "$OUT"
PREF="${DC26_WORK:?set DC26_WORK to a work directory}/adsblol/PREFERRED_RELEASES_2025.txt"
if [ -n "${ADSB_PREF_FILE:-}" ]; then cp "$ADSB_PREF_FILE" "$PREF"   # reproduce: the archived list, artifacts/adsb_pull/PREFERRED_RELEASES_2025.txt
else gh api "repos/adsblol/globe_history_2025/contents/PREFERRED_RELEASES.txt" --jq '.content' | base64 -d > "$PREF"; fi
export HERE W PREF
one() {
  d="$1"; dd="${d//./}"; [ -f "$OUT/d$dd.done" ] && return 0
  urls=$(grep "/v$d-planes-readsb-" "$PREF" | head -1)
  if [ -n "$urls" ] && ADSB_URLS="$urls" uv run --quiet --with pandas --with pyarrow python "$HERE/adsb_stream_day.py" "$d" "$W" 2>&1 | tail -1; then
    [ -f "$OUT/d$dd.done" ] && return 0
  fi
  for fl in prod staging; do
    ADSB_FLAVOR=$fl ADSB_OUT_PROD=1 uv run --quiet --with pandas --with pyarrow python "$HERE/adsb_stream_day.py" "$d" "$W" 2>&1 | tail -1
    [ -f "$OUT/d$dd.done" ] && return 0
  done
  echo "$d: FAILED on every replica"
}
export -f one; export OUT
python3 - <<'PY' | xargs -P "$P" -I{} bash -c 'one {}'
import datetime as dt, os
d = dt.date(2025, 1, 1); days = []
while d.year == 2025:
    if d.month not in (1, 7): days.append(d.strftime('%Y.%m.%d'))
    d += dt.timedelta(days=1)
print('\n'.join(reversed(days) if os.environ.get('REVERSE') == '1' else days))   # REVERSE=1: a second driver from Dec 31 back
PY
