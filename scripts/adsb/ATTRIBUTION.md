# adsb.lol data: attribution and licence

Contains information from adsb.lol, made available under the Open Database License (ODbL) 1.0,
https://opendatacommons.org/licenses/odbl/1-0/. (c) adsb.lol contributors.

Upstream: the daily releases of the adsb.lol receiver network's traces, https://github.com/adsblol/globe_history_2025
and https://github.com/adsblol/globe_history_2026.

What this repository derives from them, and how:

- `scripts/adsb_stream_day.py` (driven by `adsb_pull_all.sh` and `adsb_pull_year.sh`) streams each day's archive and
  keeps the ground samples, take-offs and landings near the ten airports in `$DC26_WORK/adsblol/days/`. No raw
  archive is stored.
- `scripts/adsb_push_features.py` builds the per-departure pushback feature tables from those days (five variants,
  `REPRODUCE.md` section 11).

Both are derivative databases of adsb.lol under the ODbL. They are not redistributed here: the scripts above are the
method of making them (ODbL section 4.6(b)), and `artifacts/adsb_push_feat.content.sha256` checks a rebuild. The
submissions from v102 to v114, v116, and v117 to v126 (the final model) are Produced Works from them and carry this
notice through `README.md` and `EXTERNAL-DATA.md` section 3.
