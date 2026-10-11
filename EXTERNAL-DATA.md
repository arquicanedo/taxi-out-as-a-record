# External data used by the shipped models

Prize eligibility requires that "all used external datasets are openly accessible/usable and
documented" and that "all additional datasets used are openly available under an open source
license" (`eligibility.qmd`). This page is that documentation. **It states each licence position
as the upstream states it, including where that is ambiguous, rather than as we would prefer.**

Three external sources reach the final submission, `quick-boat_v126.parquet`: the PRU ATFM data and OPDI reach
every model from v55 on, including the base model v100, and adsb.lol reaches every model from v102 on. One more,
IEM ASOS weather, reaches only v105, a submission that lost and is not in v126's lineage. Everything else in this
repository is exploratory and reaches no shipped path.

---

## 1. EUROCONTROL PRU: daily ATFM regulation data. IN THE SHIPPED MODELS.

```
  shipped    reference/pru/ATFM_Slot_Adherence.parquet, Airport_Arrival_ATFM_Delay.parquet   (~10 MB, in this repo)
  download   uv run --quiet --with pandas --with pyarrow --with openpyxl python scripts/pull_pru.py
  upstream   https://www.eurocontrol.int/performance/data/download/xls/
  reaches    v17_features.py:59 -> the eight atfm_* features -> FB -> every tree component
```

Supplies eight features (`atfm_reg_share`, `atfm_late_share`, `atfm_early_share`, `atfm_in_share`,
`atfm_cause_S/W/G`, `atfm_dly_per_arr`), worth **-1.00 s jointly** on validation, and they are in the
shipped `v18`/`v19` feature rungs.

**The two parquets are shipped as a pinned snapshot**, release V.1.0 of 2026-08-14 (the workbooks' META sheet),
downloaded 2026-09-01. PRU republishes monthly and revises, so a fresh `pull_pru.py` fetches a newer release, and
the features built from it will differ. The script writes the parquets from each workbook's DATA sheet;
that conversion reproduces the shipped snapshot exactly (`DataFrame.equals`, checked 2026-09-23). It keeps an
existing parquet unless `PRU_OVERWRITE=1` is set. The snapshot runs to 2026-07-31, so it already covers February and
June 2026, the final phase's added months. The final model uses it unchanged: a refresh would also revise 2025.

**Licence: open, published by EUROCONTROL's Performance Review Unit on its public performance
data portal, no registration or authentication.** This is the same organisation running the
challenge. We are not aware of any restriction on reuse.

> Note for our own record: `CLAUDE.md` listed "PRU ATFM" among *measured negative* results for
> some time. That was wrong and is corrected. The negative refers to PRU **published
> airport-month taxi means** (19.8 s of bias against our own 9.4 s), a different product that
> is not used.

## 2. OPDI (Open Performance Data Initiative): airframe identity. IN THE SHIPPED MODELS.

```
  download   uv run --quiet --with pandas --with pyarrow python scripts/pull_opdi.py
  files      flight_list_{202501..202512,202601,202602,202607}.parquet   15 files, ~591 MB
             + flight_list_{202603,202606}.parquet for the final phase (2026-10-08): June, and March as February's lookahead
  upstream   https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/
  reaches    v55_trees.py -> v55_trees.parquet -> v64_build.py:87 -> v74 -> ... -> v100 (members L55, P55, L55o, P55o)
```

**Checked 2026-09-23: a fresh `pull_opdi.py` downloads all fifteen files byte-identical to the ones the model was
built from**, and `opdi_features.py` rebuilds `opdi_airframe.parquet` from them identically.

Supplies `icao24`, the airframe identity the challenge data does not carry, joined on
`(flt_id, dof)` against `CALLSIGN_flt` at 84.7%; 90.6% of 2025 departures get a registration and
89.0% chain to a previous arrival of the same airframe. Used for per-airframe turnaround
features.

**Measured contribution: +0.0533 board-s** (v54 267.9129 -> v55 267.8596). Validation projected +0.96
val-s, which at the measured 3.927x transfer factor is +0.24, so validation over-predicted it
4.5x. It is a small contribution, and it is retained because removing it now would require
refitting the bulk blend: `lgbm_opdi` and `perapt_opdi` became the bulk base for v56 through v74.

### Licence position, stated exactly

**The OPDI portal's About page says two things that do not agree, and carries no LICENSE file,
no CC BY or ODbL statement, and no terms-of-use page:**

> "... may be freely reused provided the source is attributed"

> "The focus of the OPDI is on open data, in particular open access and open use of data **for
> non-commercial purposes**."

The first is attribution-only. The second is non-commercial. **Non-commercial does not satisfy
the Open Source Definition**, which forbids restricting fields of endeavour.

**What we did about it.** We asked the organisers in public on 2026-09-06, in these words:

> "The OPDI portal's About page states both that the data may be freely reused with attribution,
> and that the focus is on open access and open use for non-commercial purposes. There is no
> LICENSE file or terms-of-use page. Which governs, for the eligibility requirement that
> additional datasets be openly available under an open source licence?"

**That question is unanswered.** Separately and on the same day, an organiser addressed
admissibility directly:

> "OPDI milestones for now do not cover off-block, but you can eventually use it if you find a
> purpose for the challenge."
> (E. Spinielli, 2026-09-06)

So **use of OPDI in this challenge was explicitly permitted by an organiser**, while the licence
wording itself remains ambiguous upstream. On 2026-10-04, asked in the competition channel whether OPDI counts as open
data for prize eligibility and which licence to cite, an organiser answered: "yes open" (E. Spinielli). OPDI is published by the EUROCONTROL PRU, the same
group that runs this challenge and that did formally licence the DC2024 challenge dataset under
CC BY 4.0 with a 4TU DOI. We record both facts and resolve neither in our favour.

If the ambiguity is judged disqualifying, the remedy is to refit the bulk blend without
`lgbm_opdi`/`perapt_opdi`, at a cost of about 0.05 board-s plus a chain rebuild.

---

## 3. adsb.lol `globe_history`: raw ADS-B surface tracks. IN THE FINAL MODEL.

**What it supplies.** For each departure, the aircraft's own ground track at the airport: when it first moved
after a parked dwell (or was first seen moving), how far it went, where it had last parked, and when it took off.
The trajectory stage (v102 to v114, `REPRODUCE.md` section 11, and the final model built on it, v117 to v126,
section 13) learns when to trust that pushback against the airport's recorded off-block.

**Source.** adsb.lol publishes each day's receiver network traces as GitHub releases, `adsblol/globe_history_2025`
and `adsblol/globe_history_2026`. We used all 365 days of 2025 and the 62 January and July days of 2026, and for the final model the 58 February and
June days of 2026 (`ADSB_MONTHS="2026.02 2026.06" scripts/adsb_pull_all.sh`; pulled 2026-10-08, one day, 2026-06-29,
retried after an empty download). Which
replica we read for each day is archived in `artifacts/adsb_pull/PREFERRED_RELEASES_2025.txt`, and the per-day
counts in `artifacts/adsb_pull/day_manifest.tsv`. Thanks to SoK, whose post in the challenge's public channel that
their team used adsb.lol prompted us to pull the archive in full.

**Licence: Open Database License (ODbL) 1.0**, https://opendatacommons.org/licenses/odbl/1-0/. Open-Definition
conformant, commercial use permitted, attribution and share-alike required.

> Contains information from adsb.lol, made available under the ODbL 1.0. (c) adsb.lol contributors.

**How this release complies.** The per-day parses and the five pushback feature tables are derivative databases
of adsb.lol, and our submissions are Produced Works from them. ODbL section 4.6 lets whoever publicly uses a
derivative database or a produced work offer either the derivative database or "the method of making the
alterations" to the database. We offer the method: `scripts/adsb_stream_day.py`, `adsb_pull_all.sh`,
`adsb_pull_year.sh` and `adsb_push_features.py` rebuild every table from the public releases, and
`artifacts/adsb_push_feat.content.sha256` lets anyone check the result. The notice above is carried here, in
`README.md` and in `scripts/adsb/ATTRIBUTION.md`, and the paper cites the archive. The derived tables are not in this repository. The
archived member predictions in `artifacts/adsb_window/` are our model outputs, not extracts of the database.

**Coverage, which depends on receivers.** Share of departures with a detected pushback, January and July:

```
                 EDDF  EDDM  EGLL  EHAM  LEBL  LEMD  LFPG  LIRF  LSZH  LTFM
  2025 window    89.1  61.8  38.0  98.0  93.9  15.3   2.1  53.1  93.1   0.0
  2026 scored    60.6  96.9  89.8  96.9  94.7  78.0   7.3  29.8  95.0   0.0
```

## 4. IEM ASOS weather (METAR): v105 only

`scripts/pull_metar.py` pulls hourly airport observations from the Iowa Environmental Mesonet's ASOS archive
(public domain); `artifacts/metar_iem_asos.parquet` is the archived pull. Only `scripts/adsb_price_v105.py` reads
it. v105 lost on the board and is not in v126's lineage; it is shipped so every submission can be rebuilt.

## Not used by the shipped models

Listed so a reviewer does not have to determine it from the code, and so nothing looks concealed.

| source | licence | status |
|---|---|---|
| **OPDI `flight_events` / `measurements`** | as above | exploratory. Four files, 801 MB, from exploratory turnaround-veto work, which died at **-68.9 board-s**. `pull_opdi.py` deliberately does not fetch them |
| **OpenStreetMap airport geometry** | ODbL 1.0 | exploratory. Geometry bounded at a ~6 s ceiling |
| **METAR / weather in the bulk** | NOAA / Iowa State, public domain | exploratory. Worth +0.65%, closed. (Weather in a trajectory stage is section 4.) |
| **PRU published airport-month taxi means** | open, as above | tested and rejected: 19.8 s of bias against our own 9.4 s |

None of the above is read by any script in the production chain (MANIFEST.md section 1).

---

## Code licence

This repository is **GNU GPL v3**, full text in `LICENSE`, as required by `eligibility.qmd`.
