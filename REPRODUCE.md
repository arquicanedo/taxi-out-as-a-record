# What actually ships, and how to reproduce it

This document exists because the shipped model's configuration is **not** contained in the scripts
that appear to define it. It is split across script defaults, environment-variable defaults set in
other files, and a filename convention that distinguishes the live data extract from a superseded
one sitting beside it under a similar name. Every one of those four has, at some point, caused a
measurement in this project to be taken against the wrong baseline.

A reader who runs the obvious scripts with their obvious arguments gets a **materially different
model** from the one that produced our leaderboard score. This document is the corrective.

Everything below was verified by rebuilding the component and reproducing the shipped artefact
exactly, not by reading the code. Where a check was run, its result is quoted.

---

## 0. Environment

Every script runs as `uv run --quiet --with <libraries> python <script>`, the libraries unversioned. Set

```
export UV_EXCLUDE_NEWER=2026-10-02T00:00:00Z
```

and each call resolves to the versions in `requirements.txt` (`requirements-lock.txt` pins all 55 packages, transitive
ones included, from the same date; `.python-version` asks uv for Python 3.13), which are the ones that rebuilt every submission from
v100 to v121 byte for byte in the clean room of 2026-10-02 (and v122 to v126 on 2026-10-03/04): Python 3.13.13,
pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, scikit-learn 1.9.1,
SciPy 1.18.1, FLAML 2.7.0, PyTorch 2.14.1, with uv 0.11.7 on macOS 26 (Apple M5 Pro, 18 cores). LightGBM 4.7.0 and
CatBoost 1.2.10 are the only versions this project ever installed. A different LightGBM, XGBoost or CatBoost, or a
different thread count, can change the bytes of a retrained stage; every build asserts its sha256 and stops if so. The
neural components (MLPs and window transformers, PyTorch on Apple Metal) re-draw on every run whatever the versions,
which is why their outputs are archived (section 10).

---

## 1. Data

```
$DC26_DATA   (required)                             the 2026-09-04 REISSUE
  training_*.parquet   12 files   2,085,047 labelled departures + 2,082,749 labelled arrivals
  ranking.parquet      689,534 rows = 344,841 scored departures + 344,693 LABELLED arrivals
  submitting.parquet   344,841 rows, defines the submission row ORDER
```

**Trap.** `09-01-2026-09-35-06_files_list/` is the PRE-REISSUE extract. Its `ranking.parquet` holds
a dead 215,876-row scored set. The twelve `training_*.parquet` files are **identical between the
two extracts**, so a script reading only training data returns the same answer from either and
nothing fails loudly. Anything touching `ranking.parquet` must use the reissue.

Arrivals carry `BLOCK_TIME_UTC_mvt` and the target in BOTH files, including the scored one. Only
departures are blanked, and only `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`.

Verified: `y == MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` exactly on all 2,085,047 labelled
departures, `max |diff| = 0 s`.

## 2. Features

```
scripts/features.py        the shared builder
scripts/v15_augment.py     writes v15_tr / v15_rk
```

**Trap.** The live feature files are the `_jul19` variants:

```
$DC26_WORK/v15_tr_jul19.parquet    2,084,659 rows   training
$DC26_WORK/v15_rk_jul19.parquet      344,841 rows   ranking   <- REQUIRED
$DC26_WORK/v15_cols_jul19.json                      F_BULK, F, CATS
```

`v15_rk.parquet` (no suffix) holds **215,876 rows** and is the dead extract. `v15_tr.parquet` and
`v15_tr_jul19.parquet` are both 2,084,659 rows, so the training side is safe either way and only
the ranking side silently differs.

## 3. Bulk components: `scripts/v15_trees.py`

Two gradient-boosted models with **different hyperparameters**, only one of which is in the script.

```
POOLED        learning_rate .06   n_estimators 2000   num_leaves 255   min_child_samples  20
PER-AIRPORT   learning_rate .06   n_estimators  600   num_leaves 127   min_child_samples 100
both          reg_lambda 1.0, seeds (0,1,2) arithmetic mean, categorical_feature=CATS
              target = y - naive  (naive = MVT - AOBT_3, falling back to apt_hr_naive)
              trained on NON-EXTREME rows only:  trn = tr[~(y >= 7200)]
```

**Trap.** `v15_trees.py` literally reads `600/127/100` for both. The pooled model is overridden to
`2000/255/20` by `DC26_POOLED_CAP`, whose value is set in **`scripts/v23_build.py:28`**, a
different file. (`scripts/regen2.sh` has exported it, with `DC26_RECIPE=v19`, since 2026-09-23; before that a
from-scratch run of the driver built a different bulk model.) The override must reach the pooled model ONLY; the per-airport loop keeps the
script defaults.

Verified: rebuilding both models with the settings above reproduces `v15_trees_jul19.parquet`
exactly: `lgbm` and `perapt` both `max |diff| 0.000 s`, correlation `1.000000`.

## 4. Blend - `scripts/v64_build.py`, with `v73_build.py` overriding its weights

`v64` fits nothing. It blends precomputed component parquets keyed on `MVT_ID_mvt`.

**Trap, and it is one level deeper than it looks.** `v64_build.py`'s own environment defaults are
`W_LGB 0.2473 / W_MLP 0.0000 / W_PA 0.2217 / W_D 0.0849 / W_D2 0.4462 / W_K128 0.0000`. **Those are
not the shipped weights.** `v73_build.py:19-21` invokes the same script with a different
environment, and `v74_build.py:29` takes its bulk delta as `v73 - v64`, so every submission from
v74 onward carries v73's blend, not v64's:

```
W_LGB  0.2230     pooled bulk                 v15_trees_jul19.parquet
W_MLP  0.0000     residual MLP                v15_net_jul19.parquet      (driven to exactly zero)
W_PA   0.2030     per-airport bulk            v15_trees_jul19.parquet
W_D    0.0000     window transformer K = 16   d_window_rk.parquet        (zero: superseded by D2)
W_D2   0.0880     window transformer K = 32   d2_window_rk.parquet
W_K128 0.4860     window transformer K = 128  d2_window_rk_K128.parquet
                  ------
window share      0.5740
```

The docstring separately quotes `LGB 0.2016 / MLP 0.2016 / perApt 0.2016 / D 0.3952` from an
earlier OOF fit; that is a third set and also not applied. `V64_ALLOW_DRIFT=1` is required to build
with any weight overridden.

**And the window components are not the variant their filename suggests.** `d2_window_rk.parquet`
and `d2_window_rk_K128.parquet` are written by `scripts/arch/d2_rank.py` from
`scripts/arch/d2_window_wide.py`, which sets **`NF = 6`**: the six per-neighbour channels are
timing, is-arrival, arrival taxi-in, its mask, same-runway and neighbour scheduled delay.
`scripts/arch/d2_window_dep.py` sets `NF = 8` and adds a departure-side `MVT - AOBT_3` proxy and
its mask. **It was built and it does not ship.** Describing the shipped window as reading a
departure-side duration is wrong, and it would contradict the checkable claim that the two
label-carrying channels are identically zero on every departure neighbour.

## 5. Mixture: the VALIDATION instrument, and what actually ships

**Trap.** The learned five-component mixture below is what the Jan+Jul 2025 validation window uses
(`$DC26_WORK/b4ab_ctx.npz`, `b4ab_P_u.npy`, archived without labels in `artifacts/adsb_window/`), and what every offline number in our research log was
measured through. It shipped only in v62 and v70-v72, and v72 lost 2.21 on the board. **The submitted
predictions apply the same branches through a hand-built chain**: v64's cells (v29 calibration,
the v34 date-rollover rows, the leaderboard-fitted constants of section 9) and the carrier cells of
v95/v96. On the scored set the two agree within 60 s on 99.46% of rows; the chain moves 251 rows more
than 600 s off the bulk, the learned mixture 175.

```
V = [bulk, vsd, vlb, clip(86400 + bulk), bulk]        component 4 duplicates component 0
      vsd = ms clipped          (the schedule-copy hypothesis, ms = MVT - SCHED)
      vlb = MVT - LOBT clipped  (the flight-plan-copy hypothesis)
Q = P**TEMP, renormalised;  Q[:,1:][Q[:,1:] < GFLOOR] = 0;  Q[:,0] = 1 - Q[:,1:].sum(1)
TEMP = 0.90    GFLOOR = 0.10    FLOOR = 60    CAP = 200000
prediction = clip(round((Q * V).sum(1)), FLOOR, CAP)
```

Both constants were swept and are at their optimum: `TEMP` 0.80 costs 0.0761 and 1.00 costs 0.3669;
`GFLOOR` 0.05 costs 3.9179 and 0.15 costs 0.3881.

**Consequence worth stating:** `GFLOOR = 0.10` zeroes any non-bulk component weight below 0.10, so
**a signal that moves a hypothesis' probability by less than 10 points cannot reach the output at
all.**

## 6. Submission lineage

```
v64 -> {v67, v73} -> v74 -> v79 -> v84 -> v90 -> v91 -> v95 -> v96 -> v99 -> v100   (the base model)
v100 -> v102 -> v103 -> v104 -> v107 -> v108 -> v110 -> v112 -> v114                (the trajectory stage; section 11)
v114 -> v117 -> v118 -> v120 -> v121 -> v122 -> v123 -> v124 -> v125 -> v126          (the final model; section 13)
        side branches, submitted and not kept: v104 -> v105, v104 -> v106, v108 -> v109, v118 -> v119
```

A directed graph, not a chain. Each node asserts its inputs' `sha256`, so changing any upstream
artefact invalidates every assertion downstream. Every build takes `FIRE=0`, which must reproduce
its own base byte-identically; that control is the cheapest correctness check in the repository.

## 7. Verification checklist

Run these before trusting any measured delta:

1. `ranking.parquet` has 689,534 rows; feature files are the `_jul19` variants.
2. Rebuilt components reproduce `v15_trees_jul19.parquet` at `corr 1.000000`.
3. The blend-only ordinary-row RMSE on the Jan+Jul window reproduces the recorded **225.732**.
4. `FIRE=0` reproduces the base submission's `sha256`.
5. Timestamps are `datetime64[us]`: `astype('int64') // 10**9` is **1000x wrong**. Convert against
   an explicit epoch and assert `1.735e9 < median < 1.80e9`.

## 8. v100, the final submission

```
scripts/v100_build.py        v99 + the refitted bulk on v99's unfired rows      sha256 30ca298a56825a78...
public board 254.5228 (2026-09-23), predicted beforehand 254.9175 +- 0.585 by the deployed-level estimator (`scripts/p1a_instrument.py`)
```

The bulk blend is refitted as nine members with **fixed weights fitted out of fold on labelled 2025
data**, embedded in the script:

```
  L55  .0494   pooled trees (v55 recipe)          P55  .0240   per-airport trees (v55 recipe)
  K32  .0000   window transformer K = 32          K128 .2703   window transformer K = 128
  ID   .1562   identity categoricals              RQ   .1131   same-runway neighbours' naive
  CL   .2770   independently built model          L55o .0667   P55o .0431   trees on out-of-fold encodings
```

`v100 = v99` where v99 is within 60 s of `clip(MVT - SCHED)` (copy-fired rows), and
`clip(rint(v99 + M w_new - M0 w0))` elsewhere (313,332 rows).

**Trap, and it produced several wrong conclusions before it was found.** The eight `te_*` target
encodings stored in `v15_tr_jul19.parquet` were computed over all twelve months of 2025. Any validation
that trains on ten months and scores January and July must recompute them from the training months
(`scripts/p2p_te.py`); the stored ones leak the window's labels into the validation score. Production is unaffected.

Two ways to reproduce it:

```
EXACT, seconds     python scripts/v100_build.py
                   reads the archived members $DC26_WORK/p2_members_rk.parquet (copied from artifacts/) and
                   the tracked v99, and asserts the shipped sha256.       BYTE-IDENTICAL, verified 2026-09-23
FROM SCRATCH       scripts/regen_v100.sh   (DRY=1 first; 1 to 2.5 h on 18 cores)
                   rebuilds the five new members into $DC26_WORK/regen_v100 (never over the archive).
                   LightGBM times its histogram strategy at run time and CL's early stopping amplifies it
                   (section 10), so the result reproduces NUMERICALLY; the last step prints its
                   difference from the shipped file. Clean room, 2026-09-23: RMS 3.334 s with the five new
                   members re-drawn, RMS 9.795 s on 281,014 rows with all nine re-drawn.
```

The four shipped members (v55 trees, K32 and K128 windows) come from `scripts/regen2.sh` stages 7, 9 and 10, and
the chain from v79 to v99 from `scripts/regen_chain.sh`. The exact paths read archived artefacts shipped in
`artifacts/` (MANIFEST.md, section 10 below). **Two chain builds read their feature files from `/private/tmp` until 2026-09-23
(v84, v90) and could not be re-run once those files were cleared; both now read `$DC26_WORK`.**

## 9. Constants fitted to the public leaderboard

Until 2026-09-11 some scalars were obtained by submitting perturbed predictions and solving
`dSSE = 2T + Q` against the returned score. The method was retired after an organiser described
reverse-engineering the ranking as unwelcome. The values it produced remain in the chain and are listed here so a
reader can remove them. Seven scalars are defaults of `scripts/v64_build.py` (`_DEF`):

```
  constant   value       what it weights                                 checked on labelled 2025 data
  A34        0.40        six LIRF date-rollover rows, roll vs fall        no (row-specific)
  SHIFT_1   -8.778 s     January bulk offset                              reproduces: -8.629
  SHIFT_7   -3.861 s     July bulk offset                                 does NOT: -0.283
  A49        0.3376      a copy-probability band cell (53 rows)            no
  A51        0.2922      a copy cell (77 rows)                             no
  A54       -0.082271    the doubled-designator cell (33 validation rows)  validation said +0.60 and lost 3.00
  A57        0.565145    blend of v57 toward the rounded bulk             no
```

Three smaller adjustments reach v74 through `scripts/v66_build.py` and `scripts/v67_build.py`:

```
  EHAM_EPS  -0.1012      response coefficient for the EHAM disruption week, 1,634 rows    v63 probe
  LOW band   net +208    shift on 76 rows of the LOW band                                  v66 probe and inversion
  MID band   +600        shift on 20 rows of the MID band                                  v67 probe, never corrected
```

Further choices were informed by public scores without being solved from them: v29's calibration weight 0.35 was
the best of three values (0.10, 0.35, 1.00) on the board; v90's weight 0.30 was chosen against a bar derived from a
board-measured attenuation; v106 (receiver-geometry inputs) priced positive on the window and was dropped on its score
(section 12); and v119 and v126 leave the corrupted-identifier cell out because A54, the board's optimum for that cell,
had the opposite sign to its labelled rate. Every later change was kept or dropped on its score, the selection the
paper's Section 8 discusses.

**Two scores were also read as facts about the rows they moved, both after 11 September:**

```
  v96   2026-09-20   eleven carriers at LIRF, 185 rows moved to the schedule copy: the squared error fell by exactly
                     the rows' own squared shift (dSSE = -Q), so every moved row was a copy
  v120  2026-10-02   THY at LIRF in January 2026: -dSSE = 1.016 Q, every moved row a copy
```

`scripts/seam_finder.py` and `scripts/p1a_instrument.py` cite the v96 reading.

**One hand rule predates this list.** v14 (2026-09-02) moved 46 Amsterdam departures of January 2026 to the network's
off-block (`naive`) where `naive >= 4,800 s` and the prediction exceeded 6,000 s: rows of Amsterdam's January
disruption that the mixture read as extreme while the network's own clock put their taxi at 4,835 to 5,439 s. It was
written in a shell, and its recipe was recovered afterwards by diffing the shipped files (`scripts/v14_combine.py`
reproduces v14 byte for byte). The two thresholds are the predicate that reproduces that 46-row edit exactly, with a
clean gap to the next rows (`naive` 4,770 against 4,835 s), not values fitted to a score; v14 was submitted once and
kept on its score. It reaches the final model through v23 and v29 (`scripts/v15_combine.py`, `scripts/v25_build.py`).

**What they are worth.** v76 removes SHIFT_7, A49, A51, A54, A57 and the LOW and MID shifts from v74, and scored
**2.7501 s worse on the public standing, measured** (an earlier estimate of 2.53 was low). A34, SHIFT_1 and EHAM_EPS
stay in v76 (`scripts/v76_build.py` says why), so they sit in both arms of that comparison and **2.7501 s understates
the total**. A34 = 0.40 is itself a board inversion; its cell's labelled prior is 0.55. SHIFT_1 reproduces on
labelled data (-8.629). All of these are specific to this scored set: on a different draw expect them to add nothing
or to misfire, so a reproduction on new data should set them to their neutral values (0, or the labelled-data values:
-8.629 for SHIFT_1, 0.55 for A34).

## 10. Clean-room verification, and why `artifacts/` holds neural predictions

On 2026-09-23 every production stage was re-run from the released code into an empty work directory, with no
author paths, one stage at a time, and each output was compared with the original.

```
  IDENTICAL, every row and column
    v8s / v15 feature tables (_jul19), v15_trees (pooled, per-airport, classifier), v55_trees, v99 bulk components
    dc26_derived, night_{tr,va}_all, gate_stage_a2.npz          the validation-window instrument
    opdi_airframe, from a FRESH OPDI download (all 15 files byte-identical to the originals)
    te.npz, p1c_raw, the CL feature frames, OOF v55 trees seed 0
  NUMERIC DRIFT
    v15_net            8 MPS nets                 RMS  6.09 s
    gate_mlp           8 MPS nets (window side)   RMS  3.82 s
    window, rank       K16 / K32 / K128           RMS 12.13 / 18.61 / 41.23 s
    window, val side   K32 / K128                 RMS 19.49 / 36.34 s
    ID, RQ arms        LightGBM                   RMS  0.10 / 0.20 s (2-seed mean)
    CL                 LightGBM, early stopping   RMS 12.26 s (seed 1 stopped at 3,631 against 3,716)
    OOF v55 seed 1     LightGBM                   RMS 0.042 s pooled, 1e-14 per-airport
```

LightGBM here is deterministic except for its runtime-timed histogram choice (colsample does NOT cause drift: the
v99 components use it and rebuild identically). The neural components re-draw on every run. A re-drawn K=128 sits
71% of the way to a different architecture.

**The chain amplifies the MLP.** `v64_build.py` reconstructs each row's mixture weight from the archived v29 via
`P = (p29 - bo_)/(copy - bo_)` with `bo_ = 0.2 lgb + 0.4 mlp + 0.4 perapt`. A re-drawn MLP (RMS 6.1 s) makes v64's
v57-cell check fail (85 rows against 84). With the checks relaxed, it moves v74 by RMS 30.3 s. So the chain from v64
can only be reproduced with the archived MLP predictions.

**What ships in `artifacts/`, all our own predictions or fitted objects:**

```
  v15_net_jul19.parquet                    the MLP on the scored rows (read by v64 through v90)
  d_window_rk, d2_window_rk, d2_window_rk_K128.parquet    the three rank-side window transformers
  arch_d2_wide_va_K32.npy, arch_d2_wide_va_K128.npy        the window-side transformers v84 reads
  p2_members_rk.parquet, full_package.npz  v100's nine members and weights
  v99_bulk_components.parquet              v99's components (reproducible; kept for speed)
```

`scripts/regen2.sh` and `scripts/regen_chain.sh` copy the archived neural predictions into place after each
re-draw and keep the new ones as `*_redrawn`; `USE_REDRAWN=1` keeps the re-drawn ones instead.

**Bugs the clean room found and fixed, outputs unchanged.**

- Three producers of the validation-window data were missing from the release and the driver: `newcols.py`,
  `derived_ops.py` and `gate_mlp.py`.
- `regen2.sh` did not set `DC26_RECIPE`/`DC26_POOLED_CAP`.
- `pull_pru.py` never wrote the parquets the features read.
- `p1c_load.py` did not create its output folder.
- The window scripts read another stage's file for a diagnostic before writing their own output, which lost two
  GPU runs.
- `/tmp` was hardcoded (now `DC26_TMP`).

## 11. The trajectory stage: v102 to v114

```
scripts/regen_adsb.sh    puts the archived inputs in place, optionally pulls the days and rebuilds the feature
                         tables, then prices and builds v102 -> v114, asserting every sha256. Writes only to a
                         temporary directory; submissions/ is never touched.
  PULL=1 FROM_DAYS=1     first run: stream the 427 days, rebuild and hash-check the five feature tables
  (no variables)         later runs: from the tables already in artifacts/
  FROM=110               only v110, v112 and v114 (needs v108's candidate from an earlier run)
```

**Data.** adsb.lol publishes each day's traces as GitHub releases (`adsblol/globe_history_2025`, `_2026`), under
ODbL 1.0 (`EXTERNAL-DATA.md` section 3). `scripts/adsb_stream_day.py` streams one day's archive without writing it
to disk and keeps the ground samples, take-offs and landings within 8 km of the ten airports, in
`$DC26_WORK/adsblol/days/`. `adsb_pull_all.sh` pulls the 124 January and July days of 2025 and 2026, and
`adsb_pull_year.sh` the ten other months of 2025, taking each day's replica from the archived
`artifacts/adsb_pull/PREFERRED_RELEASES_2025.txt`. `artifacts/adsb_pull/day_manifest.tsv` records, for every day,
the replica and the counts of traces, ground points, take-offs and landings, so a re-pull can be checked day by
day; `scripts/adsb_pull_check.py --final` lists missing days. The pull needs `curl` and, for the January and
July days, the GitHub CLI (`gh`, signed in) to look up each day's release assets.

**Feature tables.** `scripts/adsb_push_features.py` matches departures to traces and measures the pushback. It
runs five times, one table per stage generation:

```
  key    settings                                           read by
  v102   ADSB_TIME_MATCH=0 ADSB_PARK=0                      v102
  v103   ADSB_TIME_MATCH=1 ADSB_PARK=0                      v103
  v104   ADSB_TIME_MATCH=1 ADSB_PARK=1                      v104, v107, v108, v109; v110 for coverage
  v105   ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_XY=1            v105, v106
  full   ADSB_TIME_MATCH=1 ADSB_PARK=1 ADSB_ALL_MONTHS=1    the full-year members and stage, v110 to v114
```

The tables are derivative databases of adsb.lol and are not in this repository. With `FROM_DAYS=1` the driver
rebuilds each one, compares it with `artifacts/adsb_push_feat.content.sha256` (a hash of the values, rows sorted
by movement id, independent of the parquet encoding; `scripts/adsb_feat_hash.py`), and places it in `artifacts/`,
where the pricers read it, with exactly the archived columns in their order: the stage harness uses whatever
detection columns a table carries, and today's builder adds four to the v102 table. A table that differs stops
the run.

**Window inputs.** The stages are trained on the labelled validation window (January and July 2025) through
`scripts/p1a_instrument.py`. `artifacts/adsb_window/` archives everything they read besides the challenge data:
v100's member predictions on the window, the base model's window context and mixture probabilities, and the
full-year trajectory members' predictions (window side, scored side, and month-fold out of fold). The window
context ships **without labels**; `p1a_instrument.window()` reads them from `$DC26_DATA` and still asserts the
recorded window RMSE of 313.8931. The driver copies every archived input into `$DC26_WORK` or `$DC26_SESSION` and
refuses one that differs from `SHA256SUMS` (v100's members and weights are taken from `artifacts/`, not
duplicated). It also needs `$DC26_WORK/v15_rk_jul19.parquet` from `regen2.sh`: `ONLY=1`, then `ONLY=2`, which writes it to `$DC26_TMP`; copy it into `$DC26_WORK`. It needs it
for its movement ids.

**Stages and builds.** Each pricer trains its stage, prints its pre-registered forecast and writes
`$DC26_WORK/adsblol/cand/vNNN_adsb.npz`; each `scripts/vNNN_build.py` writes the parquet, asserts its sha256, and
with `FIRE=0` reproduces its base.

```
  v102  adsb_price.py                   layer 1 on callsign-matched rows, LightGBM
  v103  adsb_price_v103.py              + the time-only match and detection features (harness adsb_stage_exp.py)
  v104  adsb_price_v104.py              + the inbound parking position
  v105  adsb_price_v105.py              side branch: weather from artifacts/metar_iem_asos.parquet (pull_metar.py)
  v106  adsb_price_v106.py              side branch: receiver-position features
  v107  adsb_price_v107.py              + XGBoost; settings frozen in artifacts/v107_xgb_params.json (v107_flaml_search.py)
  v108  adsb_price_v108.py              + CatBoost and the MLP; MLP predictions archived in artifacts/v108_stage_mlp.npz
                                        (stage_mlp_export.py + stage_mlp.py on the CPU, deterministic; FROM_DAYS=1
                                        retrains it and requires equality)
  v109  adsb_price_v109.py              side branch: v108 without the MLP
  v110  adsb_price_v110.py, then        layer 2: the full-year members replace v100's bulk on covered rows,
        adsb_v110_diag.py SAVE=Cfam     blended with v108 at a cross-fitted weight
  v112  adsb_stage_full.py SAVE=1       layer 3: the full-year stage, blended with v108 at 0.662
  v114  adsb_price_v114.py SAVE=1       LIRF kept at v108
```

**The full-year members** are archived, so the driver does not retrain them. To rebuild them from scratch, after
the pull:

```
  FROM=train scripts/overnight_v110.sh   pooled and per-airport members with the trajectory features, 3 seeds:
                                         window side on the ten non-window months, scored side on all twelve
  scripts/oof_folds_v112.sh              their month-fold out-of-fold predictions, seed 0: five two-month folds
                                         inside the ten non-window months (window), six over twelve (scored side)
```

Both overwrite `$DC26_SESSION/p2P/trees_adsb`. LightGBM is multithreaded here, so a from-scratch run can move the
predictions slightly; `V110_ALLOW_DRIFT=1` and `V112_ALLOW_DRIFT=1` let v110 and v112 finish and report their
difference from the shipped files.

**Verified.** In the author's tree, each of v102 to v114 rebuilt byte-identically from the archived tables and
members when it was added to the driver (2026-09-24 to 09-28). In a clean room on 2026-09-30, from an empty work
directory: the driver restored the 72 archived window inputs and stopped, as designed, first on the missing stage-1
file and then on the feature tables; the label-free window context reproduced the recorded window RMSE; and the
five tables rebuilt from the parsed days, placed as above, equal the archived ones in values, columns, order and
dtypes. The full clean-room run needs the 427-day pull and has not been done.

## 12. The clean rebuild: the same architecture without the constants fitted to the leaderboard

`scripts/regen_clean.sh` builds it into `$DC26_WORK/clean/` and never touches `submissions/`. It was scored once to
measure what the model is without anything section 9 lists. It differs from the submitted chain in the base; the
trajectory stage is rebuilt on it by the same rules:

```
  base on the 2026 rows   the learned five-class gate (artifacts/b4_gate_rk.parquet, restored by regen_clean.sh;
                          scripts/arch/b4_rank.py re-draws it) through pi = P^0.90 and floor 0.10,
                          the construction the window has always used, in place of the hand-built chain of
                          sections 4 to 8 (scripts/clean_base_build.py)
  month offsets           the nine-member bulk's mean residual on ordinary window rows, January -10.638 and
                          July -0.548 (clean/offsets.json), in place of SHIFT_1 -8.778 and SHIFT_7 -3.861
  section 9               absent entirely: no A34, A49, A51, A54, A57, EHAM_EPS, LOW or MID band, v29 or v90 weight
  bulk weights            WF=refit (the default): the nine-member weights refitted with a free intercept per month;
                          WF=archived: full_package.npz's weights, as in v115/v116 (see "One trace" below)
```

The trajectory stage (section 11) is then re-priced and rebuilt on that base by the same scripts with `CLEAN=1`,
`DC26_SUBDIR`, `DC26_CAND`, `DC26_WF` and the two offsets exported. With `CLEAN=1` the scripts refuse to run unless
all five are set and point away from the submitted chain's files, so a hand-run CLEAN step cannot fall back to the
leaderboard offsets or the shipped files. Every selection rule picks the variant it picked for the
submitted chain (v110 "C alpha", v112 "C2 alpha" at 0.658 against 0.662, v114 "no LIRF"). The side branches v105,
v106 and v109 are not built; v106 priced positive on the window and was dropped on its leaderboard score, so the
chain's composition is itself a selection made with the leaderboard, as the paper's Section 8 says. In CLEAN mode the
builders skip the sha256 checks against the shipped files (the FIRE=0 control still runs, against the clean base), and
the pricers' "->" projections, anchored on the submitted scores, are meaningless and are not used.

```
  window RMSE                 all rows    ordinary rows
  clean base                  309.592       237.356
  clean final                 296.619       219.889
  submitted chain's vectors   309.603 / 296.671   237.367 / 219.963   (the leaderboard offsets are the only difference)
```

On the 2026 rows the clean base agrees with the submitted v100 within 60 s on 98.95% of rows; the 225 rows that differ
by more than 600 s, 174 of them at LIRF, carry 97.7% of the squared difference (`scripts/paper_clean_vs_submitted.py`).
The paper's window numbers for this model come from `paper_baselines.py`, `paper_window_ladder.py`,
`paper_pushback_split.py` and `adsb_gate_oracle.py`, each run with the same environment as the driver.

**Verified and scored.** Two runs of `regen_clean.sh` into separate directories (2026-09-30) give byte-identical
files at every step, v100 through v114, and after the review fixes below `WF=archived` still rebuilds them byte for
byte. The clean base and final were submitted once each as v115 and v116, which `scripts/v115_v116_build.sh` rebuilds
and checks against their sha256 before copying them into `submissions/`:

```
  2026 set     base        final       stage gain   squared error the stage removed
  clean        292.0485    268.1373    23.9112      4.6190e9      v115 00bf453e..., v116 a460c540...
  submitted    254.5228    226.8129    27.7099      4.5994e9      v100, v114
```

The stage removes almost exactly the same squared error from either base. The bases' predictions differ mainly on the
225 rows where they are more than 600 s apart, 174 at LIRF, which carry 97.7% of the squared difference between the two
vectors (6.3009e9); the 37.53 s score gap also holds a cross term (+7.7196e8) that cannot be placed without labels.
Within the 225, 13 rows where the learned mixture weights the date branch carry 59.9% of the squared difference, 7 of
them failing the date-crossing condition the submitted chain enforces (v91); 123 copy-only rows, 119 at LIRF, carry
32.3%; 89 others 5.6% (`scripts/paper_clean_vs_submitted.py`). The submitted
chain remains the competition entry; its final model is v126 (section 13), which the paper reports.

**One trace, found by a code review.** v115 and v116 carry two small traces of the leaderboard month offsets:

```
  bulk weights      full_package.npz's nine-member weights were fitted (scripts/p2p_checks.py) against y minus
                    SHIFT_1/SHIFT_7. WF=refit removes this: weights move by at most 0.0026, offsets become
                    -10.6564 / -0.5278, window base 309.592 -> 309.594, window final 296.619 -> 296.596.
                    On the 2026 rows the base's predictions move 0.45 s RMS, which bounds its score change;
                    the final model's move 3.86 s RMS as the stage retrains. Not resubmitted.
  XGBoost settings  artifacts/v107_xgb_params.json was searched by FLAML on labelled Jan+Jul 2025 cross-validation
                    (scripts/v107_flaml_search.py), not on the leaderboard, but against the residual of a base that
                    carried the leaderboard offsets. Second order; not re-searched.
```

The same review measured the one suspected label leak and cleared it: the window gate reads twelve-month target
encodings, and retrained on leak-free ten-month encodings it gives the clean base 309.593 on the window against 309.592.

**The clean final, v127 (2026-10-05).** `scripts/v127_build.sh` copies the v116 run of `regen_clean.sh` into
`$DC26_WORK/clean127` and re-runs the rule scripts of v117 and v120 to v126 on it in CLEAN mode (`CLEAN=1`, the base and
output passed as `V1xx_BASE` / `V1xx_OUT`, `V121_REF` for the two overlap checks; v120 and v121 build their bulk from
`DC26_MEMBERS_RK` = `artifacts/p2_members_rk.parquet`, `DC26_WF` and the clean offsets in `SHIFT_1` / `SHIFT_7`). Without
`CLEAN=1` every one of those scripts behaves exactly as before and still reproduces its recorded sha256. v117's rule did
not pass on the clean chain (G/SE +0.97), so v127 = v116 + v120..v126, sha256 74fcf296..., scored 247.7412 against a
pre-registered 246.41 (kill line 257.89).

## 13. The final model: v117 to v126

v126 is the trajectory stage of section 11 with nine later changes. Each was pre-registered with a predicted score and
a kill line, and each build asserts the sha256 of what it writes; the single-file builds also assert their base's,
and with `FIRE=0` reproduce it byte for byte.

```
  v117  226.5352  adsb_price_v117.py SAVE=1 -> cand/v117_adsb.npz, then v117_build.py
                  the weight that moves covered rows from layer 1 toward layers 2 and 3, refitted on the window's
                  covered rows outside LIRF: 0.837 (January fit 0.834, July 0.840) in place of 0.662, which LIRF's
                  July-only coverage had pulled down; LIRF stays at layer 1
  v118  226.3833  v118_build.sh           more seeds on the scored side: K128 8 -> 18, CL 2 -> 10, L55A 3 -> 10.
                                          Nothing is refitted: regen_seeds.sh averages the seeds (seeds_members.py)
                                          and reruns the chain's scored side, v100 -> v117, with the shipped
                                          weights, alphas and selection rules (SKIP_REGEN=1 reuses its output)
  v119  226.4823  v119_build.py           NOT KEPT: LIRF flight numbers that never join a flight plan, delayed an
                                          hour or more, moved toward the copy at their 2025 rate (83 of 87)
  v120  225.9972  v120_build.py           THY at LIRF: in a 2026 month whose THY early arrivals are >= 90% copied
                                          (on >= 10), its departures inside the Nov-Dec 2025 range of ms (62 to
                                          5,695 s) -> 0.992 * clip(ms) + 0.008 * bulk; the gate opens on January only
  v121  225.9576  v121_build.py           outage spans (>= 5 consecutive copied early arrivals of one carrier):
                                          a 2026 LIRF departure inside a 2026 span of its own carrier moves to
                                          w * clip(ms) + (1 - w) * bulk where v120's implied copy weight is below w;
                                          w is the carrier's own 2025 inside-span copy rate if it followed in 2025,
                                          0.731 (the class rate) if it had no 2025 span; non-followers untouched
  v122  224.7674  v122_build.py           the flight-plan join: a plan attaches only when |BLOCK - LOBT| <= 3606 s
                                          (asserted on all 2,062,577 join-present 2025 departures). Predictions
                                          with a plan clipped onto MVT - LOBT +- 3606 (29 rows, seven airports);
                                          LIRF departures without one, plan recovered from their arrival at the
                                          destination, taking off > 3606 s after it: MVT - AOBT_3 (22 rows; 2025
                                          76 rows, 0 copies, asserted)
  v123  223.5285  v123_build.py           LIRF no plan, 3600 <= ms < 14400, plan not recoverable: 0.5 * (taxi_hat
                                          + P (ms - taxi_hat)) + 0.5 * v122, P from scripts/fable_lirf/cf_clf.py
                                          (271 rows); input l5_2026.parquet, sha-asserted
  v124  222.9558  v124_build.py           LIRF no plan, recovered plan, take-off within 3606 s of it: raised to the
                                          floor MVT - LOBT + 3606 (6 rows; 0 of 30 2025 rows below it, asserted),
                                          then toward clip(ms) at 21/23 where ms exceeds the floor (12 rows)
  v125  222.9161  v125_build.py           LIRF no plan, ms < 1800, not recovered: the class mixture p_real * 1000 +
                                          p_tz * (1000 + UTC offset) + p_copy * clip(ms), rates per carrier group
                                          counted on 2025 (27 rows); input l3_2026.parquet, sha-asserted
  v126  222.8507  v126_build.py           LIRF no plan, 30,000 <= ms < 86,000, identifier not corrupted: clip(ms)
                                          below 46,800 s, max(prediction, clip(ms)) above (6 rows; 2025 support
                                          asserted: 45 rows, 0 below ms, no date error below 46,800)

  sha256  v117 47c5f0309c6a415617c615125c8160435cb38aa3ba918e6c4fba51b13406ee11
          v118 2f2079f70ba9526ac7ee93f131711e64ece209266f4d6cf6378b364e365428de
          v119 20aa2fd24dc82009518460b572d4f6f69aae8a29f08a59598b4c3196dc014f4a
          v120 16e294f1b6a0f52c72e53db9e2115eabb040b6cac5394c5ddf13d25b57d6d507
          v121 7acc7069a8c3974cec476cb68e97c278d7fd2e473a12fef53bdc46cc3fbd8fa7
          v122 29cdf104e680abbeda21fc225df6e9cf4269e0c29061e2113fe5a5d7e41bf28e
          v123 9f2f9f4bf8e6948d241ab53218da55b839f409179b39630c9ba51d252d0ece21
          v124 a99fef2878dab2d2c457bf96f1a597cc8f6705c4bde35ddeae7ddb45f1ad70fd
          v125 64eb3556d3fa7c93c8b7f8b891b6f71bc800021e681fc01901cd841cd8ab9612
          v126 d66768fe3504ddcc6ed0bbf9a0e97bfff5d9e353e73bc4e4a4b170a079b6a951
```

**Order.** After section 11's inputs are in place (the feature tables, `artifacts/adsb_window`, stage 1's
`v15_rk_jul19.parquet`): `CONTROL=1 ./scripts/regen_seeds.sh` rebuilds v100 through v117 with the shipped seeds and
must reproduce v117 byte for byte (into `$DC26_WORK/seeds/chain_control`); `./scripts/v118_build.sh` builds v118
from every archived seed and copies it into `submissions/` only if its sha256 matches; then `v120_build.py` and
`v121_build.py`, which read `$DC26_WORK/seeds/chain/p2_members_rk.parquet` from that run. Then `v122_build.py` to
`v126_build.py` from the repository root; v123 and v125 read the classifier's and the class mixture's 2026 outputs from
`artifacts/fable/` (sha-asserted). `FABLE_DIR=$DC26_WORK/fable_lirf scripts/fable_lirf/run_all.sh` regenerates them
from the challenge files, v121's window vector (`$DC26_WORK/adsblol/cand/v121_window.npz`, written by
`scripts/paper_final_window.py` once the v117 chain has run) and v121's submission; run from the repository into an empty directory
on 2026-10-03 it reproduced both files byte for byte. Its LightGBM is multithreaded and could move the classifier's
probabilities on other hardware; `FABLE_DIR=$DC26_WORK/fable_lirf` on the builds then uses the fresh copies. That pipeline was written by a separate AI agent session given only the challenge's files
and our predictions; every rule it found was checked here with independent code before a build used it.

**The extra seeds** are archived in `artifacts/seeds/` with a `SHA256SUMS` (32 files: K128 seeds 8 to 17 on the
scored rows and their row order, CL 0 to 9, ID and RQ 0 and 1, L55A 3 to 9; L55A 0 to 2 and P55A are in
`artifacts/adsb_window`). `regen_seeds.sh` restores any that are missing from `$DC26_WORK` and `$DC26_SESSION` and
refuses one that is present and differs (`RESTORE_ONLY=1` stops there). `seeds_members.py` averages every seed file it
finds, so a work directory holding further seeds builds a different model. They were trained by
`scripts/seeds_train.sh` (CL, L55A and the CPU arms) and `scripts/arch/d2_rank.py` with `SEED0=8 SEED_DIR=...` (K128,
on Metal); the K128 net re-draws and the trees can drift slightly, so retraining reproduces v118 only to within seed
noise, which is why the predictions are archived.

**What the two rules read.** v120 and v121 read the 2025 training files and the arrivals in `ranking.parquet`, which
keep their labels in the scored months; no label of a scored departure exists in the extract. Every rate, threshold
and range they use is computed in the script from labelled 2025 data (v120 asserts its 241 of 243, v119 its 83 of
87). Their `bulk` is the base model's nine-member blend on the scored rows plus the base model's month offsets,
`SHIFT_1` and `SHIFT_7` of section 9, so those two leaderboard-fitted constants reach the rows the rules move. The
implied copy weight is `(v - bulk) / (clip(ms) - bulk)`, clipped to [0, 1], for the base submission's prediction `v`.

**On the window.** `scripts/paper_final_window.py` writes the final model's window vector for the paper: v117's
cross-fitted window vector (v118 has no window counterpart; its seeds were trained for the scored side), with v120's
gate checked on the window months (it opens on neither) and v121's rule replayed with its classes, rates and ms range
learned on the ten other 2025 months. The replay leaves out the eleven carriers v96 commits to the copy, because the
window's validation base does not carry that cell and the 2026 side already sits at the copy there. It moves 97 window
rows, 90 of them copies, and costs 1.3212e7 of squared error, a 95% day-clustered interval from -8.822e7 to +4.295e7:
the 90 copies gain 2.9875e7 and four KMM non-copies cost 4.250e7. It fires there only on carriers with their own 2025
rate (KMM, MEA, GFA), so the class-rate arm, 73 of the 81 rows v121 moved in 2026, has no window test; with the eleven
carriers in, the replay gains 1.1515e8 (`V96_OUT=0`). v121 paid +0.0396 on the leaderboard, 7% of its expected gain.
v120 paid -dSSE = 1.016 Q, every moved row a copy. `paper_baselines.py`, `paper_window_ladder.py` and `paper_pushback_split.py`
read that vector.

**Verified** in the author's tree on 2026-10-02: v117 and v119 to v121 rebuild to their recorded sha256 and every
`FIRE=0` control reproduces its base; `v118_build.sh` with `SKIP_REGEN=1` matches v118's sha256; `CONTROL=1
regen_seeds.sh` rebuilt v100 through v117 byte for byte on 2026-10-01; the seed restore, run into empty directories,
places all 32 files and refuses a corrupted one. A from-scratch run of `v118_build.sh` from an empty work directory
has not been done.

## 14. The final phase: the final model on four months

On 2026-10-08 the organisers added February and June 2026 and asked for one file covering January, February, June and
July, ranked without feedback. **The entry is the final model**, `submissions/quick-boat_final.parquet`: the clean
design of section 12, plus the chain's cells whose evidence is labelled, with no value fitted to a leaderboard score.
It is retrained once on the twelve 2025 files and predicts the four months together. `FINAL.md` describes both
delivered files and lists the source of every value on the final model's path.

```
  submissions/quick-boat_final.parquet            the final model, 670,790 rows        97f3819fcc04c1f4...a9cab
  submissions/quick-boat_final_with_v126.parquet  Jan/Jul from v126, Feb/Jun as above  16cfd173334db2ae...ca2f9
```

**The build.** `scripts/final/run_final.sh` runs twelve steps from the raw extract into empty work folders, reading
`final_ranking.parquet` as `ranking.parquet` (a folder of symlinks):
1. `regen2.sh` stages 1 to 7;
2. the window networks K128 and K32;
3. the chain's copy model, not used by the entry (read 2 below);
4. the nine bulk members;
5. the learned gate;
6. the 2025 window inputs restored from `artifacts/adsb_window` (never the scored-side files);
7. the four ADS-B feature tables;
8. the full-year ADS-B members;
9. the clean base, the seam rules v95 and v96 (`scripts/final/seam_rules.py`), and the stage v102 to v114 with every
   variant pinned to the clean chain's (`PIN_PICKS=1`);
10. v117, Fable on four months, then v120 to v126 in CLEAN mode;
11. the LIRF day-scale bands (`scripts/final/lirf_dayscale.py`), which write the entry;
12. the splice (`scripts/final/splice_final.py`), which writes the comparison file.

The 2026-10-08 build ran from 13:03Z to 21:12Z on one laptop. K128 on the GPU took 5 h 34 min and the v55 trees 3 h 1 min.

**What had to change, and the control that it changed nothing else.** The patches are switches whose defaults keep
every earlier build byte-identical:
- per-month offsets: `SHIFT_2` +3.800 and `SHIFT_6` +2.727 from `scripts/xfit12.py`; January and July unchanged;
- the ADS-B day set (`ADSB_SCORED_MONTHS`, `ADSB_MONTHS`) and the table paths (`ADSB_FEAT_*`, `ADSB_RK_IDS`, `ADSB_RANK_DIR`);
- `PIN_PICKS`, because four pricers compare variants on the scored rows and would compare different rows on four months;
- the Fable pipeline reading this run's v121 (`FABLE_V121`);
- row-count asserts that now take the template's count.

On the 2026-09-04 extract, `scripts/verify.sh` passes all 22 checks, and the pinned clean chain rebuilds v116
(`a460c540...`) and v127 (`74fcf296...`) byte for byte.

**Controls on the four-month run.**

| Control | Result |
|---|---|
| Jan/Jul feature rows against the archive | differ only in the columns the re-jittered take-off and block times feed |
| 2025 rows of the four ADS-B tables | identical to the archived tables in every column |
| Nine members on Jan/Jul, RMS from the archived members | L55 19.3, P55 20.1, K32 19.4, K128 40.1, ID 15.7, RQ 16.5, CL 30.7, L55o 20.9, P55o 20.5 s |
| End to end against v127, Jan/Jul | RMS 68.54 s |
| Feb/Jun median prediction by airport | tracks the 2025 label median of the same month |

The neural members' differences match their recorded re-draw spread. The tree members follow the take-off shift with
slope 0.89 (L55) and 0.71 (P55). End to end, 72 rows move by more than 600 s. 46 of them sit in the cells the port adds and carry 97.6% of the squared
change: 17 in the day-scale bands (91.2%), 24 of v96's carriers (3.0%) and 5 of v95's stands (3.4%). The other 26
carry 2.3%.

**Three reads, each pre-registered** (`analysis/final/PREREGISTRATION.md`, committed before upload):
```
  v128  v127 unchanged                         expected 247.7412           scored 247.7412   the labels are unchanged
  v129  the chain's copy classifier, alpha 1   expected 247.41 [244.34, 250.45], adopt below 246.18
                                                                            scored 248.2763   not adopted
  v130  the final model's Jan/Jul rows         expected 239.25 [232.60, 245.73]
                                                                            scored 233.7464
```
`scripts/final/v128_v129_build.sh` and `scripts/final/read3_build.py` rebuild the three files.

**Traces of the fitted values that remain:**
- the stage's XGBoost settings, searched against the residual of a base that carried the leaderboard month offsets (section 12);
- the Fable classifier, which trains on the competition chain's window vector (`FABLE_WINDOW`).
