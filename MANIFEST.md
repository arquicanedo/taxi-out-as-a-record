# What is in this repository

201 code files in three groups: the production chain that builds the final model (four months, `scripts/final/`), the competition model (v126) and its base (v100), the
retired leaderboard-probing code published for disclosure, and the instruments that decided what shipped. They are
listed in the order they run.

## 1. The production chain: raw extract to the base model (v100, 254.5228), the competition model (v126, 222.8507) and the final model

Drivers, in order: `scripts/regen2.sh` (stages 1-13, raw data to v76), `scripts/regen_chain.sh` (the
validation-window artefacts v84 reads, then v79 to v99) and `scripts/regen_v100.sh` (v100's new members and
the final build). `DRY=1` prints each plan. Paths come from `DC26_DATA` and `DC26_WORK` (required), `DC26_TMP` (default
`/tmp`) and `DC26_SESSION` (default `$DC26_WORK/session`).

```
features         scripts/features.py  v8s_build.py  v15_augment.py  opdi_features.py
bulk members     scripts/v15_trees.py  v15_net.py  v55_trees.py
window instrument scripts/newcols.py  derived_ops.py  night_prep.py  gate_a2.py  gate_mlp.py  arch/d2_window_wide.py       (read by v84)
                 scripts/arch/harness.py  arch/d_rank.py  arch/d2_rank.py          (window transformers, K=32 / 128)
the chain        scripts/v29_build.py -> v64_build.py -> v73_build.py -> v74_build.py -> v79_build.py
                 -> v84_build.py -> v90_build.py -> v91_build.py -> v95_build.py -> v96_build.py
                 -> v99_components.py + v99_build.py
v100 members     scripts/p2p_te.py  p2p_v55.py  p2p_arm.py (+ p1e_arms.py)  p1c_load.py
                 p2p_clean_feat1.py  p2p_clean_feat2.py  p2p_clean_feat3.py  p2p_clean.py  p2p_save_members.py
final            scripts/v100_build.py        asserts sha256 30ca298a56825a78..., FIRE=0 reproduces v99
```

Every build asserts the sha256 of its inputs, and `FIRE=0` reproduces its base byte-identically.
`REPRODUCE.md` documents what the scripts do not say, and section 9 lists the constants fitted to the
public leaderboard.

**Archived artefacts** (`artifacts/`, 66 MB, all our own predictions or fitted objects). The exact paths read
`p2_members_rk.parquet` (v100's nine members on the scored rows), `full_package.npz` (v100's weights and priced
vector) and `v99_bulk_components.parquet` (v99's colsample components). The chain from v64 on also needs the neural
predictions, because they re-draw on every run and v64 amplifies the MLP's re-draw about five-fold
(`REPRODUCE.md` section 10). Those are `v15_net_jul19.parquet`, the three rank-side windows (`d_window_rk`, `d2_window_rk`,
`d2_window_rk_K128`) and the two window-side `arch_d2_wide_va_K{32,128}.npy`. The drivers pin those after each
re-draw (`USE_REDRAWN=1` opts out). For the exact builds, copy the parquets into `$DC26_WORK` and the npz into
`$DC26_SESSION/p2P`. Checked on 2026-09-23 in a clean room, from the archived bases:
- v91, v95, v96, v99 and v100 rebuild byte-identically;
- v73, v79 and v84 rebuild with identical values (only parquet metadata differs);
- v90 moves 162 rows through its five-seed classifier, RMS 0.45 s.

Every stage from raw data was also compared with its original; see `REPRODUCE.md` section 10.

### The trajectory stage: v100 to v114

Driver: `scripts/regen_adsb.sh` (`REPRODUCE.md` section 11). Paths as above.

```
adsb.lol days    scripts/adsb_stream_day.py  adsb_pull_all.sh  adsb_pull_year.sh  adsb_pull_check.py  adsb_day_manifest.py
feature tables   scripts/adsb_push_features.py  adsb_feat_hash.py
window context   scripts/p1a_instrument.py                                     (labels read from $DC26_DATA)
full-year trees  scripts/p2p_v55_adsb.py  overnight_v110.sh  oof_folds_v112.sh
stages           scripts/adsb_price.py -> adsb_stage_exp.py (harness) -> adsb_price_v103.py -> adsb_price_v104.py
                 -> adsb_price_v107.py (+ v107_flaml_search.py) -> adsb_price_v108.py (+ stage_mlp_export.py, stage_mlp.py)
                 -> adsb_price_v110.py + adsb_v110_diag.py -> adsb_stage_full.py -> adsb_price_v114.py
hedge oracle     scripts/adsb_gate_oracle.py                                   (reads adsb_v110_diag.py; the paper's hedge split)
side branches    scripts/adsb_price_v105.py (+ pull_metar.py)  adsb_price_v106.py  adsb_price_v109.py
builds           scripts/v102_build.py ... v114_build.py        each asserts its base and its own sha256
```

**Archived artefacts for the trajectory stage** (`artifacts/`, all our own predictions, fitted settings or pull
records): `adsb_window/` (the window inputs the pricers read and the full-year members' predictions, 72 entries in
its `SHA256SUMS`, labels removed; two of those entries, `p2_members_rk.parquet` and
`session-20260922/p2P/full_package.npz`, ship at `artifacts/` itself and `regen_adsb.sh` finds them there by base name), `v107_xgb_params.json`, `v108_stage_mlp.npz`, `adsb_pull/` (the replica list and
the per-day manifest), `adsb_push_feat.content.sha256` and `metar_iem_asos.parquet` (v105's weather, public domain).
The adsb.lol-derived feature tables themselves are not included (`EXTERNAL-DATA.md` section 3).

### The final model: v114 to v126

`REPRODUCE.md` section 13. Paths as above; it needs the trajectory stage's inputs in place.

```
v117             scripts/adsb_price_v117.py -> v117_build.py            the stage's weight refitted outside LIRF
v118             scripts/v118_build.sh -> regen_seeds.sh -> seeds_members.py, then the chain v100 -> v117 on the
                 scored side with the extra seeds (seeds_train.sh and arch/d2_rank.py trained them)
v119             scripts/v119_build.py                                    side branch, not kept
v120, v121       scripts/v120_build.py  v121_build.py                    the two rules read from labelled arrivals
v122 to v126     v122_build.py ... v126_build.py   the flight-plan join rule, the coin-flip classifier, the
                 local-time class and the day-scale floor; inputs archived in artifacts/fable/, regenerated from the
                 challenge files only by scripts/fable_lirf/run_all.sh (FABLE_DIR = its working directory)
one command      scripts/reproduce_final.sh                               v118 from the seeds, then v120 to v126 (TAIL=1: seconds)
verify           scripts/verify.sh                                        the two-minute check: hashes, v100, v120 to v126
clean final      scripts/v127_build.sh                                    v116 + the rules of v120 to v126 (CLEAN mode)
window           scripts/paper_final_window_v126.py                       v122 to v126 replayed on the labelled window
window           scripts/paper_final_window.py                            the competition model on the labelled window
```

**Archived artefacts for the competition model**: `artifacts/seeds/` (32 per-seed predictions on the scored rows and their
`SHA256SUMS`; `regen_seeds.sh` restores them into the work directories), and the submissions v117 to v126; `artifacts/fable/` (the classifier's and the class mixture's
2026 outputs, with `SHA256SUMS`).

### The final model: four months (`REPRODUCE.md` section 14, `FINAL.md`)

```
driver           scripts/final/run_final.sh                               twelve steps, raw extract to quick-boat_final.parquet
rules            scripts/final/seam_rules.py                              v95 and v96 at their 2025 rates, any scored months
rules            scripts/final/lirf_dayscale.py                           LIRF day-scale bands from the 2025 counts
comparison       scripts/final/splice_final.py                            Jan/Jul from v126, Feb/Jun from the final model; upload checks
reads            scripts/final/v128_v129_build.sh                         the two pre-registered reads of 2026-10-08 (v128, v129)
reads            scripts/final/exp_copyswap.py                            v129: the chain's copy classifier on v127, with its window replay
reads            scripts/final/read3_build.py                             v130: the final model's Jan/Jul rows and its pre-registration
candidate        scripts/final/ltfm_local.py                              Istanbul local-state correction: tested, NOT in the entry
reads            scripts/final/read4_build.py                             v131: that candidate's Jan/Jul rows (scored 234.2287)
```

### The clean rebuild: v115 and v116

`REPRODUCE.md` section 12: the same architecture without the constants fitted to the leaderboard, scored once as a
check. `scripts/regen_clean.sh` (the learned gate's base, `clean_base_build.py`, then the trajectory stage in CLEAN
mode) and `scripts/v115_v116_build.sh`; the learned gate's 2026 probabilities are archived in
`artifacts/b4_gate_rk.parquet` (its producer, `scripts/arch/b4_rank.py`, re-draws like the other nets); `scripts/paper_clean_vs_submitted.py` compares it with the submitted chain.

## 2. Published for disclosure: the retired leaderboard-probing code

These scripts implement the probe-and-invert method we used until 2026-09-11 and then retired. They are
kept deliberately so the method is inspectable, and each opens with a RETIRED notice. Do not run them.

```
scripts/v45_build.py  v54_probe.py  v54_audit.py  v54_valcheck.py  v57_probe.py  v63_eham_probe.py
scripts/v65_band_probe.py  v66_build.py  v67_build.py  v69_build.py  window_20260912.py  v34_build.py
```

Not probe code, but carrying constants solved from public scores: the tail chain `scripts/v64_build.py` (section 1;
`REPRODUCE.md` section 9 lists every constant) and its predecessor `scripts/v55_build.py`.

## 3. Instruments and guards used to decide what shipped

```
scripts/presubmit_guard.py     refuses a submission whose evidence class does not match where its error moved
scripts/p1a_instrument.py      the deployed-level estimator that predicted v100 to within 0.67 SE (also the
                               window context the trajectory stages train on, section 1)
scripts/p1a_window_v55.py      the window version of v55_trees.py, which p2p_v55_adsb.py was copied from
scripts/seam_finder.py         the nulled slice sweep behind v95/v96, with its two gates
scripts/paper_*.py fig_*.py    the paper's window measurements and figures
```

## 4. The research log is not published

Comments in the code cite sections of our internal research log by number (`s86`, `s513`, ...). The log is a
month-long working record and is not part of this release; `analysis/RESEARCH-LOG-INDEX.md` gives, for every
section the code cites, its date and one line on what it established. Every fact the reproduction depends on is restated in
`README.md`, `REPRODUCE.md` and `MODELS.md`, and the paper (`analysis/joas/main.pdf`, source `main.tex`) carries the argument. The cover letter
(`analysis/COVER-LETTER.md`) states how the leaderboard was used during the work, and how AI tools were used.
