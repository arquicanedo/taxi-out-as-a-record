# The final phase: two files, one build

On 2026-10-08 the organisers added February and June 2026 to the scored months. They asked for a single final file
covering January, February, June and July, ranked without feedback three ways: on January and July, on February and
June, and on all four. This repository delivers two files. Both are built by one script and differ only on January
and July.

| File | What it is | January and July | February and June | Uploaded |
|---|---|---|---|---|
| `quick-boat_final.parquet` | **the not-fitted model** | the not-fitted model; leaderboard **233.7464** (v130) | the not-fitted model | **yes, the entry** |
| `quick-boat_final_with_v126.parquet` | the competition model, for comparison | v126, **with** the constants fitted to the public leaderboard (REPRODUCE.md section 9); leaderboard **222.8507** | the not-fitted model | no |

**The not-fitted model is our final model.** It is the model of REPRODUCE.md sections 12 and 13, rebuilt with **no
value fitted to a score on the public leaderboard**, plus the chain's rules whose evidence is labelled. It is retrained
once on the twelve 2025 files and predicts the four months of `final_ranking.parquet` together, with one code path.
Every constant on its path, and where it comes from, is in the table below. Its January and July rows were scored
once on the leaderboard (v130, 233.7464) against an expectation written down before the upload (239.25, 95% band
232.60 to 245.73).

**Which number measures the method: 233.7464**, the final model's. The competition model's 222.8507 is not comparable
and not a reference: it includes values fitted to the leaderboard on the same months it is scored on.

**The competition model** is v126, the lowest January and July score this work reached. The
recording failures the paper characterises (schedule copies at Fiumicino, carriers whose data feeds fail, date errors and the flight-plan join) were found by studying its errors,
and the final model keeps them as rules whose evidence is labelled. It was pushed with everything the leaderboard offered: scalars obtained by probing the scored set until 11 September, and
changes kept or dropped on their scores. All of it is listed in REPRODUCE.md section 9. The 10.90 s between v126 and
v130 on January and July holds the fitted constants, the extra seeds, the hand-built chain's remaining cells and the
retraining, together (the paper's Section 7). The fitted values cannot carry to February and June, so the file takes
those months from the final model.

v126 was built from the extract of 2026-09-04. `final_ranking` moves every take-off time by up to 12 s, so the 509
v126 predictions that are a function of the take-off time are recomputed from `final_ranking`:
- 417 schedule copies, MVT - SCHED;
- 63 at MVT - LOBT;
- 29 at a flight-plan bound, MVT - LOBT -+ 3606.

Those 509 rows are identified exactly on the extract v126 was built from (`scripts/final/splice_final.py`).

## Rebuild

```
ADSB_MONTHS="2026.01 2026.02 2026.06 2026.07" scripts/adsb_pull_all.sh     # the 2026 adsb.lol days; 2025: adsb_pull_year.sh
uv run --with pandas --with pyarrow python scripts/pull_opdi.py           # OPDI flight lists, incl. 2026-03 and 2026-06

export DC26_DATA=<folder with final_ranking.parquet as ranking.parquet, final_submitting.parquet as submitting.parquet,
                  and the twelve training_2025-*.parquet>
export DC26_WORK=<empty> DC26_TMP=<empty> DC26_SESSION=<empty> ADSB_DAYS=<the parsed days>
export FABLE_WINDOW=<v121_window.npz>  OLD_DATA=<the 2026-09-04 extract, read only to identify v126's anchored rows>
bash scripts/final/run_final.sh                          # 12 steps; FROM=N resumes, ONLY=N runs one, DRY=1 lists them
```

Step 11 writes `quick-boat_final.parquet` (the entry) and step 12 writes `quick-boat_final_with_v126.parquet`. The neural members
re-draw on every run, so a rebuild reproduces the files numerically; the shipped predictions are archived (sha256 below).

## Every value on the not-fitted path (the entry)

| Step | Value | Source |
|---|---|---|
| Base mixture | learned five-class gate (`arch/b4_rank.py`), pi = P^0.90, floor 0.10 | trained on 2025; replaces section 9's A34, A49, A51, A54, A57, v29 and v90 weights |
| Bulk weights | nine members, refitted with a free intercept per month (`WF=refit`) | labelled 2025 window |
| Month offsets | January and July from the window (step 9 prints them; -10.6564 / -0.5278 with `WF=refit` on Jan/Jul); February +3.800, June +2.727 | mean residual on ordinary 2025 rows; Feb/Jun from the twelve-month cross-fit (`scripts/xfit12.py`, SE 0.64 / 0.60) |
| Seam rule v95 | LIRF stand-4xx, no LOBT, schedule deviation >= 7,200 s, moved to the cell's median taxi (780 s) | 28 labelled rows, 0 copies |
| Seam rule v96 | eleven LIRF carriers committed to the copy where that lowers expected squared error at P = 0.9911 | 1,688 labelled rows; months 1-6 98.44%, months 7-12 99.59% |
| Trajectory stage | v102 to v114, each variant as the clean chain shipped (`PIN_PICKS=1`) | weights fitted on the 2025 window; the four steps whose variant was chosen by a comparison on scored rows keep that variant |
| v117 | blend weight outside LIRF (0.837 on the submitted chain; the value refitted on this base is printed by step 10) | fitted on the 2025 window's covered rows |
| Rules v120 to v126 | REPRODUCE.md section 13 | rates on labelled 2025 rows; carrier feed state from the scored months' labelled arrivals |
| LIRF day-scale | no LOBT, schedule deviation >= 20,000 s: by band, the 2025 counts of copy / date error / ordinary (`scripts/final/lirf_dayscale.py`) | 70 labelled rows (paper Table final_bands) |

**Traces that remain, measured:**
- The stage's XGBoost settings (`artifacts/v107_xgb_params.json`) were searched on labelled 2025 data, against the
  residual of a base that carried the leaderboard month offsets (REPRODUCE.md section 12).
- The Fable LIRF classifier trains on the 2025 window against the submitted chain's window vector (`FABLE_WINDOW`).

## Measured before the upload (January and July leaderboard)

Each submission was written down with its expected score and a decision rule before it was made
(`analysis/final/PREREGISTRATION.md`; the files are tracked in `submissions/`).

| Submission | What it tests | Expected | Scored |
|---|---|---|---|
| v128 = v127 | the leaderboard's labels are unchanged | 247.7412 | 247.7412 |
| v129 | the submitted chain's copy classifier, recalibrated on labelled data, in place of the gate | 247.41 [244.34, 250.45]; adopted only below 246.18 | 248.2763: not adopted |
| v130 | the not-fitted model's own January and July rows (four-month build) | 239.25 [232.60, 245.73] | 233.7464 |

## Files

```
submissions/quick-boat_final.parquet            97f3819fcc04c1f4f2994acaaeb509fa89315944841c5340fcfb8b3109df9cab   the entry (candidate 1)
submissions/quick-boat_final_with_v126.parquet  16cfd173334db2ae9d28924217a04047700d739965d36f380a175e121dbca2f9   the competition model (candidate 1)
```
