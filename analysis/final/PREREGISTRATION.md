# Pre-registration of the final phase's three leaderboard reads

Each read was written down, with its expected score and decision rule, and committed to the working repository before
its file was uploaded. The blocks below are copied verbatim from those commits. The working repository's commit times
and the leaderboard's processing times (UTC, 2026-10-08):

```
  reads 1 and 2 registered   15:58:46     v128 scored 15:59:36     v129 scored 16:02:25
  read 3 registered          21:13:15     v130 uploaded 21:14:47   v130 scored 21:15:20
```

The numbers of read 2 come from `scripts/final/exp_copyswap.py` (its full pre-registration is
`analysis/final/prereg_v129.json`); those of read 3 from `scripts/final/read3_build.py`. No value in either final file
changes with these scores. Read 2 decided one design choice by a rule fixed beforehand, and its answer was to keep the
learned mixture.

### Pre-registration of reads 1 and 2 (committed before either upload, 2026-10-08)

```
read 1   quick-boat_v128.parquet = v127 byte for byte (sha256 74fcf296...)
         expected 247.7412 exactly if the leaderboard's truth is unchanged since 2026-10-05.
         Any other score means the truth moved; read 2 is then judged against v128's score, not 247.7412.

read 2   quick-boat_v129.parquet = S (sha256 11fe7fc7..., scripts/final/exp_copyswap.py, analysis/final/prereg_v129.json)
         alpha 1.00 (labelled window, pooled; cross-fit unstable: Jan-fitted 0.35 loses on July, Jul-fitted 1.00 loses
         on January). Window replay at base level: dSSE -5.64e7, day-clustered SE 2.66e8 (no detectable effect).
         S vs v127: 159 rows moved more than 600 s, Q 4.8725e8 (LIRF 3.07e8), 4 rows held by the 5% cap, largest 4.6%.
         PREDICTION (the window's own gain, because Q_w = 2.2 x Q_b; CLAUDE.md):
             247.4103, 95% [244.3358, 250.4471], 3 SE [242.6886, 252.0436]
         For the record, the 2026-anchored estimate: 243.95, 95% [240.29, 247.54].
         DECISION RULE (replaces "S scores below v127"):
             the copy swap goes into the Feb/Jun model only if SSE(v129) < SSE(v128) - 2.66e8, one window SE;
             with v128 at 247.7412 that is a score below 246.1795. Otherwise Feb/Jun keeps the gate.
         STOP: a score outside the 3 SE band means the build or the instrument is broken; nothing ships until explained.
```

Neither score is used for anything but these two rules. Read 3 is pre-registered when the four-month pipeline exists.

**Results (2026-10-08).**
```
read 1   quick-boat_v128  15:59:36Z  247.7412   = v127 exactly: the leaderboard's truth is unchanged
read 2   quick-boat_v129  16:02:25Z  248.2763   dSSE +9.15e7 against v128; inside the 95% band, so no stop
         decision: 248.2763 is not below 246.1795, so Feb/Jun KEEPS THE GATE. The chain's copy model, recalibrated on
         labelled data, does not beat the learned gate on the 2026 rows, as the window had said (no detectable effect).
```
No score below is used to set any value.

### Pre-registration of read 3 (committed before upload, 2026-10-08)

```
read 3   quick-boat_v130.parquet: the four-month not-fitted model's own January and July rows (scripts/final/read3_build.py),
         from the build of scripts/final/run_final.sh steps 1-11 (candidate 1, finished 21:11:56Z)
         against v127: 331,218 rows differ, RMS 68.54 s, 72 rows by more than 600 s, Q 1.6198e9
         EXPECTATION from labelled rates only (v127's 247.7412 plus):
           27 LIRF day-scale rows (band A 21, B 5, C 1)   expected dSSE -1.3744e9, SD 5.4143e8 (2025 band rates)
           279 v96 rows                                   expected dSSE -5.10e7 (copy at 0.9911)
           every other row                                Q 1.9531e8 over 330,912 rows, expectation 0 (not priced)
         PREDICTION 239.2533;  +-2 SD [232.5982, 245.7281];  STOP outside +-3 SD [229.1983, 248.9024]
```

Read 3 is a measurement of the not-fitted model, not a selection: no value in either file changes with its score.
A score outside the stop band means the port is broken, and nothing is uploaded until that is explained. The
day-scale rows carry most of the read's variance, so the score also measures whether January and July's
27 day-scale rows followed the 2025 band rates. It is reported as such, and no rate is refitted from it.

**Result (2026-10-08 21:15:20Z): v130 = 233.7464.**
- dSSE against v127 is -2.3236e9, against an expected -1.4254e9: -1.66 SD of the day-scale term, inside the +-2 SD band.
- The not-fitted model recovers 13.9948 of the 24.8905 board-s between v127 and v126 (56.2%).
