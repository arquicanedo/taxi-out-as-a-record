# Figures

Every figure is recomputed from the parquet files by its own script. Each script prints a CONTROL
block comparing what it measured against the number the paper records, and asserts on the ones that
must match exactly; a figure that cannot reproduce its paper number fails rather than ships.

Build one with:

```
uv run --quiet --with pandas --with numpy --with pyarrow --with matplotlib python scripts/fig_<name>.py
```

Data comes from `$DC26_DATA` (the challenge extract), the OPDI extract in `data/opdi/`, and, for two figures,
model artefacts in `$DC26_WORK`. `loss-concentration` reads the validation-window mixture (`b4ab_ctx.npz`,
`b4ab_P_u.npy`) and `slice-months` reads the twelve-month cross-fit (`xfit12_bulk.npy`). Both are research
artefacts built from challenge labels, and they are not in this release, so those two figures can be inspected but
not regenerated from it. `model-architecture` reads the same window mixture, the blend weights (`full_package.npz`),
the adsb.lol ground samples of 2025-07-30 (`adsb_stream_day.py` rebuilds that day) and the pushback feature table,
which is not redistributed (ODbL). The other four need only the challenge data and OPDI.

| figure | shows | script | in the paper |
| --- | --- | --- | --- |
| `loss-concentration` | the target's tail over the labelled year, and the cumulative share of squared error by row rank under the deployed model: one row is 20.5% of the window's loss | `scripts/fig_loss_concentration.py` | Figure 5, Section 6.2 |
| `physical-bound` | recorded taxi-out against the turnaround bound from the previous arrival of the same registration, and the violation rate at every tolerance out to 3,600 s | `scripts/fig_physical_bound.py` | Figure 1, Section 4.1 |
| `clock-grid` | the second-of-minute distribution of `BLOCK_TIME` at each of the ten airports, and the exact-minute rate of `BLOCK_TIME` against `MVT_TIME` per airport | `scripts/fig_clock_grid.py` | Figure 2, Section 4.3 |
| `date-error` | the one-day date error drawn from one real Fiumicino row, and the local clock times of all 1,488 Fiumicino `LOBT`-null departures | `scripts/fig_date_error.py` | Figure 3, Section 4.5 |
| `airframe-carryover` | the schedule-copy rate conditioned on the same airframe's previous leg, within airport, against the lift each airport would need before a commitment pays | `scripts/fig_airframe_carryover.py` | Figure 6, Appendix 3.1 |
| `slice-months` | the month-by-month mean out-of-fold residual of the slice search's headline slice, with the row count behind each month | `scripts/fig_slice_months.py` | Figure 7, Appendix 5.3 |
| `model-architecture` | the final model drawn from real rows: one departure's record, the nine members' blend weights, the mixture on one hedged Fiumicino row, every adsb.lol ground sample at Amsterdam on 2025-07-30 with one flight's track, the stage's effect by airport, THY's copied arrivals and departures at Fiumicino, and the RMSE on the 2026 set after each part | `scripts/fig_model_architecture.py` | Figure 4, Section 5 |

Each figure is written as a vector `.pdf` for the paper and a 200 dpi `.png` for reading on a
phone. Typography is Charter at 6 to 10 pt, a serif with lining figures, so a number sits on the baseline beside the
letters around it. The colours come from `scripts/fig_palette.py`, by role. Two hues carry the paper's
reading: a red-orange `#C94001` for the record (schedule copies, extremes, the recorded off-block) and a navy
`#085182` for the measurement (the ADS-B pushback, ordinary rows). They differ in lightness as well as hue, so they
separate in greyscale and under every common form of colour blindness, and both pass 4.5:1 as text on white;
lighter steps of the same two hues and greys carry everything else. No figure encodes meaning in colour alone.
