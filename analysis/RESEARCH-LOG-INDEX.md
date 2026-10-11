# Research-log index

Comments in the code cite sections of our internal research log by number (`s86`, `s513`, ...). The log is a
month-long working record and is not part of this release (`MANIFEST.md` section 4): every fact a reproduction needs is
restated in `REPRODUCE.md`, `MODELS.md` and the paper. This index gives every section the released code cites, with the
date it was written and one line on what it established, so that a citation can be read without the log. Numbers are
as the section recorded them on the day; later sections sometimes revise earlier ones, and the paper and `REPRODUCE.md`
carry the final figures.

```
  s28    2026-09-01   surface structure: the rush-hour backup is real and the features already capture it
  s37    2026-09-02   v5's public score, and the candidate operational columns it prompted (all below noise)
  s47    2026-09-02   month-level drift is real, and a correction for it cannot be validated
  s50    2026-09-02   a leftover prediction cap (CLIP_HI) was truncating the schedule-copy branch
  s61    2026-09-02   the bulk is signal-limited: the decorrelation frontier, measured
  s64    2026-09-02   audit of hardcoded guards, caps and thresholds against the current data
  s75    2026-09-02   the tuning harness measured recall only; fixing it moved the optimum
  s77    2026-09-02   block_dev, the first new feature to improve the bulk
  s78    2026-09-02   sched_vs_hr: the slot-holding (CTOT) signature, worth -1.80 s
  s82    2026-09-02   the pair sweep of feature interactions, and the rule that predicted which would win
  s86    2026-09-04   PRU ATFM regulation data as features: -1.00 s jointly
  s88    2026-09-04   DDR2 is closed; the open PRU catalogue is larger than expected
  s89    2026-09-04   pre-departure delay features: no gain
  s100   2026-09-05   four transfer points between validation and the public score, and the rule consistent with all
  s102   2026-09-05   audit of the commitment rules, and the first hour of tuning
  s103   2026-09-05   the learned five-class gate: a gain on the tail
  s107   2026-09-05   three candidates, each with its prediction recorded before submission
  s121   2026-09-06   the pricing algebra (dSSE = 2T + Q) written out as lemmas
  s124   2026-09-06   v30 scored: the date-rollover cell probed on the public score (retired method; README section 6)
  s181   2026-09-07   a component 1.29 s better gains the blend 0.128 s; the blend weight revisited
  s216   2026-09-09   an external review: the bulk sits about 4 s above its estimated floor, and seven claims were wrong
  s217   2026-09-09   the date-rollover value hypothesis and the calendar test that found it; predicting a row versus
                      reverse-engineering the ranking
  s218   2026-09-09   the corrupted callsign: one string defect explains the failed join and the schedule copy
  s219   2026-09-09   the corrupted-identifier detector, generalised
  s220   2026-09-09   the return-to-stand double push, a second tail mechanism
  s221   2026-09-09   OPDI supplies the airframe; the turnaround constraint still loses, and why
  s222   2026-09-09   timestamp granularity, record ids and cross-table disagreement: no signal
  s223   2026-09-09   airframe features improve the blend, the first new information source after the data reissue
  s224   2026-09-09   the expanded airframe features lose: the signal is the aircraft's turnaround, not its day
  s225   2026-09-09   v55: the airframe model through the shipped chain, with a byte-identical control
  s226   2026-09-10   two submissions measured against a rule already written down
  s227   2026-09-10   v56: two changes weighted from public scores instead of validation (retired; README section 6)
  s228   2026-09-10   the v57 inversion (retired; README section 6); three channels measured empty
  s230   2026-09-10   four alternative architectures, and the movement-window transformer
  s232   2026-09-10   the learned gate against the full hand-built chain on validation
  s234   2026-09-10   v62: the learned gate replaces the hand-built tail chain, shipped with its limitation stated
  s236   2026-09-10   an external review corrects s232; the EHAM disruption week is the largest cell validation cannot see
  s237   2026-09-10   calibration: the right diagnosis and the wrong remedy
  s241   2026-09-10   multi-task learning with the full feature set: the independence came from the restriction
  s243   2026-09-10   the widened movement window
  s244   2026-09-10   a solver bug: the blend solver returned its starting point, so earlier fitted weights were equal
  s247   2026-09-11   the 2026-09-11 window measurement, and a six-row cell
  s254   2026-09-11   corrections applied from the wrong base, solved as a system instead
  s257   2026-09-11   which axis still carries state: the runway queue, whose key is blanked
  s264   2026-09-11   K=128: tuning to standalone accuracy lowers the blend's gain
  s270   2026-09-12   density beats horizon in the movement window
  s272   2026-09-13   the oracle on the six date-rollover rows: an assignment problem, not a mechanism
  s273   2026-09-13   the holdout constants audited one by one; the exposure is regenerability
  s277   2026-09-13   clipped and unclipped RMSE mixed in our own records, corrected
  s280   2026-09-13   what v74 is made of, and a robustness specification
  s281   2026-09-13   v73: the transfer factor for window-family components
  s284   2026-09-14   the "Bayes floor" bounds error from above, not achievable error from below
  s287   2026-09-14   v76, the model without most leaderboard-fitted constants: those constants are worth 2.7501 s
  s289   2026-09-14   the chain reproduces numerically, not bit for bit: the window nets are non-deterministic on Metal
  s290   2026-09-14   A34 = 0.40 came from a leaderboard inversion, not from the labelled prior (0.55)
  s305   2026-09-15   an external review of external-data joins
  s346   2026-09-17   the unclassified tail rows are real durations, and the only predictable ones
  s365   2026-09-18   an audit of an earlier analysis of the public leaderboard; one closure reversed
  s367   2026-09-18   v89 lost; the ratio of validation gain to Q predicts it
  s369   2026-09-18   why the flight-plan join fails: flight-number rarity
  s370   2026-09-18   v90: the rarity discriminator, priced
  s389   2026-09-19   two hypotheses and a near-miss, closed
  s390   2026-09-19   two bugs in our own scripts, corrected
  s391   2026-09-19   January and July are a systematically biased fold
  s393   2026-09-20   v95: a cell found by automated slice search with a permutation null
  s395   2026-09-20   v96: every one of the 185 moved rows was a schedule copy (dSSE = -Q)
  s396   2026-09-20   v97 lost: one carrier's feed was repaired between the years
  s398   2026-09-20   the support-range gate for committed cells
  s401   2026-09-20   the bulk's slice bias is real and not correctable out of sample
  s413   2026-09-20   a mechanism verified on labelled 2026 arrivals
  s417   2026-09-20   the twelve-month cross-fit against the January-July window: select on the scored months
  s421   2026-09-20   identity columns handed to the model cost 3.02 s
  s480   2026-09-21   feature subsampling (colsample) is a lever on the bulk
  s481   2026-09-21   colsample 0.4 on both bulk tree members
  s482   2026-09-21   the remaining non-capacity levers swept on top of colsample: nothing worth adding
  s483   2026-09-21   a baseline error, corrected
  s484   2026-09-21   v99's transfer factor (10.25x)
  s487   2026-09-22   the window instrument differs from the shipped model in four ways, one of them a label leak
  s488   2026-09-22   the 10.25x was the estimator: component-level against deployed-level pricing
  s489   2026-09-22   what the bulk has left, priced on the corrected instrument
  s492   2026-09-22   v100 pre-registered before either build
  s512   2026-09-23   raw adsb.lol surface tracks on one labelled day: pushbacks help
  s513   2026-09-23   the clean-room rebuild, bottom-up; the chain needs the archived MLP predictions
  s513b  2026-09-23   the LightGBM drift has a cause beyond the thread count
  s514   2026-09-23   the window members re-draw like new seeds; the chain pins all five
  s516   2026-09-24   v102 pre-registered: the ADS-B pushback stage
  s517   2026-09-24   v102 scored: +16.07 s, above its pre-registered estimate
  s518   2026-09-24   extending the ADS-B stage; v103 pre-registered
  s518b  2026-09-24   a relaxed time-only matching rule (not shipped)
  s520   2026-09-24   v104 pre-registered: the inbound parking position
  s521   2026-09-24   v104 scored: the lower bound was the forecast
  s522   2026-09-24   where v104's error sits; three leads closed
  s523   2026-09-24   further stage inputs (loss, stale-free training, operator, wake class, gate position): at most 1%
  s524   2026-09-24   the departures ADS-B cannot see: hour-level structure only
  s525   2026-09-25   more ADS-B training data does not close the remaining gap
  s527   2026-09-25   v105 pre-registered: weather where ADS-B cannot see
  s528   2026-09-25   v105 lost: the weather stage does not transfer across years
  s534   2026-09-25   v106 pre-registered: the refined stage alone
  s535   2026-09-25   v106 lost: receiver-geometry features drift across years
  s536   2026-09-26   one AutoML run: FLAML picks XGBoost
  s537   2026-09-26   v107 pre-registered: a 50/50 LightGBM and XGBoost stage
  s538   2026-09-26   v107 scored: a learner change on drift-safe features transfers
  s539   2026-09-26   CatBoost and a neural net as third and fourth stage learners
  s540   2026-09-26   v108 pre-registered: the four-learner stage
  s541   2026-09-26   v108 scored: positive, at 0.26x its forecast
  s543   2026-09-27   v109 pre-registered: the three tree learners without the MLP
  s544   2026-09-27   v109: the tree-only stage transferred as forecast
  s546   2026-09-27   better pushback detection stops at its oracle
  s548   2026-09-28   the full-year adsb.lol pull: per-day manifest and completeness check
  s550   2026-09-28   the full year of adsb.lol streamed (427 days), and the full-year bulk members
  s551   2026-09-28   v110 pre-registered: the full-year members replace the bulk and the stage on covered rows
  s552   2026-09-28   v110 scored: inside its pre-registered range
  s554   2026-09-28   v112 pre-registered: a full-year stage on month-fold out-of-fold residuals
  s555   2026-09-28   the mixture's hedge still pays on covered rows; its cost sits at LIRF
  s556   2026-09-28   v112 scored: the full-year stage paid above its forecast
  s558   2026-09-28   v114 pre-registered: LIRF left at layer 1
  s559   2026-09-28   v114 scored: LIRF is better without the full-year layers
  s560   2026-09-30   the clean models scored: v115 292.0485 and v116 268.1373
  s561   2026-09-30   code review of the clean rebuild: two traces of the fitted offsets, one suspected leak cleared
  s562   2026-10-02   v117 pre-registered: the blend weight refitted outside LIRF
  s563   2026-10-02   v117 scored: the refit paid 1.83x its window linear term
  s564   2026-10-02   more seeds for the scored-side bulk members
  s568   2026-10-02   v118 pre-registered: v117 with the extra seeds
  s572   2026-10-02   v119 pre-registered: LIRF identifiers that never join a flight plan (it lost)
  s574   2026-10-02   v120 pre-registered: THY's Fiumicino feed failed in late 2025, and the 2026 arrivals show it
  s578   2026-10-02   provider outages timed by arrival runs; whether departures follow is a carrier property
  s579   2026-10-02   v121 pre-registered: departures inside 2026 outages, by carrier class
  s586   2026-10-03   v122 pre-registered: the flight-plan join rule, found by an independent agent session from the
                      challenge files only
  s590   2026-10-03   v124 pre-registered: the join rule's second branch
  s592   2026-10-03   the join rule does not generalise beyond LIRF; v125 pre-registered: the local-time record class
  s595   2026-10-03   a second agent round; v126 pre-registered: the day-scale floor on clean identifiers
```
