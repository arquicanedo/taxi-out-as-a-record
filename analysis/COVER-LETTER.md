# Cover letter: how the leaderboard and AI were used

Every number below is from this repository (https://github.com/arquicanedo/taxi-out-as-a-record): `REPRODUCE.md` sections 9, 12 and 13, the paper (`analysis/joas/main.pdf`) and
`scripts/paper_clean_vs_submitted.py`.

---

To the editors, and to the organisers of the PRC Data Challenge 2026,

We submit "Taxi-out time as a database record" and want to state plainly how we used the leaderboard during the work,
what that did to the models we submitted, and what we think it shows about the challenge, and then how the work was done
with AI tools.

**Two models.** Our entry is the final model: one design with no value fitted to a returned score, run on all four
months of the final phase. Its January and July rows scored 233.7464, and that is the number that measures the method.
The competition model, v126 (222.8507), is the same design pushed with everything the leaderboard offered. The
recording failures the paper characterises (schedule copies at Fiumicino, carriers whose data feeds fail, date errors and the flight-plan join) were found by studying its errors,
and the final model keeps them as rules whose evidence is labelled. Its score includes values fitted to the months it
is scored on and is not comparable. What follows is how the leaderboard was used for it.

**What we did.** Until 11 September 2026 we sometimes used the leaderboard as a measuring instrument. We submitted a
prediction that differed from an already scored one by a known perturbation, read the change in the returned score,
and solved for the value of a constant that the change implied (the identity `dSSE = 2T + Q` of the paper's Section
7.1). Seven scalars and three smaller adjustments in our base model were obtained this way, and two further choices were
informed by scores without being solved from them. The repository's `REPRODUCE.md` (section 9) lists every one and what
it weights. We did this to improve our standing. We stopped on 11 September, when the organisers said in the competition
channel that they do not want the ranking reverse-engineered, and did not solve for a constant again. After that, every
submission was a model whose score we had predicted beforehand. We still kept or dropped changes on their scores, and for
two rules (v96, v120) we read from the score that every row they moved was a schedule copy: the squared error fell by the
rule's own squared shift. `REPRODUCE.md` section 9 lists both. The paper discusses these as limitations (Section 8). The scripts that did the probing are included in the released code and
marked as retired. In all we made 90 submissions.

**What it was worth.** Removing seven of the fitted constants from a submitted model cost 2.7501 s of RMSE on the scored
set. Three others were present on both sides of that comparison, so their full contribution is larger than that. In
squared error those constants are about 7% of the gap between the submitted base and the rebuild below, and six later
rule-based steps of the chain another 12.5%. Most of the rest is the chain's handling of Fiumicino's schedule copies and
date errors, which was built from labelled data, tuned with scores, and never measured on its own.

**How we checked the paper against it.** On 30 September we rebuilt the model without the constants fitted to
leaderboard scores. On the scored rows, the base model's learned mixture replaced the hand-built chain that carried the constants;
its month offsets were estimated on labelled 2025 data; and the trajectory stage was retrained on that base by the same
rules. We submitted the rebuild once, base and final, only to measure it:

| | base model | with the trajectory stage as first submitted | squared error removed by the stage |
|---|---:|---:|---:|
| submitted models | 254.5228 | 226.8129 | 4.5994e9 |
| rebuild without the fitted constants | 292.0485 | 268.1373 | 4.6190e9 |

The trajectory stage, the paper's main modelling contribution, removes almost exactly the same squared error from
either base, so it does not depend on the fitted constants. The base is a different matter. The rebuild's base
scores 37.53 s worse, and the two bases' predictions differ mainly on 225 rows, 174 of them at Rome Fiumicino. On those
rows the submitted chain enforces rules about how Fiumicino writes its records that the learned mixture does not, and it
also carries fitted constants. These two scores cannot separate the two.

A code review of the rebuild afterwards found one trace of the fitted month offsets in it: the bulk's blend weights
had been fitted against a target shifted by them, and the trajectory stage's XGBoost settings were searched, by
cross-validation on labelled 2025 data, against the residual of a base that carried them. Refitting the weights without
them moves the rebuilt base's predictions by 0.45 s RMS, which bounds the change in its score, and the rebuilt final
model's by 3.86 s RMS as the stage retrains; on the labelled window that model changes by 0.023 s. We did not resubmit.

Our competition model, which scored 222.8507, adds to the submitted stage a refitted blend weight, more ensemble seeds, two
rules that read carriers' data feeds at Fiumicino from the scored months' labelled arrivals, rules that follow from
how the extract joins a departure to its flight plan, a copy classifier for Fiumicino's coin-flip rows, its
local-time record class, and a floor on its day-scale delays without a plan (the paper's Sections 4.7 and 5.3). None of these was fitted to a returned score: the weight was fitted on the labelled
window, the seeds were chosen from the scored rows' own spread between seeds, and the rules' rates and the
classifier come from labelled 2025 data. Each was submitted once
with its predicted score written down beforehand. The two rules blend with the base model's bulk, which carries two of
the fitted month offsets.

**The final phase.** At the close the organisers added February and June 2026 to the scored months. The paper's final
model is the design with no value fitted to a returned score: the learned mixture, month offsets from labelled data,
the two cells of the hand-built chain whose evidence is labelled, the trajectory stage and the rules above, retrained
once and run on all four months. Its January and July rows scored 233.7464 against 239.25 predicted beforehand. It is
our entry for the final ranking, and the competition model ships beside it (the paper's Section 7).

**What we think it shows.** Our submitted base and the rebuild differ mainly on a few hundred records, most of them at
one airport. Fiumicino wrote 480 of the 584 extreme records in the 2025 labelled data, 29.87 per 10,000
departures against at most 1.54 at the other nine airports, and its extreme records are mostly failures of record
keeping rather than long taxis: schedules copied into the off-block field, and dates resolved against the wrong day.
Our observables identify some of these rows and miss others entirely, and our labelled window could not tell us how to
treat them; the only further information about them we had was the score. Iterating against it drew our choices toward
those rows, including the choices we made without solving for a constant.

For future editions we suggest one or more of the following:

- score Fiumicino's departures separately, or leave them out of the ranking;
- report beside the RMSE a companion metric on rows below a threshold such as 7,200 s, which the organisers can compute
  because they hold the labels;
- split the scored set into a public part for the leaderboard and a private part for the final ranking.

The paper discloses this in Section 5.1 and measures the model without the fitted constants in Section 6.3.
The repository rebuilds every submitted model and the rebuild above, each checked against its file's hash.

**Use of AI.** This work was done with the Claude Code agent (Anthropic) running Claude Opus 5 and 5.5, with Claude
Fable 5.1 and Sonnet 5 subagents for independent review and verification; the agents wrote the code, ran the analyses
and drafted the paper under the author's direction and review. Over the logged sessions, 8 September to 5 October 2026
(the project's first week predates the logs), the author directed them with 979 prompts in 16.5 hours of interaction,
measured from the logs with at most 75 seconds counted per reply; a language-model classifier judged 338 of the prompts
(35%) to introduce something the agents had not raised. The agents produced 13.95 million output tokens and worked
143.6 hours: 45.0 generating, 29.9 in subagents running in parallel, and 68.7 in computation on one laptop (Apple M5
Pro), 41.1 on its CPU and 27.6 on its GPU. The paper carries the same statement.

Sincerely,

Arquimedes Canedo, team `quick-boat`
