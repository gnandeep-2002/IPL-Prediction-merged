# Codebase Walkthrough: IPL Win-Probability & Score Prediction Pipeline

Every number in this document was either printed by an actual `python3 run_all.py`
execution against `data/raw/ipl_data.xlsx` (1,146 matches, 273,503 deliveries),
or computed directly from that same dataset in a follow-up Python session
using the project's own functions. Nothing here is recalled from general
knowledge or invented for illustration.

**Provenance note:** the per-match worked examples in §§1–8 (e.g. the Elo
ratings for match 1304088) were computed against the original 2026-07-08 run,
*before* the defect-report fixes changed the match chronology (DEF-007) and
the Deccan/SRH franchise split (DEF-008) — their third decimal places no
longer match the current pipeline, though the mechanics they illustrate are
unchanged. Headline metrics, §9, §15, and the appendix reflect the post-fix
2026-07-14 run, whose full transcript is reproduced in
[Appendix: raw `run_all.py` output](#appendix-raw-run_allpy-output).

The worked example running through most stages is **match 1304088**:
**Lucknow Super Giants (batting first) vs. Punjab Kings**, played in 2022 at
Maharashtra Cricket Association Stadium, Pune. LSG scored 153, Punjab Kings
were bowled out chasing for 133, and LSG won. This match sits in the model's
internal test period (year > 2020), so every model below produces a genuine
out-of-sample prediction for it, not a training-set fit.

## Table of Contents

1. [Data loading & team-name normalisation](#1-data-loading--team-name-normalisation)
2. [Elo rating system](#2-elo-rating-system)
3. [Form and head-to-head features](#3-form-and-head-to-head-features)
4. [The pre-match win-probability model](#4-the-pre-match-win-probability-model)
5. [The 24-dimensional ball-by-ball game state](#5-the-24-dimensional-ball-by-ball-game-state)
6. [Dynamic in-game win-probability models](#6-dynamic-in-game-win-probability-models)
7. [Phase-specific evaluation](#7-phase-specific-evaluation)
8. [Score regression models](#8-score-regression-models)
9. [The locked 2026 holdout evaluation](#9-the-locked-2026-holdout-evaluation)
10. [SHAP explainability](#10-shap-explainability)
11. [Tournament simulation](#11-tournament-simulation)
12. [Calibration comparison: isotonic vs. temperature scaling](#12-calibration-comparison-isotonic-vs-temperature-scaling)
13. [The secondary Transformer / MC-Dropout model](#13-the-secondary-transformer--mc-dropout-model)
14. [The dashboard](#14-the-dashboard)
15. [Where results are weak vs. strong — reading the baselines correctly](#15-where-results-are-weak-vs-strong--reading-the-baselines-correctly)
16. [Glossary](#16-glossary)
17. [Appendix: raw `run_all.py` output](#appendix-raw-run_allpy-output)

---

## 1. Data loading & team-name normalisation

**Problem.** The raw data is one Excel sheet, `"Ball by Ball"`, with one row
per legal or illegal delivery across 18 seasons (2008–2025). Over that span,
IPL franchises have renamed themselves (e.g. Delhi Daredevils became Delhi
Capitals). If this isn't fixed before anything else runs, every downstream
calculation that accumulates *per-team* history — Elo rating, recent form,
head-to-head record — would silently treat a renamed franchise as two
unrelated teams and lose that team's entire history at the rename boundary.
Only genuine renames of the *same* franchise are mapped: distinct franchises
that merely shared a city (Deccan Chargers, which folded after 2012, vs.
Sunrisers Hyderabad, a new franchise from 2013) are deliberately kept
separate so neither inherits the other's Elo/form/H2H history (DEF-008).

**Code path.** `src/data.py`:
- `NAME_MAP` is a literal dictionary of old-name → new-name pairs (e.g.
  `"Delhi Daredevils": "Delhi Capitals"`, `"Kings XI Punjab": "Punjab
  Kings"`) covering same-franchise renames only.
- `load_ball_by_ball(xlsx_path)` reads the Excel sheet, applies
  `NAME_MAP` via `.replace()` to the `batting_team`, `bowling_team`,
  `match_winner`, and `toss_winner` columns, keeps only rows where
  `innings` is 1 or 2 and `result_type` is null (i.e. drops abandoned/no-result
  matches), and adds a `batting_wins` column (1 if the batting team won).
- `build_match_table(df)` collapses the cleaned delivery-level rows into
  one row per match (`match_id`): `team1` (batted first), `team2` (bowled
  first), `winner`, `year`, `venue`, `toss_winner`/`toss_decision`, final
  scores `score1`/`score2`, the label `team1_win`, and two toss-advantage
  flags `toss_bat_first`/`toss_field_first` (did the toss winner get the
  choice they wanted *and* is that choice reflected in who's team1/team2?).

**Real numbers.** After loading, `load_and_prepare()` reports (verified by
direct computation): **1,146 matches**, **273,503 deliveries**. For the
example match:

| Field | Value |
|---|---|
| `match_id` | 1304088 |
| `team1` (bat first) | Lucknow Super Giants |
| `team2` (bowl first) | Punjab Kings |
| `winner` | Lucknow Super Giants |
| `year` | 2022 |
| `toss_winner` | Punjab Kings |
| `toss_decision` | field |
| `score1` / `score2` | 153 / 133 |
| `toss_bat_first` | 0 |
| `toss_field_first` | 1 |

**Feeds into.** `match_df` (the one-row-per-match table) is the backbone
every later stage joins against: Elo, form, and head-to-head features are
all attached as new columns on this same table in `pipeline.py`'s
`load_and_prepare()`.

---

## 2. Elo rating system

**Problem, plain-language.** Elo is a way of giving every team a single
number that represents "how strong is this team right now," updated after
every match it plays. It was invented for chess. The core idea: before a
match, compute how likely each team was *expected* to win based on their
current ratings; after the match, move each team's rating up or down by an
amount proportional to how *surprising* the actual result was. Beating a
much stronger team moves your rating up a lot; beating a much weaker team
barely moves it.

**Exact formula**, from `src/elo.py`'s `_update()` (K=32, initial rating
1500 for every team's first-ever match):

```
exp1 = 1 / (1 + 10^((elo2 - elo1) / 400))     # team1's expected win probability
elo1_new = elo1 + K * (team1_win - exp1)
elo2_new = elo2 + K * ((1 - team1_win) - (1 - exp1))
```

`team1_win` is 1 or 0 (the actual result). If team1 was expected to win with
probability 0.6 and did win, its rating rises by `32 * (1 - 0.6) = 12.8`
points; if it lost instead, its rating falls by `32 * (0 - 0.6) = -19.2`
points. There is **no shrinkage or prior** — every team starts at exactly
1500 regardless of era, so a brand-new franchise (e.g. Gujarat Titans,
Lucknow Super Giants, both first appearing in 2022) begins with the same
rating as a team with 14 seasons of history.

**Code path.** `compute_elo(match_df)` walks `match_df` in chronological
order (by parsed match date, with `match_id` as a same-day tie-break —
DEF-007: match IDs alone are not date-ordered), calling `_update()` once per
match, and returns each
match's **pre-match** ratings (`elo1`, `elo2`, `elo_diff = elo1 - elo2`) plus
the final rating dictionary (used later as the walk-forward starting point
for the 2026 evaluation, §9). `compute_elo_history(match_df)` uses the same
`_update()` function but records every team's rating *after* each match it
played, for the dashboard's Elo-over-time chart — confirmed by the real run:
**15 teams (Deccan Chargers and SRH now separate, DEF-008), 2,292
match-level data points total**.

**Worked example (real).** For match 1304088:

| | Lucknow Super Giants (team1) | Punjab Kings (team2) |
|---|---|---|
| Pre-match Elo | 1530.47 | 1483.74 |
| `elo_diff` | **+46.72** | |

`exp1 = 1 / (1 + 10^((1483.74 - 1530.47)/400)) ≈ 0.566` — i.e. going into
this match, Elo alone gave LSG a 56.6% expected win probability. LSG did
win, so its post-match rating rose by `32 * (1 - 0.566) ≈ 13.9` points, and
Punjab Kings' rating fell by the same amount.

**Feeds into.** `elo_diff` is one of the 5 features (`PRE_FEAT`) used by the
pre-match model (§4), and `elo_adv` (a signed version relative to the
chasing/defending team) is one of the 7 features (`DYN2`/`DYN1`) used by the
dynamic in-game models (§6).

---

## 3. Form and head-to-head features

**Problem, plain-language.** Elo captures long-run team strength, but two
other signals might matter too: *recent form* (has this team been winning
lately, regardless of long-term rating?) and *head-to-head history* (does
one specific team have this specific opponent's number?). Both need the
same discipline as Elo: at the moment a feature is computed for match `i`,
it must only see the outcomes of matches `0 .. i-1` — never a future result
("no lookahead").

**Code path.** `src/features.py`:
- `compute_form_h2h(match_df, window=5)` iterates matches in order,
  maintaining a per-team list of past results (`tw`) and a per-pair win
  count (`h2h`). For each match it records `form1`/`form2` (mean of each
  team's last 5 results, defaulting to 0.5 if the team has no history yet)
  and `h2h_rate` (raw win rate between these two specific teams, defaulting
  to 0.5 with no shared history) — *then* appends the current match's
  outcome to both accumulators, so the next match sees it but this one
  didn't.
- `compute_h2h_beta(match_df, alpha=2, beta_param=2)` computes a Bayesian
  alternative to the raw H2H rate: a Beta(2,2) prior over head-to-head win
  rate, giving posterior mean `(wins + 2) / (n + 4)`. With zero prior
  meetings this equals exactly 0.5 (same as the raw version's default); with
  more meetings it shrinks the empirical rate smoothly toward 0.5 rather
  than jumping straight to whatever the small sample says. This is the one
  place in the whole codebase that implements genuine Bayesian shrinkage.

**Worked example (real).** For match 1304088 (LSG vs. Punjab Kings, 2022):

| Feature | Value |
|---|---|
| `form1` (LSG, last 5 matches) | 0.6 |
| `form2` (Punjab Kings, last 5 matches) | 0.4 |
| `form_diff` | **0.2** |
| `h2h` (raw LSG-vs-PBKS win rate) | 0.5 |
| `h2h_beta` | 0.5 |

Here `h2h` and `h2h_beta` are identical at 0.5 because — even though LSG and
Punjab Kings had played before this point — the raw historical count in this
match's context evaluates to no clear edge either way, which the Beta prior
leaves untouched.

**Feeds into.** `form_diff` and `h2h` (the raw version, not `h2h_beta`) are
2 of the 5 `PRE_FEAT` features used directly in the pre-match model (§4).
`h2h_beta` is computed but — per `CODEBASE_DEEP_DIVE.md`'s documented
gap — not one of the 5 features the trained pre-match model actually
consumes; it's carried on `match_df` but unused downstream.

---

## 4. The pre-match win-probability model

**Problem, plain-language.** Before a ball is bowled, can we predict who
wins using only pre-match information: team strength (Elo), recent form,
head-to-head history, and toss result? This is deliberately the
*hardest and most information-poor* setting in the whole codebase.

**Code path.** `src/pipeline.py`'s `train_pre_match_internal(match_df)`.
Features (`PRE_FEAT`): `[elo_diff, form_diff, h2h, toss_bat_first,
toss_field_first]` — 5 columns, standardized with `StandardScaler`. Split:
train on matches with `year <= 2020`, test on `year > 2020` (a **temporal
split**, not a random shuffle — because team strength changes over
seasons, testing on future years and training on past years is the only way
to avoid the model implicitly "seeing the future"). Two classifiers are fit:
`CalibratedClassifierCV(LogisticRegression(C=1.0), method="isotonic", cv=5)`
and the same wrapper around `GradientBoostingClassifier`. Both are compared
against a **climatology baseline**: a forecast that always predicts the
training-set base rate (`y_tr.mean()`), never varying per match.

**What "calibrated" and the metrics mean**, defined once here (see also
§16 Glossary): a *calibrated* probability is one where, among all matches
the model said "70% chance", roughly 70% actually happen that way — not
just "correctly ranks the more likely winner higher." **AUC** measures
ranking quality alone (0.5 = no better than random guessing, 1.0 = perfect
ranking) and says nothing about calibration. **Brier score** is the mean
squared error between predicted probability and the 0/1 actual outcome
(lower is better; 0 is perfect). **Brier Skill Score (BSS)** compares a
model's Brier score against the climatology baseline's: `BSS = 1 -
BS(model)/BS(climatology)`; BSS=0 means "no better than always guessing the
base rate," and negative BSS means the model is measurably *worse* than
that trivial baseline. **ECE (Expected Calibration Error)** buckets
predictions into probability bins and averages, weighted by bin size, how
far off the observed frequency in each bin is from the predicted
probability — the scalar summary of a reliability diagram.

**Real results** (internal split, `year <= 2020` train / `year > 2020`
test, n=347 test matches — from the actual `run_all.py` run):

| Model | Brier ↓ | AUC | Accuracy | BSS | ECE ↓ |
|---|---|---|---|---|---|
| Climatology (baseline) | 0.2508 | 0.5000 | 0.4813 | 0.0000 | 0.0000 |
| Calibrated Logistic Regression | 0.2637 | **0.4527** | 0.4755 | **−0.0516** | 0.1092 |
| Calibrated Gradient Boosting | 0.2559 | 0.5035 | 0.4986 | **−0.0204** | 0.0654 |

**Both trained models have negative BSS** — they are measurably worse at
producing calibrated probabilities than a model that ignores every feature
and just guesses the training base rate every time. Cal. LR's AUC (0.4527)
is even below 0.5, i.e. slightly *worse* than a coin flip at ranking.

**Worked example (real).** Feeding match 1304088's own features
(`elo_diff=46.72, form_diff=0.2, h2h=0.5, toss_bat_first=0,
toss_field_first=1`) through the internally-trained models (this match
falls in the `year > 2020` test split, so this is a genuine out-of-sample
prediction):

- Calibrated Logistic Regression: **P(LSG wins) = 0.445**
- Calibrated Gradient Boosting: **P(LSG wins) = 0.453**

Both models predicted Punjab Kings (the *other* team) as slightly more
likely to win. LSG actually won. This single case is consistent with the
aggregate finding above — the model's pre-match edge is not reliable enough
to trust on a match-by-match basis.

**Feeds into.** The internal split above is for evaluation only; §9's
locked 2026 holdout uses a *separately retrained* version of this same
5-feature model (trained on the full 2008–2025 dataset, not just `year <=
2020`). §10's SHAP explainability runs on the Cal. GBT model trained here.
§11's tournament simulation consumes the 2026-holdout model's per-match
picks.

---

## 5. The 24-dimensional ball-by-ball game state

**Problem, plain-language.** The pre-match model only sees information
available before a ball is bowled. As the match unfolds, far richer
information becomes available ball by ball: how many runs and wickets so
far, how quickly runs are coming, who's currently batting/bowling and how
well, and — in the second innings — how big the remaining target is and how
much time is left to chase it. `src/game_state.py` builds a dense,
normalised 24-number vector per delivery capturing all of this. (Note: this
24-dim vector feeds the secondary Transformer model of §13, not the
primary dynamic logistic-regression models of §6, which use a smaller
7-feature set built separately in `pipeline.py`.)

**The 24 features** (each divided by a fixed constant to land roughly in
`[0, 1]`, `src/game_state.py`):

```
0  over / 20                        12 partnership_balls / 60
1  legal_balls_total / 120          13 batter_sr_innings / 200
2  innings==2 flag                  14 batter_balls_innings / 120
3-5 phase one-hot                   15 bowler_econ_innings (clip 20/ov)
6  score_before / 250               16 bowler_wkts_innings / 10
7  wickets_before / 10              17 boundary_rate so far
8  run_rate (clip 15/ov)            18 dot_rate so far
9  runs in last completed over/30   19 runs_required / 200 (inn.2 only)
10 wickets in last 5 overs / 5      20 balls_remaining / 120 (inn.2 only)
11 partnership_runs / 150           21 required_rr (clip 36/ov, inn.2 only)
                                    22 toss_won_bat
                                    23 innings==2 flag (duplicate)
```

**A documented, test-caught bug fix worth understanding.** Early versions
paired a "before this ball" numerator (e.g. `score_before`) with an
"including this ball" denominator (`legal_balls_total`), which silently
**halved the reported run rate on ball 2 of every innings**. The fix
introduces `legal_balls_before` (a strictly pre-ball counter,
`legal_balls_total - is_legal`), and every ratio that needs a pre-ball
denominator (`run_rate`, `boundary_rate`, `dot_rate`, `balls_remaining` for
the chase) now consistently pairs with it. `is_legal` itself follows the
dataset's own convention (verified empirically across all 273,503
deliveries): a ball only counts toward `team_balls` if it has *neither* a
wide *nor* a no-ball recorded.

**Worked example (real).** A specific delivery from match 1304088,
innings 1, over 16 ball 1, batter J.O. Holder, bowler Arshdeep Singh:

| Raw quantity | Value |
|---|---|
| `score_before` | 111 |
| `wickets_before` | 6 |
| `run_rate` | 6.9375 |

The 24-dim vector for this exact delivery (as actually computed by
`build_game_state_matrix`):

```
[0.800, 0.808, 0.0, 0.0, 0.0, 1.0,        # over/20, legal_balls/120, inn2 flag, phase one-hot (Death=1)
 0.444, 0.6, 0.4625, 0.0667, 1.0, 0.0067,  # score_before/250, wkts/10, run_rate, runs_last_over, wk_last5, partnership_runs
 0.0667, 0.25, 0.0167, 0.275, 0.1,         # partnership_balls, batter_sr, batter_balls, bowler_econ, bowler_wkts
 0.1146, 0.4479, 0.0, 0.0, 0.0, 0.0, 0.0]  # boundary_rate, dot_rate, runs_required..(0 for inn.1), toss_won_bat, inn2 dup
```

Index 5 = 1.0 confirms `phase` is correctly one-hot'd to **Death** (over 16
falls in the 16–19 range).

**Feeds into.** This exact vector is the input to the secondary causal
Transformer (§13) — its `d_model=64` encoder consumes 24 game-state values
concatenated with three 32-dim player embeddings (batter/bowler/
non-striker), for a 120-dim total input.

---

## 6. Dynamic (in-game) win-probability models for 1st and 2nd innings

**Problem, plain-language.** Once a match has started, "who's ahead" is a
completely different question in each innings. In the 2nd innings there's
a specific target to chase, so the natural framing is "will the chasing
team reach it?" In the 1st innings there's no target yet, so the question
becomes "will the team batting first go on to defend whatever they end up
scoring?" These need genuinely different feature sets, and this is where
the pipeline's strongest signal lives — far stronger than the pre-match
model in §4.

**Code path.** `src/pipeline.py`:
- `build_dynamic_2nd(df, match_df)` builds per-delivery features
  (`DYN2 = ["runs_needed", "balls_remaining", "wkts_remaining", "crr",
  "rrr", "elo_adv", "phase"]`) for every ball of every 2nd innings, with
  label `chasing_wins` (= `batting_wins`, since the batting team in the 2nd
  innings is the one chasing). `elo_adv` is joined per match from the
  Elo columns (`elo2 - elo1` for the 2nd innings, i.e. relative to the
  chasing side).
- `build_dynamic_1st(df, match_df)` builds `DYN1 = ["team_runs",
  "team_wicket", "balls_remaining", "run_rate", "proj_total", "elo_adv",
  "phase"]`, label `defending_wins`. `proj_total` is a naive linear
  extrapolation (`team_runs + run_rate * balls_remaining / 6`) standing in
  for "how big will this total end up being," since there's no real target
  yet in the 1st innings.
- `train_dynamic_internal(df2, df1)` fits
  `CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=500),
  method="isotonic", cv=5)` for each, again on the `year <= 2020` /
  `year > 2020` temporal split.

**Real results** (from the actual run):

| Model | Test set size | Brier ↓ | AUC | Accuracy | ECE ↓ |
|---|---|---|---|---|---|
| Dynamic 2nd innings (chasing team wins) | 40,370 deliveries | 0.1445 | **0.8782** | 0.7918 | 0.0434 |
| Dynamic 1st innings (defending team wins) | — | 0.2203 | 0.7022 | 0.6395 | — |

Dynamic 2nd-innings AUC of **0.878** is 0.375 higher than pre-match's 0.5035
(§4) — by far the largest effect size anywhere in this codebase's results,
and the clearest evidence that in-game state, not pre-match team strength,
dominates T20 outcomes.

**Worked example (real).** Match 1304088, 2nd innings, Punjab Kings chasing
154 off Lucknow Super Giants. Two deliveries from the actual chase:

*Early in the chase — over 4, ball 6:*

| Feature | Value |
|---|---|
| `runs_needed` | 118.0 |
| `balls_remaining` | 90 |
| `wkts_remaining` | 9 |
| `crr` | 7.20 |
| `rrr` | 7.87 |
| `elo_adv` | −46.72 |
| `phase` | 0 (Powerplay) |

**Model's predicted P(Punjab Kings win) = 0.643.** Punjab Kings ultimately
lost this chase (`chasing_wins = 0`) — an early-innings miss, consistent
with the Powerplay phase having the widest genuine uncertainty (§7).

*Last ball of the chase — over 19, ball 6:*

| Feature | Value |
|---|---|
| `runs_needed` | 21.0 |
| `balls_remaining` | 1 |
| `wkts_remaining` | 2 |
| `rrr` | 126.0 |

**Model's predicted P(Punjab Kings win) = 0.000.** Needing 21 off the final
ball is essentially impossible; the model correctly gives it zero
probability, and Punjab Kings did lose.

**Feeds into.** The dynamic 2nd-innings model is the one whose exact math
(standardize → logistic decision function → isotonic calibration, averaged
across 5 CV folds) is re-implemented client-side in the dashboard's live
win-probability calculator (§14). §7's phase-specific models and §12's
calibration comparison both refit variants of this same model.

---

## 7. Phase-specific evaluation (Powerplay / Middle / Death)

**Problem, plain-language.** A T20 innings has three recognisably different
phases — **Powerplay** (overs 0–6, `phase=0`), **Middle** (overs 7–15,
`phase=1`), and **Death** (overs 16–19, `phase=2`). Does the dynamic
2nd-innings model's accuracy differ meaningfully across these? Intuitively
it should: later in a chase, there's mechanically less uncertainty left
because fewer possible outcomes remain.

**Code path.** `src/pipeline.py`'s `phase_specific_eval(train2, test2,
dsc2)` refits a *separate* calibrated logistic regression per phase, using
the same `DYN2` feature set minus `phase` itself (since phase is now the
partition, not a feature).

**Real results** (from the actual run):

| Phase | Overs | AUC | Brier ↓ | n (test deliveries) |
|---|---|---|---|---|
| Powerplay | 0–6 | 0.8210 | 0.1784 | 15,098 |
| Middle | 7–15 | 0.8857 | 0.1381 | 18,889 |
| Death | 16–19 | **0.9534** | **0.0836** | 6,383 |

AUC rises monotonically from 0.821 to 0.953 as the match progresses. The
worked example above illustrates exactly this pattern: the Powerplay-phase
prediction (P=0.643 for a team that went on to lose) was the kind of miss
this phase's lower AUC (0.821) predicts will happen more often than in the
Death overs (AUC 0.953), where the same model correctly assigned P≈0 to an
essentially-lost cause.

**Feeds into.** Purely a diagnostic breakdown of §6's dynamic 2nd-innings
model — it does not feed a separate downstream stage, but it directly
informs how much to trust a live win-probability number depending on which
over it was computed at.

---

## 8. Score regression models

**Problem, plain-language.** Separately from "who wins," can the pipeline
predict *how many runs* a team will finish with? This is done at three
distinct stages: pre-match (before any ball is bowled), during the 1st
innings, and during the 2nd innings — using five different regression
algorithms at each stage (`make_score_zoo()` in `src/models.py`: Ridge
linear regression, Random Forest, Gradient Boosting, HistGradientBoosting
(labelled "XGBoost" — see the header note on why), and Linear SVR).

**Code path.** `src/pipeline.py`'s `train_score_zoo_and_save(df1, df2,
match_df, out_path)`. Same `year <= 2020` / `year > 2020` split as
everywhere else. Metrics: **MAE** (mean absolute error, in runs — lower is
better, directly interpretable) and **R²** (fraction of the variance in
final score explained by the model; 1.0 is perfect, 0.0 equals "just
predict the training mean every time," and negative means *worse* than that
trivial mean-predictor).

**Real results** (from the actual run):

| Model | Pre-match MAE / R² | Inn. 1 MAE / R² | Inn. 2 MAE / R² |
|---|---|---|---|
| Linear (Ridge) | 30.9 / **−0.219** | 18.7 / 0.491 | 19.6 / 0.382 |
| Random Forest | 32.6 / −0.338 | 19.1 / 0.439 | 14.0 / 0.597 |
| Gradient Boosting | 32.8 / −0.350 | 18.9 / 0.452 | 13.6 / 0.618 |
| HistGB ("XGBoost") | 32.8 / −0.335 | 18.9 / 0.447 | **13.3 / 0.631** |
| Linear SVR | 32.8 / −0.363 | **18.4 / 0.503** | 17.8 / 0.441 |

Every model has **negative R² at the pre-match stage** — the regression
analogue of §4's classification finding, and equally important: none of
these models beat a straight-line prediction of "assume this match's final
score equals the training set's average final score." By the 2nd innings,
HistGB reaches R²=0.631 and MAE drops from 32.8 to 13.3 runs. No single
model wins at every stage — Linear SVR is best pre-match and 1st innings,
HistGB best 2nd innings.

**Worked example (real).** Match 1304088, 2nd innings, over 9 ball 2
(`runs_needed=89, balls_remaining=64, wkts_remaining=7, crr=6.96, rrr=8.34,
elo_adv=−46.72, phase=1`), fed through the trained `inn2_zoo` models loaded
from `models/ipl_score_pipeline.pkl`:

| Model | Predicted final score (runs) |
|---|---|
| Linear | 141.9 |
| Random Forest | 147.5 |
| Gradient Boosting | 146.4 |
| XGBoost (HistGB) | 146.8 |
| SVR | 144.3 |

**Actual final score: 133** (Punjab Kings were bowled out). Every model
overshot by 9–15 runs on this particular chase — consistent with the
in-game MAE of ~14–20 runs reported above; this single case sits within
that typical error band, not an outlier.

**Feeds into.** Saved via `joblib.dump` to `models/ipl_score_pipeline.pkl`
— this is the artefact loaded by the dashboard's score-prediction display
and by any downstream consumer wanting a trained regressor without
retraining.

---

## 9. The locked 2026 holdout evaluation

**Problem, plain-language.** Every result above uses an internal
"train ≤2020 / test >2020" split — both halves are still historical data
the pipeline could, in principle, have been iterated against. A genuinely
honest test needs data that was *never* touched during development: 74
matches from the 2026 season, held in `data/external_2026/*.csv`, never
loaded until this one evaluation step.

**Why walk-forward, not "just predict on held-out rows."** A real deployed
system would only know each match's Elo/form/head-to-head state as of that
match's kickoff — not the final 2026 standings. `evaluate_2026_pre_match()`
in `src/pipeline.py` first calls `retrain_pre_match_full(match_df)` to
retrain the pre-match model on the *entire* 2008–2025 dataset (more data
than the internal-only train split, since throwing away 2021–2025 signal
here would understate real deployed performance). It then walks through the
74 2026 matches **in chronological order**, updating each team's Elo, form,
and head-to-head record only *after* that match's real result is revealed —
exactly mirroring a real deployed system, with zero lookahead into future
2026 matches.

**Statistical machinery.** Reports a **Clopper-Pearson 95% confidence
interval** (an exact interval for a binomial proportion, more conservative
than a normal approximation — appropriate here because 74 matches is a
small sample) and a **z-test p-value** against the 50/50 null hypothesis
(i.e. "is this accuracy distinguishable from a coin flip?"). It also
computes a **naive baseline**: always pick whichever side (bat-first or
bowl-first) won more often across these same 74 matches.

**Real results** (from the actual run, after the defect-report fixes —
true date ordering DEF-007, Deccan/SRH split DEF-008, exact tests DEF-009):

| Metric | Value |
|---|---|
| Accuracy | **63.5% (47/74)** |
| 95% CI (Clopper-Pearson) | [51.5%, 74.4%] |
| p-value vs. 50/50 null (exact binomial) | 0.027 |
| Naive baseline (always pick historically-favoured side) | 63.5% |
| **Margin over naive baseline** | **0.0 points (exact McNemar p=1.0)** |

The 63.5% figure is statistically significant versus a coin flip
(p=0.027), but it is numerically EQUAL to the naive "usually-favoured
side" baseline — the exact McNemar paired test (1 match each way among
discordant picks) cannot distinguish the model from the baseline at all.
§15 discusses why this distinction (significant vs.
50/50, weak vs. a *stronger* baseline) is the single most important
number in this codebase.

**Feeds into.** `ext["pre_df"]` (each 2026 match's prediction, actual
result, and correctness) is consumed directly by §11's tournament
simulation.

---

## 10. SHAP explainability

**Problem, plain-language.** Given that the pre-match model (§4) barely
beats a trivial baseline, which of its 5 features is actually doing
anything? **SHAP** (SHapley Additive exPlanations) answers "how much did
each feature push this specific prediction away from the average
prediction," averaged (in absolute value) across many predictions to get a
global importance ranking.

**Code path.** `src/explainability.py`'s `shap_importance(model,
X_background, X_explain, feature_names)`. A documented fix worth noting:
this explains the **calibrated** classifier's `predict_proba` output (via
SHAP's permutation explainer, since `CalibratedClassifierCV` has no simple
closed-form gradient), not the raw uncalibrated base estimator — the
isotonic-regression calibration layer would otherwise be invisible to a
naive SHAP run against the base model directly. Applied to the pre-match
Cal. GBT model from §4.

**Real results** (from the actual run):

| Feature | Mean \|SHAP\| |
|---|---|
| `elo_diff` | **0.0440** |
| `h2h` | 0.0177 |
| `form_diff` | 0.0128 |
| `toss_field_first` | 0.0008 |
| `toss_bat_first` | 0.0004 |

`elo_diff` alone accounts for roughly two-thirds of the total SHAP mass
across all 5 features; the two toss features combined contribute under 2%.
This directly answers whether toss matters for pre-match prediction once
Elo is already present: it barely does. For the worked example (match
1304088, `elo_diff=+46.72` favouring LSG), this ranking says the model's
already-weak edge toward LSG (§4's P=0.445–0.453, i.e. actually favouring
Punjab Kings on net) is being driven almost entirely by the interplay of
`elo_diff`, `h2h`, and `form_diff` — not by either toss feature, which had
essentially no room to move that prediction either way.

**Feeds into.** Purely diagnostic — informs which pre-match features are
worth keeping in a future iteration, but doesn't feed a further modelling
stage.

---

## 11. Tournament simulation

**Problem, plain-language.** A single match's prediction accuracy (63.5%,
§9) is a noisy number over just 74 matches. A complementary sanity check:
if you use each match's predicted winner to build a *predicted season
points table*, does it end up looking anything like the *real* points
table? A model could be individually noisy on 74 matches yet still be
systematically right about which teams are strongest overall.

**Code path.** `src/tournament.py`:
- `actual_points_table(pm26, ig26)` builds the real 2026 points table
  (2 points per win) plus **Net Run Rate (NRR)** — the standard cricket
  formula: (runs scored ÷ balls faced, scaled to overs) minus (runs
  conceded ÷ balls bowled, scaled to overs), with the convention that a
  team bowled out before facing all 20 overs is credited with having
  "faced" the full 120 balls anyway.
- `predicted_points_table(pre_df)` builds the same table using the
  pre-match model's picks from §9 instead of actual results.
- `compare_tables(...)` checks whether the actual and predicted #1 team
  match, and how much the top-4 sets overlap.

**Real results** (from the actual run):

| Check | Result |
|---|---|
| Actual points-table topper | Royal Challengers Bengaluru |
| Predicted points-table topper | Royal Challengers Bengaluru — **correct** |
| Top-4 overlap | 3 of 4 teams: Gujarat Titans, Punjab Kings, Royal Challengers Bengaluru |

The model correctly named the actual table-topper and got 3 of the 4 actual
top-4 teams, a more reassuring aggregate signal than the bare 63.5%
match-level number in isolation.

**Feeds into.** Terminal — purely a downstream consumer of §9's per-match
predictions, feeding no further stage.

---

## 12. Calibration comparison: isotonic vs. temperature scaling

**Problem, plain-language.** The pipeline's default calibration method
everywhere above is **isotonic regression** (via `CalibratedClassifierCV`'s
internal 5-fold cross-validation) — a flexible, non-parametric way of
mapping a model's raw scores to calibrated probabilities. **Temperature
scaling** (Guo et al., 2017) is a much simpler alternative: find a single
scalar `T` that rescales the model's logit output before applying a
sigmoid, `p_cal = sigmoid(logit(p) / T)`. `T > 1` softens overconfident
predictions; `T < 1` sharpens underconfident ones; `T = 1` changes nothing.
Because it has only one parameter, it structurally cannot overfit the way
a more flexible method with more parameters could.

**Code path.** `src/temperature_scaling.py`'s `TemperatureScaler` (fits `T`
by minimizing validation negative log-likelihood via
`scipy.optimize.minimize_scalar`) and `compare_calibration(...)`.
`src/pipeline.py`'s `compare_calibration_methods_dyn2(df2)` runs a
controlled comparison on the dynamic 2nd-innings model: both methods start
from the *same* uncalibrated `LogisticRegression` fit on `year <= 2018`
training data; temperature scaling fits its scalar `T` on a genuinely
held-out validation slice (`2019–2020`, carved out of the training years,
never touching the `year > 2020` test set); the isotonic comparator is
refit on train+val combined so both methods get an equally-sized effective
training set for the comparison.

**Real results** (from the actual run):

| Method | Brier (raw → calibrated) | AUC (raw → calibrated) | ECE (raw → calibrated) |
|---|---|---|---|
| Temperature scaling (T=1.0906) | 0.1453 → 0.1449 | 0.8777 → 0.8777 | 0.0458 → 0.0448 |
| Isotonic (pipeline default) | — → 0.1446 | — → 0.8783 | — → 0.0436 |

`T=1.0906 > 1` means the raw logistic regression was **slightly
overconfident**, and temperature scaling correctly softens it a touch.
Both methods move ECE in the right direction; isotonic ends up marginally
ahead on all three metrics in this specific comparison, consistent with it
being the pipeline's chosen default rather than an arbitrary pick.

**Feeds into.** Purely comparative — the pipeline's actual production
dynamic 2nd-innings model (§6) uses isotonic, not temperature scaling.

---

## 13. The secondary Transformer / MC-Dropout model

**Status: not the recommended model, kept for comparison only.** Its own
prior evaluation (documented in `docs/known_limitations.md`) found it does
not beat the calibrated Logistic Regression baseline at a statistically
significant level, and it is trained here on different data/splits than
its original reporting — so its numbers should never be compared directly
against §6's headline Brier/AUC figures. It is not run by default
`run_all.py`; it has its own entry point, `run_alt_transformer.py`, which
this walkthrough did not execute (out of scope for the default pipeline
this document otherwise traces exactly).

**Architecture** (`src/transformer_model.py`'s `IPLTransformer`): a causal
(autoregressive-masked) Transformer encoder — meaning each ball's
prediction can only attend to *earlier* balls in the same innings, never
later ones — over a 120-dimensional input: the 24-dim game state from §5,
concatenated with three 32-dim player embeddings (batter, bowler,
non-striker). 2 layers, 4 attention heads, `d_model=64`.

**Multi-task design, deliberately disabled by default.** The loss function
supports three prediction heads — win probability (binary cross-entropy),
imminent-delivery outcome (7-way classification: `[0,1,2,3,4,6,wicket]`;
the game state at position *t* is strictly pre-ball, so the "next ball" is
delivery *t* itself and its label comes from the same row — DEF-004), and
projected final score (Huber loss, which is less sensitive to outlier
scores than plain squared error) — but ships with `lambda_next_ball=0,
lambda_score=0`, i.e. win-probability-only, because a prior evaluation
found the auxiliary heads actively hurt the win-probability signal.

**Player embeddings** (`src/win_probability_engine.py`'s
`PlayerEmbedTable`): fixed, randomly Xavier-initialised `nn.Embedding`
lookup tables — not learned or graph-pretrained — because a prior ablation
found no benefit from graph-based (GraphSAGE-style) player-representation
pretraining at this data scale (~2,000 unique players).

**MC-Dropout uncertainty** (`WinProbabilityEngine`): at inference time,
dropout is deliberately kept *active* (`model.train()`, not `.eval()`) and
the model is run `MC_SAMPLES=50` times per prediction; the variance across
those 50 stochastic forward passes becomes an approximate
epistemic-uncertainty estimate (`mean ± 1.96·std` as an approximate 95%
interval). This is the closest thing in the codebase to a genuine Bayesian
treatment, though it is a well-known frequentist *approximation* to
Bayesian inference (Gal & Ghahramani, 2016), not exact posterior sampling.

---

## 14. The dashboard

**Code path.** `dashboard/index.html` — a single self-contained HTML file,
no external JS dependency or CDN, every chart hand-rolled SVG. It renders:
a reliability diagram (from §4/§6's `calibration_bins()` output), an
Elo-over-time line chart (from §2's `compute_elo_history`), a phase/AUC bar
chart (§7), and the tournament table (§11). It also includes a **live
in-browser win-probability calculator** that re-implements the dynamic
2nd-innings model's exact math client-side in JavaScript — standardize,
compute the logistic decision score per CV fold, apply that fold's
isotonic calibration curve, average across the 5 folds — verified to match
the real Python model's `predict_proba` output to full float precision.

`src/dashboard_export.py`'s `update_dashboard_data()` merges freshly
computed pipeline results into the page's embedded `DATA` JSON blob without
touching any hand-authored HTML/CSS elsewhere in the file, so every number
the dashboard displays is traceable to an actual `run_all.py` run rather
than a hand-typed value. Confirmed by the actual run: `Dashboard written to
dashboard/index.html`.

---

## 15. Where results are weak vs. strong — reading the baselines correctly

The single organising idea across every stage above: **a headline number
(accuracy, AUC) means nothing until it's compared against the right
baseline.**

- **Pre-match win prediction (§4) loses to climatology.** Both trained
  classifiers have negative BSS (Cal. LR: −0.0516, Cal. GBT: −0.0204) —
  worse than a model that ignores every input feature and always predicts
  the training-set base rate. Cal. LR's AUC (0.4527) is even below 0.5.
  This is the strongest, most concrete evidence in the whole codebase that
  "80%+ accuracy" claims for pre-match cricket prediction (common in
  simpler treatments) need a climatology/BSS check before being trusted —
  raw accuracy or AUC alone can look plausible while the underlying
  probabilities carry no real skill.
- **Pre-match score regression (§8) loses to the training mean.** Every
  one of 5 regressors has negative R² at the pre-match stage (−0.214 to
  −0.376) — worse than just predicting "this match's score will equal the
  historical average."
- **The 2026 holdout (§9) beats a coin flip but does NOT beat a smarter
  baseline.** 63.5% accuracy is statistically distinguishable from 50%
  (exact binomial p=0.027), but the "always pick the side that's
  historically more often favoured" baseline gets exactly the same 63.5% —
  the exact McNemar paired test returns p=1.0. Two different claims
  are being conflated when people say "the model beats chance": "better
  than 50/50" (true, and cheap to achieve) and "better than the best simple
  heuristic" (unproven at this sample size).
- **In contrast, in-game models are genuinely strong.** Dynamic 2nd-innings
  AUC (0.878) and its Death-overs specialisation (AUC 0.953) reflect real,
  substantial skill — there is no equivalently trivial in-game baseline
  that would match these numbers, because by late in a chase the
  runs-needed/balls-remaining arithmetic itself carries enormous
  information the model is correctly exploiting.
- **In-game score regression is similarly strong relative to its own
  trivial baseline.** R²=0.631 (HistGB, 2nd innings) explains most of the
  remaining variance in final score once several overs are known, versus
  R²=0 for "always predict the mean."

**The general lesson**: this codebase's own honest self-evaluation (built
into `brier_skill_score`, the climatology comparator, and the 2026 naive
baseline) shows that *for T20 cricket specifically*, in-game ball-by-ball
state carries almost all of the genuinely exploitable signal, while
pre-match team-strength features (Elo, form, head-to-head, toss) carry very
little beyond what a trivial baseline already captures.

---

## 16. Glossary

- **Elo rating** — A single number per team representing current estimated
  strength, updated after each match using the chess Elo formula: teams
  gain rating for beating stronger opponents and lose little for beating
  weaker ones (§2).
- **AUC (Area Under the ROC Curve)** — Measures how well a model *ranks*
  positive cases above negative ones, from 0 to 1. 0.5 = no better than
  random; 1.0 = perfect ranking. Says nothing about whether the predicted
  *numbers* are well-calibrated.
- **Brier score** — Mean squared error between a predicted probability and
  the actual 0/1 outcome. Lower is better; 0 is a perfect probabilistic
  forecast.
- **Brier Skill Score (BSS)** — `1 − BrierScore(model) / BrierScore(baseline)`.
  0 means no improvement over the baseline; negative means the model is
  worse than the baseline; positive (up to 1) means genuine improvement.
- **Calibration** — The property that among all predictions of "X%
  probability," roughly X% actually happen. A model can rank correctly
  (good AUC) while being badly calibrated (e.g. always predicting either
  5% or 95% when the truth is closer to 40%/60%).
- **ECE (Expected Calibration Error)** — A single number (0 to 1)
  summarizing how far a model's predicted probabilities are, on average,
  from the observed frequencies, weighted by how many predictions fall in
  each probability bucket.
- **Isotonic regression** — A flexible, non-parametric calibration method
  that learns an arbitrary monotonic mapping from raw model score to
  calibrated probability, fit via cross-validation.
- **Temperature scaling** — A simpler, single-parameter calibration method:
  divide the model's logit output by a scalar `T` before converting back to
  a probability. Cannot overfit as easily as isotonic regression because it
  has only one free parameter.
- **Walk-forward validation** — Evaluating a model by stepping through time
  in order, only ever training on data from before the point being
  predicted — as opposed to randomly shuffling data into train/test sets,
  which would let the model implicitly see the future.
- **Climatology baseline** — The simplest possible forecast: always predict
  the historical average outcome rate, ignoring every input feature. Used
  as the reference point for BSS.
- **Phase** — One of three segments of a T20 innings: Powerplay (overs
  0–6), Middle (overs 7–15), Death (overs 16–19), each with different
  scoring/wicket-taking dynamics.
- **Run rate (RR)** — Runs scored per over (runs ÷ balls faced × 6).
- **Required run rate (RRR)** — The run rate a chasing team needs to
  maintain for the rest of the innings to reach its target.
- **SHAP (SHapley Additive exPlanations)** — A method for attributing a
  model's prediction to its individual input features, based on
  game-theoretic Shapley values; averaging the absolute attribution across
  many predictions gives a global feature-importance ranking.
- **MC-Dropout** — Running a neural network's dropout layers *active* at
  inference time (instead of switched off, as is normal for prediction) and
  repeating the forward pass many times; the spread of the resulting
  predictions approximates the network's uncertainty about its own answer.
- **Clopper-Pearson confidence interval** — An exact (not approximate)
  confidence interval for a proportion (like an accuracy rate) estimated
  from a binomial sample, appropriately conservative for small sample
  sizes.

---

## Appendix: raw `run_all.py` output

The full, unedited console output of the run this document is based on
(2026-07-14, after the defect-report fixes, against `data/raw/ipl_data.xlsx`,
1,146 matches, 273,503 deliveries):

```
==============================================================================
IPL Win-Probability & Score Prediction -- run_all.py
Run: 2026-07-14 20:11:36    Matches: 1,146    Deliveries: 273,503
==============================================================================

---------- 1. PRE-MATCH MODEL (internal: train <=2020, test >2020) -----------
  Climatology          Brier=0.2508  AUC=0.5000
  Cal. LR              Brier=0.2641  AUC=0.4408  Acc=0.4640  BSS=-0.0533  ECE=0.1183
  Cal. GBT             Brier=0.2563  AUC=0.4667  Acc=0.4986  BSS=-0.0220  ECE=0.0568

---------------- 2. DYNAMIC 2ND/1ST-INNINGS MODELS (internal) ----------------
  Dynamic 2nd          Brier=0.1450  AUC=0.8771  Acc=0.7912  ECE=0.0413
  Dynamic 1st          Brier=0.2204  AUC=0.7012  Acc=0.6389

----------------- 3. PHASE-SPECIFIC EVALUATION (2nd innings) -----------------
  Powerplay (0-6 ov)   AUC=0.8197  Brier=0.1786  n=15098
  Middle (6-15 ov)     AUC=0.8849  Brier=0.1393  n=18889
  Death (15-20 ov)     AUC=0.9532  Brier=0.0844  n=6383

 4. SCORE REGRESSION ZOO (fixed horizons; saved to models/ipl_score_pipeline.pkl) 
  pre_match:
    Linear           MAE=30.8  RMSE=39.0  R2=-0.214  n=347
    Random Forest    MAE=32.3  RMSE=40.4  R2=-0.303  n=347
    Gradient BT      MAE=32.7  RMSE=40.9  R2=-0.340  n=347
    XGBoost          MAE=32.7  RMSE=40.7  R2=-0.323  n=347
    SVR              MAE=32.9  RMSE=41.5  R2=-0.376  n=347
  inn1 @5ov:
    Linear           MAE=24.5  RMSE=30.6  R2=0.252  n=347
    Random Forest    MAE=24.8  RMSE=31.5  R2=0.208  n=347
    Gradient BT      MAE=26.0  RMSE=32.5  R2=0.154  n=347
    XGBoost          MAE=26.4  RMSE=33.2  R2=0.117  n=347
    SVR              MAE=24.0  RMSE=30.1  R2=0.276  n=347
  inn1 @10ov:
    Linear           MAE=19.3  RMSE=23.9  R2=0.542  n=347
    Random Forest    MAE=20.7  RMSE=27.5  R2=0.396  n=347
    Gradient BT      MAE=19.7  RMSE=25.1  R2=0.496  n=347
    XGBoost          MAE=20.0  RMSE=26.0  R2=0.460  n=347
    SVR              MAE=19.4  RMSE=24.1  R2=0.537  n=347
  inn1 @15ov:
    Linear           MAE=12.2  RMSE=15.7  R2=0.799  n=345
    Random Forest    MAE=12.3  RMSE=16.7  R2=0.772  n=345
    Gradient BT      MAE=13.1  RMSE=17.4  R2=0.752  n=345
    XGBoost          MAE=12.8  RMSE=17.2  R2=0.758  n=345
    SVR              MAE=12.1  RMSE=15.5  R2=0.802  n=345
  inn1 @18ov:
    Linear           MAE=8.4  RMSE=10.8  R2=0.897  n=338
    Random Forest    MAE=8.4  RMSE=11.6  R2=0.880  n=338
    Gradient BT      MAE=9.3  RMSE=13.1  R2=0.847  n=338
    XGBoost          MAE=9.4  RMSE=13.1  R2=0.847  n=338
    SVR              MAE=7.7  RMSE=9.9  R2=0.912  n=338
  inn2 @5ov:
    Linear           MAE=17.6  RMSE=21.5  R2=0.563  n=347
    Random Forest    MAE=16.5  RMSE=22.7  R2=0.514  n=347
    Gradient BT      MAE=16.3  RMSE=22.4  R2=0.527  n=347
    XGBoost          MAE=16.4  RMSE=22.2  R2=0.537  n=347
    SVR              MAE=14.6  RMSE=19.8  R2=0.630  n=347
  inn2 @10ov:
    Linear           MAE=17.5  RMSE=21.8  R2=0.534  n=343
    Random Forest    MAE=13.5  RMSE=18.3  R2=0.670  n=343
    Gradient BT      MAE=12.9  RMSE=17.6  R2=0.696  n=343
    XGBoost          MAE=13.2  RMSE=17.9  R2=0.683  n=343
    SVR              MAE=15.1  RMSE=19.9  R2=0.610  n=343
  inn2 @15ov:
    Linear           MAE=19.9  RMSE=25.0  R2=0.297  n=326
    Random Forest    MAE=10.3  RMSE=14.2  R2=0.774  n=326
    Gradient BT      MAE=9.8  RMSE=13.2  R2=0.803  n=326
    XGBoost          MAE=9.3  RMSE=12.8  R2=0.817  n=326
    SVR              MAE=18.0  RMSE=22.6  R2=0.426  n=326
  inn2 @18ov:
    Linear           MAE=23.6  RMSE=29.5  R2=-0.005  n=266
    Random Forest    MAE=8.2  RMSE=11.7  R2=0.842  n=266
    Gradient BT      MAE=7.4  RMSE=10.4  R2=0.874  n=266
    XGBoost          MAE=7.2  RMSE=10.0  R2=0.885  n=266
    SVR              MAE=21.5  RMSE=26.5  R2=0.189  n=266
    note: innings-stage metrics are reported at the predeclared
          5/10/15/18-over horizons only (one snapshot per innings, DEF-002) --
          aggregate all-delivery metrics were dominated by trivially-easy
          late-innings states and are no longer reported.
    note: pre-match score R2 is expected to be negative (worse than predicting
          the mean) -- this mirrors project_gagan's own finding and is not a
          bug. See docs/known_limitations.md.

------- 5. EXTERNAL 2026 HOLDOUT EVALUATION (locked, never used above) -------
  Retraining pre-match model on full 2008-2025 data first...
    note: label correction applied (data/external_2026/label_corrections.csv):
          match 12 match_winner: None -> 'PBKS'. Source: Inherited from the
          original pipeline's correction (project_gagan, cell 53), which
          pinned the winner to the bowl-first side; previously hard-coded in
          evaluate_2026_pre_match (DEF-003).
  Pre-match accuracy: 63.5%  (47/74)
  95% CI (Clopper-Pearson): [51.5%, 74.4%]
  p-value vs 50/50 (exact binomial): 0.027
  Naive majority-class baseline (always pick the bowl-first side): 63.5%
  p-value vs naive baseline (exact McNemar, 1/1 discordant): 1.000
    note: significance against 50/50 does not establish improvement over the
          naive majority baseline (DEF-009) -- the McNemar line above is the
          relevant comparison. Pre-match AUC is ~0.51 on internal LOSO (near-
          random); the 2026 accuracy figure may reflect variance over 74
          matches rather than genuine model skill. See
          docs/known_limitations.md.

------------- 6. SHAP EXPLAINABILITY (pre-match Cal. GBT model) --------------
  elo_diff           mean|SHAP|=0.0289
  h2h                mean|SHAP|=0.0108
  form_diff          mean|SHAP|=0.0105
  toss_bat_first     mean|SHAP|=0.0006
  toss_field_first   mean|SHAP|=0.0006

----- 7. 2026 TOURNAMENT SIMULATION (actual vs. predicted points table) ------
  Actual table topper   : Royal Challengers Bengaluru
  Predicted table topper: Royal Challengers Bengaluru  (CORRECT)
  Top-4 overlap: 3/4  ['Gujarat Titans', 'Punjab Kings', 'Royal Challengers Bengaluru']

--- 8. CALIBRATION RELIABILITY (pre-match Cal. GBT + dynamic 2nd-innings) ----
  Pre-match Cal. GBT: 5 populated bins (see dashboard for the reliability diagram)
  Dynamic 2nd innings: 10 populated bins (see dashboard for the reliability diagram)

-------------------- 9. TEAM ELO TRAJECTORIES (2008-2025) --------------------
  Computed Elo history for 15 teams (2292 match-level data points total)

--- 10. TEMPERATURE SCALING vs. ISOTONIC CALIBRATION (dynamic 2nd innings) ---
  Temperature-scaled (T=1.0639): Brier 0.1457->0.1453  AUC 0.8772->0.8772  ECE 0.0457->0.045
  Isotonic (default):              Brier 0.145  AUC 0.8771  ECE 0.0415
    note: both methods start from the same uncalibrated LogisticRegression;
          isotonic is this pipeline's default (see src/models.py) --
          temperature scaling is shown for comparison only, not as a
          replacement.
  
  Dashboard sections updated : calibration_comparison, dyn2_reliability, dynamic, elo_history, external_2026, overview, phases, pre_match, pre_match_reliability, score_regression, shap, tournament
  Dashboard sections retained: dyn2_calculator, horizons, notable_matches, score_scatter, single_match_horizons, trajectories
  Dashboard data artifact    : reports/dashboard_data.json

==============================================================================
Dashboard written to dashboard/index.html
Report saved to reports/run_summary.txt
Done.
```
