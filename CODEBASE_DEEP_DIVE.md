# IPL Win-Probability & Score Prediction — Codebase Deep Dive

> Written as standalone context for research-paper writing. Assumes no prior
> knowledge of how this codebase came to exist — it describes the system as
> it stands today: what it computes, why, and how each design choice maps
> onto (or departs from) the published literature in `IPL_ML_Papers_Reference.md`.

---

## 1. What this system does, in one paragraph

Given IPL ball-by-ball data (2008–2025) plus a locked-away 2026 holdout set,
the system predicts two things — **who wins a match** and **what score a
team will finish on** — at two different moments: **before a ball is
bowled** (pre-match) and **as the match unfolds, ball by ball** (dynamic/
in-game). Every prediction is a *calibrated probability*, not just a label,
and every evaluation reports calibration (Brier score, Expected Calibration
Error) alongside accuracy/AUC — a distinction the literature (Paper 31)
shows matters enormously in practice (a model picked by accuracy alone lost
money at −35% ROI in a betting simulation; the calibration-selected model
made +34.69%). The codebase treats that as a first-class design constraint,
not an afterthought.

---

## 2. Where this sits in the literature (mapping to `IPL_ML_Papers_Reference.md`)

The 51 papers split into two philosophies:

| | Papers 1–35 (standard ML) | Papers 36–51 (Bayesian/probabilistic) |
|---|---|---|
| Output | Win/loss label | Probability distribution |
| Team strength | Engineered average (a feature) | Latent parameter, often with a prior |
| Temporal dynamics | Static feature table, one row per match | Dynamic — updates as new information arrives |
| Evaluation | Accuracy | Calibration + accuracy |

**This codebase sits between the two camps, closer to the probabilistic
side than any of Papers 1–26, but without going fully Bayesian (no MCMC, no
posterior distributions, no PyMC/Stan).** Concretely:

- It **does** adopt the probabilistic framing from Step 1 of the papers'
  synthesized methodology (§"Methodology" in the reference doc): the target
  is `P(win | state at ball t)` for every `t`, not a single pre-match label.
  This is exactly the `Paper 49` framing (dynamic logistic regression,
  extended Bayesian in a follow-up), and Paper 8's argument that a single
  sequential model can implicitly learn how features interact differently
  across the match.
- It **does** treat calibration as a primary evaluation axis (Brier Skill
  Score, ECE, reliability diagrams) — the Paper 31 lesson, applied
  throughout `src/metrics.py` and `src/pipeline.py`.
- It **does** use a team-strength layer estimated from historical results
  (Elo, `src/elo.py`) rather than static per-match averages — the same
  spirit as the football Poisson lineage's "team ability" parameters
  (Papers 36–44), but via the chess Elo update rule rather than a
  hierarchical Poisson/Bayesian model.
- It **does not** implement a full Bayesian hierarchical model (no priors
  on team strength, no posterior sampling, no power-prior season-decay
  weighting as in Papers 40/41/43, no Gaussian Process smoothing of
  coefficients over the match as in Paper 49's key innovation). Every model
  here is a frequentist point-estimate classifier/regressor, calibrated
  post-hoc via isotonic regression or temperature scaling.
- It **does** use SHAP for interpretability (Paper 29's contribution to the
  NBA literature, ported here to IPL pre-match features) — one of the few
  IPL-specific papers in the list (1–26) to include any interpretability
  layer at all.
- It **does not** touch player-price/auction prediction (Papers 22–25) or
  fantasy team selection (Paper 26) — those are out of scope for this
  codebase entirely.

**This gap is itself a legitimate research angle**: the codebase already
proves out the *evaluation methodology* the Bayesian papers argue for
(calibration-first, walk-forward, no lookahead), on a genuinely large IPL
dataset (273,503 deliveries, 1,146 matches), but stops short of the full
hierarchical/Bayesian treatment those same papers describe. A paper could
credibly frame itself as: *"Here is a calibration-honest evaluation of
standard ML methods on IPL win-probability, establishing a baseline that
the (unimplemented) hierarchical Bayesian extension from Paper 49 would
need to beat — with the gap quantified precisely because the evaluation
methodology here is already rigorous."*

---

## 3. Data layer — `src/data.py`

Input is a single Excel workbook (`ipl_data.xlsx`), sheet `"Ball by Ball"`,
one row per legal/illegal delivery across 1,146 matches (2008–2025), plus a
separate locked 2026 holdout (`data/external_2026/*.csv`, 74 matches) that
is **never touched** until the final evaluation step — this is the
walk-forward discipline the Bayesian papers insist on (Paper 43's
methodology; Step 6 of the synthesized pipeline in the reference doc).

Two responsibilities:

1. **Team-name normalisation** (`NAME_MAP`) — IPL franchises have renamed
   over 18 seasons (`Delhi Daredevils → Delhi Capitals`,
   `Kings XI Punjab → Punjab Kings`, etc.). Every downstream
   computation that accumulates *per-team* history (Elo, form, H2H) would
   silently treat a renamed franchise as two different teams without this
   normalisation. DEF-008: only genuine renames of the same franchise are
   mapped — distinct franchises that merely shared a city (Deccan Chargers
   vs. Sunrisers Hyderabad) are deliberately kept separate, so each carries
   its own Elo/form/H2H history. The map is applied once, upstream of
   everything else
   (`load_ball_by_ball`), and every other module's docstring explicitly
   notes it depends on this having already happened.
2. **Match-table construction** (`build_match_table`) — collapses
   delivery-level rows into one row per match: `team1` (batting first),
   `team2` (bowling first), `winner`, `toss_winner`/`toss_decision`, and the
   two binary toss-advantage features `toss_bat_first` /
   `toss_field_first` (did the team that won the toss also get the
   outcome they chose?). This becomes the backbone `match_df` that every
   other feature-engineering step joins against.

---

## 4. Team-strength layer — `src/elo.py`

Standard chess Elo (K=32, initial rating 1500), applied match-by-match in
chronological order — **this is the codebase's closest analogue to the
Bayesian papers' "team ability" parameter** (Papers 36–44's attack/defence
strength; Paper 47's player-conditioned priors), except estimated via the
Elo point-update rule rather than a hierarchical posterior.

```
exp1 = 1 / (1 + 10^((elo2 - elo1) / 400))     # expected win prob for team1
elo1_new = elo1 + K * (actual_outcome - exp1)  # actual_outcome ∈ {0, 1}
elo2_new = elo2 + K * ((1-actual_outcome) - (1-exp1))
```

Two entry points, sharing one `_update()` helper so the update rule exists
in exactly one place:

- `compute_elo(match_df)` — returns each match's **pre-match** rating for
  both teams (`elo1`, `elo2`, `elo_diff`) plus the final end-of-history
  rating dict, used as the walk-forward starting point for 2026 evaluation.
- `compute_elo_history(match_df)` — returns each team's **full
  chronological trajectory** (rating after every match it played), for the
  dashboard's Elo-over-time visualisation. The update math is identical to
  `compute_elo`; only what gets recorded differs.

**No shrinkage/prior.** Every team starts at exactly 1500 regardless of how
much historical data exists elsewhere in the league — this is the one
place where the "new/sparse team" problem the Bayesian papers solve via
shrinkage-toward-league-mean (Paper 51's log5 model; Paper 47's stratified
priors) is *not* addressed. A new franchise's first Elo value is a fixed
constant, not a partially-pooled estimate. This is a concrete, well-scoped
extension point for a paper: replace the point-estimate Elo update with a
Bayesian rating system (e.g., TrueSkill or a Bayesian Elo variant) and
measure whether the shrinkage improves early-season predictions for young
franchises (GT, LSG — both post-2022 entrants in this dataset).

---

## 5. Match-level features — `src/features.py`

Five scalar helper functions (used both directly and as the pattern that
`pipeline.py`'s vectorised versions follow) plus two walk-forward
accumulators. **Every function in this module explicitly enforces
no-lookahead**: at the point a feature is computed for match `i`, it has
only seen the outcomes of matches `0..i-1`.

- `balls_remaining`, `runs_needed`, `crr` (current run rate), `rrr`
  (required run rate), `phase` (Powerplay 0–6 / Middle 7–15 / Death 16–19,
  encoded 0/1/2) — the standard in-game state primitives every dynamic
  model in `pipeline.py` is built from. `phase_vec` is the pandas-vectorised
  twin of `phase`, used across four different feature-construction sites
  (`pipeline.py`'s `build_dynamic_2nd`/`build_dynamic_1st`,
  `player_features.py`, `game_state.py`) so the phase-boundary rule exists
  in exactly one place.
- `compute_form_h2h(match_df, window=5)` — walk-forward **recent form**
  (rolling win-rate over each team's last 5 matches, defaulting to 0.5 with
  no history) and **raw head-to-head win rate** between the two specific
  teams (also defaulting to 0.5 with no shared history). Both are computed
  by iterating matches in order and only ever reading state accumulated
  from *strictly earlier* matches — the walk-forward discipline is
  structural, not just documented.
- `compute_h2h_beta(match_df, alpha=2, beta=2)` — **this is the one place
  in the codebase that actually does what the Bayesian papers describe**:
  a Beta(2,2) conjugate prior over head-to-head win rate, giving posterior
  mean `(wins + 2) / (n + 4)`. With zero prior meetings this is exactly
  0.5 (uninformative); with more meetings it shrinks smoothly toward the
  raw empirical rate. This is a direct, if small-scale, implementation of
  the shrinkage idea from Paper 51 (Bayesian log5 batter/pitcher matchups)
  and Paper 47 (Bayesian priors on player-vs-opponent history) — applied
  at the team level rather than the player level. **A paper could extend
  this exact mechanism to the player-vs-player matchup level** (e.g.
  Bumrah-vs-Kohli, using `player_features.py`'s existing
  `compute_matchup_features` as the raw-count input) since the raw counts
  already exist but currently feed nothing downstream (see §8).

---

## 6. The 24-dimensional per-ball game state — `src/game_state.py`

This is the densest feature-engineering module, and the one with a real,
documented, test-caught bug fix baked into its design — worth
understanding in detail because the fix generalises to a broader
methodological point.

**The bug (now fixed, and permanently regression-tested):** early
implementations paired a "before this ball" numerator (e.g. `score_before`)
with an "including this ball" denominator (`legal_balls_total`), which
silently **halved the reported run rate on ball 2 of every innings** (and
similarly distorted `boundary_rate`, `dot_rate`). The fix introduces
`legal_balls_before` (a strictly "before this ball" counter) and pairs it
consistently everywhere a ratio needs a pre-ball denominator. This is
exactly the kind of subtle off-by-one that a walk-forward, ball-level
system is prone to, and the reason `scripts/validate_game_state.py` and
`tests/test_game_state.py` exist as standalone cross-checks against the
real dataset (not synthetic data) — validating monotonicity, phase
boundaries, and the "before this ball" invariant on 5 real matches chosen
to exercise no-ball/wide edge cases.

**The 24 features** (indices, each pre-normalised into roughly [0,1] via a
fixed divisor — a deliberate simplification, not a learned scaler, since
this feature vector feeds the Transformer described in §9, which expects
bounded inputs):

```
0  over / 20                        12 partnership_balls / 60
1  legal_balls_total / 120          13 batter_sr_innings / 200
2  innings==2 flag                  14 batter_balls_innings / 120
3-5 phase one-hot                   15 bowler_econ_innings (clipped 20/ov)
6  score_before / 250               16 bowler_wkts_innings / 10
7  wickets_before / 10              17 boundary_rate so far
8  run_rate (clipped 15/ov)         18 dot_rate so far
9  runs in last completed over/30   19 runs_required / 200 (inn.2 only)
10 wickets in last 5 overs / 5      20 balls_remaining / 120 (inn.2 only)
11 partnership_runs / 150           21 required_rr (clipped 36/ov, inn.2)
                                    22 toss_won_bat
                                    23 innings==2 flag (duplicate of #2)
```

Everything is computed with vectorised pandas `groupby`/`cumsum`/`shift`
operations rather than a row-by-row loop — a deliberate performance choice
given 273,503 deliveries, and one that required re-deriving the
accumulation logic from scratch rather than porting a row-by-row
implementation that assumed different upstream columns.

**A methodological note relevant to a paper's related-work section:**
none of Papers 1–26 discuss this pre-ball/post-ball timing subtlety at all
— they describe features like "current run rate" and "wickets in hand"
without specifying whether the delivery being predicted is included in its
own run-rate denominator. This codebase's fix (and the test that catches
regressions) is a concrete, reportable methodological contribution: a
worked example of how silently-wrong feature timing degrades exactly the
early-innings predictions that matter most for in-game win-probability
graphics.

---

## 7. Player-level features — `src/player_features.py`

Two walk-forward accumulators, both season-boundary (not match-boundary)
in granularity — i.e. "season Y's features see only balls from seasons
`< Y`", coarser-grained than the ball-level no-lookahead elsewhere, and
explicitly documented as such:

- `compute_rolling_player_stats` — per-player batting (runs, balls, SR,
  boundary rate, phase-specific SR for Powerplay/Middle/Death) and bowling
  (economy, wickets, dot rate, phase-specific economy) stats, computed by
  accumulating one season at a time and snapshotting stats *before* adding
  that season's data.
- `compute_matchup_features` — same walk-forward pattern, but keyed by
  `(batter, bowler)` pair: balls faced, runs scored, dismissals, boundary
  rate, dot rate, strike rate **in that specific matchup**. This is the
  cricket analogue of Paper 51's baseball batter-pitcher log5 matchup
  model.

**This module is additive and currently unconsumed** — its outputs don't
feed into any trained model in `src/pipeline.py` (the README documents
this explicitly: "does not replace or feed into `src/pipeline.py`'s
team-level Elo/form/H2H features unless a specific model is built to use
both"). This is the single clearest gap between what the codebase computes
and what Papers 2 ("Batting Index"/"Bowling Index" composite features) and
51 (Bayesian matchup shrinkage) describe as valuable signal. **This is
probably the highest-leverage extension for a research paper**: the raw
ingredients for a player-level, matchup-aware win-probability model already
exist and are tested; they're simply not wired into a model yet.

---

## 8. The core prediction pipeline — `src/pipeline.py`

This is the largest module and the one that actually trains and evaluates
models. It implements five distinct prediction tasks, each evaluated with
a **temporal train/test split** (train ≤ 2020, test > 2020) rather than a
random split — directly following the Bayesian papers' insistence
(Paper 43) that sports data requires season-respecting validation, not
i.i.d. shuffling, because team strength evolves over time.

### 8.1 Pre-match model (`train_pre_match_internal`)

Features: `[elo_diff, form_diff, h2h, toss_bat_first, toss_field_first]`
(5 features — deliberately minimal, since this is the "before a ball is
bowled" setting where information is genuinely scarce). Two calibrated
classifiers — `CalibratedClassifierCV(LogisticRegression, isotonic, cv=5)`
and the same wrapper around `GradientBoostingClassifier` — are compared
against a **climatology baseline** (always predicting the training-set
base rate). This climatology comparison is what lets `brier_skill_score`
report something meaningful: **the honest finding here is that both
calibrated pre-match models have *negative* BSS** — they are measurably
worse than just predicting the league base rate. This single number is
the strongest piece of evidence in the whole codebase for the standard-ML
papers' blind spot: Papers 1–9 report 80–94% "accuracy" on pre-match
prediction without ever benchmarking against a trivial baseline or
checking calibration, and this codebase's own pre-match model — built with
the same feature families those papers use (Elo/form/H2H, toss) — fails
that more honest bar. This is a directly citable, quantified confirmation
of the calibration-first argument in Paper 31.

### 8.2 Dynamic in-game models (`build_dynamic_2nd`, `build_dynamic_1st`, `train_dynamic_internal`)

Two separate models, one per innings, because the batting team's task is
fundamentally different in each:

- **2nd innings** (`DYN2` features: `runs_needed, balls_remaining,
  wkts_remaining, crr, rrr, elo_adv, phase`) predicts `chasing_wins` —
  this is the classic "run chase" win-probability problem, and the model
  the dashboard's live calculator re-implements client-side (see §10).
- **1st innings** (`DYN1` features: `team_runs, team_wicket,
  balls_remaining, run_rate, proj_total, elo_adv, phase`) predicts
  `defending_wins` — there's no "target" yet in the 1st innings, so the
  feature set is necessarily different (a naive linear extrapolation
  `proj_total` stands in for "how big a target will this become").

Both use `CalibratedClassifierCV(LogisticRegression, isotonic, cv=5)`.
**Verified result: dynamic 2nd-innings AUC 0.878, dramatically better than
pre-match's ~0.51** — the standard, expected finding that in-game state
carries almost all the predictive signal in T20 cricket, consistent with
Paper 4's framing (ball-by-ball win probability, not pre-match).

### 8.3 Phase-specific evaluation (`phase_specific_eval`)

Refits a separate calibrated model per phase (Powerplay / Middle / Death)
using the *same* feature set minus `phase` itself (since phase is now the
partition variable, not a feature). Result: **AUC climbs monotonically
through the innings — 0.821 (Powerplay) → 0.886 (Middle) → 0.953
(Death)** — because uncertainty mechanically collapses as the chase
resolves. This is the closest thing in the codebase to Paper 49's
"Gaussian Process smoothing of coefficients across the match" idea, except
implemented as three discrete, independently-fit models rather than one
model whose coefficients vary continuously and smoothly with a GP prior.
**A paper could frame this exact three-bucket split as a coarse
discretisation of Paper 49's continuous GP approach**, and measure whether
a genuinely smooth coefficient trajectory (fit via a GP or spline prior
over over-number) improves on the discrete phase-boundary artefacts
visible at over 6 and over 15 in this implementation.

### 8.4 Score regression zoo (`train_score_zoo_and_save`)

Five regressors (Ridge, Random Forest, Gradient Boosting, HistGradient-
Boosting-as-"XGBoost", Linear SVR — the `make_score_zoo()` factory in
`src/models.py`) fit separately at three stages: pre-match, 1st innings
(features: `DYN1`), 2nd innings (features: `DYN2`). **Honest finding
mirrored from the pre-match classification result**: pre-match score
regression has *negative* R² for every single model (worse than predicting
the training mean), while in-game score regression reaches R² 0.44–0.63
once a few overs have been bowled. Both findings — pre-match win
prediction and pre-match score prediction being near-useless — are
internally consistent and mutually reinforcing evidence that T20 cricket's
outcome genuinely is dominated by in-match events, not pre-match team
strength differentials, at least as captured by Elo/form/H2H.

### 8.5 The locked 2026 holdout (`retrain_pre_match_full`, `evaluate_2026_pre_match`)

This is the walk-forward external validation the Bayesian literature
insists on (Paper 43's methodology; Step 6 of the reference doc's
pipeline). Critically, it is **not** just "predict on held-out rows" —
it's a genuine simulation:

1. Retrain the pre-match model on the *full* 2008–2025 dataset (more data
   than the internal train≤2020 split used), because using the
   internal-only model here would understate real performance by
   discarding 2021–2025 signal.
2. Walk through the 74 locked 2026 matches **in chronological order**,
   updating Elo, form, and head-to-head state after each match's result is
   revealed — exactly mirroring how a real deployed system would behave,
   with zero lookahead into future 2026 matches.
3. Report accuracy with a **Clopper-Pearson 95% confidence interval** (not
   just a point estimate — appropriately conservative for a 74-match
   sample) and a z-test p-value against the 50/50 null, plus a naive
   "always pick the historically-more-likely side" baseline for
   comparison.

**Verified result: 64.9% accuracy (48/74), 95% CI [52.9%, 75.6%], p=0.011
vs. 50/50 — but only 1.4 points above the 63.5% naive baseline.** The
statistically-significant-vs-50/50 result is real but weak evidence of
genuine skill once benchmarked against the *right* baseline (not "coin
flip" but "the team that's usually favoured wins"). This is the single
most important number in the codebase for a paper's discussion section:
it's a textbook example of why "statistically significant vs. a naive
null" and "practically useful over a strong baseline" are different
claims, and why a 74-match sample size can't distinguish between them.

### 8.6 Calibration-method comparison (`compare_calibration_methods_dyn2`)

Compares **isotonic regression** (the pipeline's default, via
`CalibratedClassifierCV`'s internal 5-fold CV) against **temperature
scaling** (`src/temperature_scaling.py`, ported as a standalone method) on
the same dynamic 2nd-innings model, using a genuinely held-out validation
slice (2019–2020, carved out of the training years) to fit temperature's
single scalar `T`, so neither method sees its own evaluation data during
fitting. This directly operationalises Paper 31's core argument
(calibration matters, and different calibration *methods* aren't
interchangeable) as a controlled, reproducible comparison rather than an
assertion.

---

## 9. Temperature scaling — `src/temperature_scaling.py`

A from-scratch implementation of Guo et al. (2017)'s single-parameter
calibration method: find scalar `T > 0` minimising validation
negative-log-likelihood of `sigmoid(logit(p) / T)`. `T > 1` softens
overconfident predictions; `T < 1` sharpens underconfident ones. Kept
deliberately separate from the isotonic `CalibratedClassifierCV` path in
`models.py` — the two solve genuinely different problems (isotonic is
non-parametric and needs k-fold CV; temperature scaling is a single scalar
fit on one held-out split and structurally cannot overfit as badly) and
the codebase treats them as two comparison points, not a single
"calibration module" abstraction.

---

## 10. The secondary model: causal Transformer + MC-Dropout — `src/transformer_model.py`, `src/alt_transformer_*.py`, `src/win_probability_engine.py`

A clearly-labeled **secondary, non-default** win-probability path,
included for comparison rather than as the recommended model (its own
prior evaluation found it does not beat the calibrated Logistic Regression
baseline at a statistically significant level — the codebase is explicit
about this rather than presenting it as an improvement).

**Architecture** (`IPLTransformer`): a causal (autoregressive-masked)
Transformer encoder over the 120-dimensional input
(24-dim game state ⊕ 3×32-dim player embeddings for batter/bowler/
non-striker), 2 layers, 4 heads, `d_model=64`. This is the closest thing
in the codebase to Paper 32's 1D-CNN+Transformer hybrid idea (here,
Transformer alone, no CNN front-end) and to Paper 8's "single sequential
model spans the whole match" argument versus per-over model ensembles.

**Multi-task design, deliberately disabled by default**: the loss function
supports three heads — win probability (BCE), imminent-delivery outcome
distribution (7-way cross-entropy: `[0,1,2,3,4,6,wicket]`; the game state
at position *t* is strictly pre-ball, so "the next ball" is delivery *t*
itself and the label comes from the same delivery row — DEF-004), and projected
final score (Huber loss) — but ships with `lambda_next_ball=0,
lambda_score=0` (win-only), because prior evaluation found the auxiliary
heads *actively hurt* the win-probability signal. This mirrors Paper 14's
point about robust loss functions (Huber over MSE for outlier scores) even
though here the auxiliary score head is switched off entirely rather than
just reweighted.

**Player embeddings**: `PlayerEmbedTable` is a fixed, randomly-initialised
(Xavier) `nn.Embedding`, not a learned or graph-pretrained representation
— a prior ablation found no significant benefit from graph-based
(GraphSAGE-style) pretraining at this data scale, so the simpler fixed
table is what ships. This is a direct, if negative, empirical data point
relevant to Paper 27's "graph-based methods for player interaction" as a
novel direction — worth citing as a counter-example at this specific data
scale (IPL, ~2000 unique players, 273K deliveries) even though it may not
generalise to larger datasets.

**MC-Dropout uncertainty** (`WinProbabilityEngine`): at inference, dropout
is deliberately kept **active** (`model.train()`, not `model.eval()`) and
the model is run `MC_SAMPLES=50` times per prediction; the variance across
those stochastic forward passes is used as an epistemic-uncertainty
estimate (`mean ± 1.96·std` as an approximate 95% CI). **This is the
single closest thing in the codebase to a genuine Bayesian treatment** —
it produces the *shape* of a posterior predictive interval (a mean and a
credible band) even though the underlying mechanism (dropout-as-approximate-
Bayesian-inference, Gal & Ghahramani 2016) is a well-known frequentist
approximation to a true Bayesian neural network, not exact posterior
inference via MCMC/variational methods the way Papers 36–44 do it for
football. **This is a second strong extension angle for a paper**:
compare MC-Dropout's uncertainty estimates against a properly Bayesian
alternative (e.g. a Bayesian logistic regression via PyMC/NumPyro on the
same dynamic 2nd-innings features, as the reference doc's synthesized
methodology recommends) and report whether the two produce meaningfully
different credible intervals, especially in early-innings, high-uncertainty
states.

---

## 11. Explainability — `src/explainability.py`

SHAP permutation-explainer values for the **calibrated** classifier's
`predict_proba` output (not the raw uncalibrated base estimator — a
specific, documented fix, since `CalibratedClassifierCV`'s isotonic layer
would otherwise be invisible to a naive SHAP explanation run on the base
model directly). Applied to the pre-match Cal. GBT model.

**Verified result**: `elo_diff` dominates (mean |SHAP| ≈ 0.044), `h2h` and
`form_diff` contribute modestly, and both toss features are essentially
irrelevant (≈0.0004–0.0007). This is a clean, quantified confirmation that
of the five pre-match features, Elo carries almost all of the (admittedly
weak) signal — directly relevant to any paper section debating whether
toss should even be included as a pre-match feature, a question several
of the standard-ML papers (Papers 2, 6, 20) raise without a SHAP-level
answer.

This is also the codebase's implementation of Paper 29's core contribution
(XGBoost + SHAP for NBA), applied to IPL's pre-match setting rather than
NBA's in-game setting.

---

## 12. Tournament simulation — `src/tournament.py`

Builds and compares **actual vs. model-predicted** 2026 points tables (Net
Run Rate included, computed the standard cricket way with the "all-out
before 20 overs counts as 120 balls faced" rule). Purely a downstream
consumer of the pre-match model's per-match predictions from §8.5 — no new
modeling, but a genuinely useful sanity check at the tournament level
rather than just per-match: **verified result — the model correctly
predicted the actual table topper (Royal Challengers Bengaluru) and got
3 of the actual top 4 teams right**, a more intuitively meaningful
aggregate check than a bare 64.9% match-level accuracy figure, since a
model that's individually noisy on 74 matches can still be systematically
right about which teams are strongest overall.

---

## 13. Evaluation-metric definitions — `src/metrics.py`

Three functions, deliberately built so the scalar and the diagnostic plot
data can never disagree:

- `calibration_bins(y_true, y_prob, n_bins=10)` — per-bin reliability-
  diagram data (predicted-mean vs. observed-frequency per bin, empty bins
  omitted). **Recently fixed bug, worth noting for methodological
  transparency**: `np.linspace(0,1,n_bins+1)` makes the final bin's upper
  edge exactly `1.0`; the original half-open interval convention
  (`lo <= p < hi`) meant predictions of *exactly* `p=1.0` (which isotonic
  calibration produces routinely at saturation) matched no bin at all and
  were silently dropped from both the bin counts and `ece()`'s weighting
  denominator. Fixed by closing the last bin on both ends. Verified impact
  on this dataset: 1,020 of 40,370 dynamic-2nd-innings test predictions
  were affected (recovered from being dropped), moving the reported dyn2
  ECE from 0.0430 to 0.0434 — small in this instance, but the *class* of
  bug (silently dropping boundary-saturated probabilities from a
  calibration metric) generalises to any pipeline using isotonic
  calibration, which is worth a methodological footnote in a paper that
  reports ECE.
- `ece(...)` — Expected Calibration Error, built strictly on top of
  `calibration_bins` (no independent recomputation).
- `brier_skill_score(y, p, p_clim)` — requires an *explicit* training-set
  climatology array rather than computing it from the test set in-place,
  specifically to avoid leaking test-set label distribution into the
  reference forecast (a subtle but real leakage vector documented inline).

---

## 14. Presentation layer — `dashboard/index.html`, `src/dashboard_export.py`

A single self-contained HTML file (no external JS dependency, no CDN —
every chart is hand-rolled SVG) that renders every result above as an
interactive reliability diagram, trajectory chart, Elo-over-time line
chart, phase/AUC bar chart, and tournament table, **plus a live
in-browser win-probability calculator** that re-implements the trained
dynamic 2nd-innings model's exact math (standardize → per-fold logistic
decision score → per-fold isotonic calibration → average across 5 folds)
in JavaScript, verified to match the real Python model's `predict_proba`
to full float precision. `dashboard_export.py` merges freshly-computed
pipeline results into the page's embedded `DATA` JSON blob without
touching hand-authored content elsewhere in the file, so every number on
the dashboard is traceable to an actual `run_all.py` run rather than a
hand-typed value.

---

## 15. Test suite — `tests/`

150 tests, organised so that: (a) every scalar formula (`phase`, `crr`,
`rrr`, Elo update, Beta-shrinkage math) has direct unit tests with
hand-computed expected values; (b) integrity tests check monotonicity/
boundedness invariants on synthetic data mirroring the real schema (no
dataset required, so CI can run without the 37MB Excel file); (c)
`test_game_state.py`/`scripts/validate_game_state.py` cross-check the
24-dim feature vector against the real dataset's own derived columns on 5
real matches chosen to exercise no-ball/wide edge cases; (d) "model
sanity" tests train tiny synthetic-data models and assert they've learned
obvious cricket logic (fewer wickets remaining → lower win probability,
etc.) — a check that catches "the code runs but learned nothing sensible"
bugs that pure shape/type tests would miss.

---

## 16. Results — the full numbers, consolidated

Every table below is pulled directly from a real `run_all.py` execution
against the actual dataset (1,146 matches, 273,503 deliveries, train ≤2020
/ test >2020 internal split unless noted). Numbers are reproduced exactly
as printed, not rounded further or estimated. Where §8–§13 discuss a
number in prose, this section is the canonical, complete version of it —
use these tables directly for a results/experiments section.

### 16.1 Pre-match win prediction (internal: train ≤2020, test >2020, n=347 test matches)

| Model | Brier ↓ | AUC | Accuracy | BSS | ECE ↓ |
|---|---|---|---|---|---|
| Climatology (base-rate baseline) | 0.2508 | 0.5000 | 0.4813 | 0.0000 | 0.0000 |
| Calibrated Logistic Regression | 0.2637 | 0.4527 | 0.4755 | **−0.0516** | 0.1092 |
| Calibrated Gradient Boosting | 0.2559 | 0.5035 | 0.4986 | **−0.0204** | 0.0654 |

**Reading this table**: both trained models have *negative* BSS — worse
than the trivial climatology baseline that just predicts the training base
rate every time. Cal. LR is also worse than a coin flip on AUC (0.4527 <
0.5). This is the pre-match model's headline result, and it should be the
first number quoted in any paper section discussing pre-match prediction.

### 16.2 Dynamic (in-game) win prediction

| Model | Test set | Brier ↓ | AUC | Accuracy | ECE ↓ |
|---|---|---|---|---|---|
| Dynamic 2nd innings (chasing team wins) | 40,370 deliveries | 0.1445 | **0.8782** | 0.7918 | 0.0434 |
| Dynamic 1st innings (defending team wins) | — | 0.2203 | 0.7022 | 0.6395 | — |

Dynamic 2nd-innings AUC (0.878) is **+0.375 over pre-match's 0.5035** —
the single largest effect size in the whole results set, and the clearest
empirical support for "in-game state dominates pre-match signal" in T20.

### 16.3 Phase-specific dynamic 2nd-innings models

| Phase | Overs | AUC | Brier ↓ | n (test deliveries) |
|---|---|---|---|---|
| Powerplay | 0–6 | 0.8210 | 0.1784 | 15,098 |
| Middle | 7–15 | 0.8857 | 0.1381 | 18,889 |
| Death | 16–19 | **0.9534** | **0.0836** | 6,383 |

Monotonic improvement through the innings — expected (uncertainty
mechanically shrinks as the outcome resolves), but the magnitude (AUC
0.821 → 0.953) is worth quoting precisely rather than just "improves."

### 16.4 Score regression — MAE (runs) and R² by stage and model

| Model | Pre-match MAE / R² | Inn. 1 MAE / R² | Inn. 2 MAE / R² |
|---|---|---|---|
| Linear (Ridge) | 30.9 / **−0.219** | 18.7 / 0.491 | 19.6 / 0.382 |
| Random Forest | 32.6 / −0.338 | 19.1 / 0.439 | 14.0 / 0.597 |
| Gradient Boosting | 32.8 / −0.350 | 18.9 / 0.452 | 13.6 / 0.618 |
| HistGB ("XGBoost") | 32.8 / −0.335 | 18.9 / 0.447 | **13.3 / 0.631** |
| Linear SVR | 32.8 / −0.361 | **18.4 / 0.504** | 17.8 / 0.440 |

**Reading this table**: every model has negative R² at the pre-match
stage (worse than predicting the training mean every time) — the
regression analogue of §16.1's classification result, and equally
important to report. By 2nd innings, HistGB reaches R²=0.631 (best),
roughly halving the pre-match MAE (32.8 → 13.3 runs). No single model
wins across all three stages — Linear SVR is best pre-match and 1st
innings, HistGB best 2nd innings — worth noting if a paper claims "model
X is best" without specifying the stage.

### 16.5 Locked 2026 external holdout (74 matches, walk-forward, never used in training/internal tuning)

| Metric | Value |
|---|---|
| Accuracy | 64.9% (48/74) |
| 95% CI (Clopper-Pearson) | [52.9%, 75.6%] |
| p-value vs. 50/50 null | 0.011 |
| Naive baseline (always pick historically-favoured side) | 63.5% |
| **Margin over naive baseline** | **+1.4 points** |

Statistically significant vs. a coin flip, but the 95% CI is wide enough
(a 22.7-point span) that it cannot distinguish "genuine skill" from "the
naive baseline plus noise" at this sample size — the CI overlaps a
scenario where the true accuracy equals the naive baseline exactly.

### 16.6 SHAP feature importance (pre-match Cal. GBT model)

| Feature | Mean \|SHAP\| |
|---|---|
| `elo_diff` | **0.0438** |
| `h2h` (raw head-to-head rate) | 0.0177 |
| `form_diff` (recent-form differential) | 0.0129 |
| `toss_field_first` | 0.0007 |
| `toss_bat_first` | 0.0004 |

Elo alone carries ~60% of total SHAP mass across the five features; toss
carries under 2% combined — a direct, quantified answer to whether toss
matters for pre-match prediction (it barely does, once Elo is present).

### 16.7 Tournament-level simulation (2026 season)

| Check | Result |
|---|---|
| Actual points-table topper | Royal Challengers Bengaluru |
| Predicted points-table topper | Royal Challengers Bengaluru — **correct** |
| Top-4 overlap (actual ∩ predicted) | 3 of 4 teams (GT, PBKS, RCB) |

### 16.8 Calibration method comparison — dynamic 2nd innings (isotonic vs. temperature scaling)

| Method | Brier (raw → calibrated) | AUC (raw → calibrated) | ECE (raw → calibrated) |
|---|---|---|---|
| Temperature scaling (T=1.0906) | 0.1453 → 0.1449 | 0.8777 → 0.8777 | 0.0458 → 0.0448 |
| Isotonic (pipeline default) | — → 0.1446 | — → 0.8783 | — → 0.0436 |

Both methods move ECE in the right direction; isotonic ends slightly
ahead on all three metrics in this comparison, consistent with it being
the pipeline's chosen default rather than an arbitrary choice.

### 16.9 An unconsumed result worth noting: `make_zoo()`

`src/models.py` defines a **5-classifier win-probability zoo**
(`make_zoo()`: Logistic, Random Forest, Gradient Boosting,
HistGradientBoosting-as-"XGBoost", LinearSVC), but grep confirms it is
**never called anywhere in the codebase** — not in `pipeline.py`, not in
`run_all.py`, not in any test. Only `make_score_zoo()` (the regression
equivalent, §16.4) is actually used. The pre-match/dynamic classification
tables above (§16.1, §16.2) use two hand-built classifiers
(`CalibratedClassifierCV(LogisticRegression)` and
`CalibratedClassifierCV(GradientBoostingClassifier)`) constructed directly
in `pipeline.py`, not via `make_zoo()`. **This means the full 5-model
classifier comparison (RF, HistGB, SVM alongside LR/GBT) that the
regression side already has has never actually been run and reported** —
a small, low-effort extension: wiring `make_zoo()` into
`train_pre_match_internal`/`train_dynamic_internal` would directly answer
"does a stronger classifier close the pre-match gap, or is the ceiling
feature-driven?" with real numbers instead of speculation.

---

## 17. Honest limitations (for a paper's limitations section)

1. **Pre-match win/score prediction is close to useless** as currently
   featured (Elo/form/H2H/toss only) — both classification (AUC ~0.51 LOSO)
   and regression (negative R²) confirm this independently. Any paper
   claim needs this caveat prominently, not buried.
2. **No hierarchical/Bayesian team-strength model.** Elo is a point
   estimate with no shrinkage for new/sparse teams; the Beta-smoothed H2H
   is the only genuinely Bayesian element in the whole codebase.
3. **Player-level features exist but are unconsumed** by any trained
   model — the highest-leverage, lowest-effort extension available.
4. **The secondary Transformer's numbers are not directly comparable**
   across any two runs unless trained on identical data splits — it's
   included for architecture comparison, not as a validated headline
   result.
5. **74-match 2026 holdout is small.** The Clopper-Pearson CI is wide
   (52.9–75.6%) specifically because of this; any claim of "beats the
   baseline" should quote the interval, not just the point estimate.
6. **No opponent-adjusted or venue-adjusted scoring model** (no Poisson/
   bivariate-Poisson analogue of the football literature) — score
   regression uses raw match-state features without a venue-strength
   latent parameter, unlike the football goal-scoring lineage (Papers
   36–44) which always includes venue/home-advantage as a modeled
   parameter, not just a raw feature.
7. **The 5-classifier win-probability zoo (`make_zoo()`) is dead code.**
   Only 2 of its 5 classifiers (Logistic Regression, Gradient Boosting)
   are ever actually trained, via hand-built calibrated estimators in
   `pipeline.py` — Random Forest, HistGB, and Linear SVM are defined but
   never run for classification (see §16.9), even though their regression
   counterparts (`make_score_zoo()`) are fully used. Any paper claiming
   "we compared five classifiers" against this codebase's numbers must
   verify that claim against §16.1/§16.2, not against what `models.py`
   merely defines.

---

## 18. Suggested framing for a research paper built on this codebase

Given the above, three defensible, distinct paper angles emerge from what
this codebase already proves versus what the literature (Papers 36–51)
argues should exist:

**A. "Calibration-first evaluation of standard ML on IPL win prediction"**
— Use §8.1/8.5/§13's results as the empirical core. Contribution:
demonstrate concretely (not just argue, as Paper 31 does for NBA betting)
that IPL pre-match models reported with only accuracy in Papers 1–9 fail a
climatology/BSS baseline, and that a naive "usually-favoured team wins"
baseline nearly matches a trained model's 2026 holdout accuracy. Directly
extends Paper 31's argument into the IPL-specific literature gap.

**B. "From point-estimate Elo to Bayesian team strength: does shrinkage
help IPL prediction?"** — Use §4/§8.1's Elo implementation as the
baseline to beat. Contribution: implement the missing shrinkage layer
(Bayesian Elo, or a hierarchical model per Papers 36–44's lineage) and
measure whether it improves predictions specifically for new/sparse
franchises (GT, LSG) where point-estimate Elo has no informative prior.

**C. "MC-Dropout vs. true Bayesian inference for in-game win probability"**
— Use §10's MC-Dropout engine as one arm, implement a genuinely Bayesian
dynamic logistic regression (PyMC/NumPyro, per the reference doc's Step 4)
on the same `DYN2` features as the other arm. Contribution: quantify how
much MC-Dropout's approximate uncertainty diverges from exact posterior
uncertainty on identical data — directly answers the question Paper 49
raises (Gaussian Process-smoothed Bayesian coefficients) without
committing to the full GP machinery.

Any of the three is scoped to use existing, tested, working code as the
control/baseline arm — meaning the paper's novel contribution is isolated
to one clearly-bounded extension rather than requiring a rebuild from
scratch.
