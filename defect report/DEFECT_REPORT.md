# Defect Report & Resolution: IPL Score & Win-Probability Prediction

**Review type:** Static, read-only source review (original report), followed by a verification-and-fix pass
**Scope:** Architecture, code quality, ML workflow, tests, deployment, and documentation
**Fix pass:** 2026-07-14. Every defect was independently re-verified against the source before being fixed. A later post-fix review recorded four residual findings (VF-001–VF-004, below); a second fix pass the same day verified all four (one was a false alarm about a file's location, three were genuine latent gaps), resolved them, and re-executed the checks: **all 172 tests pass and `run_all.py` was re-run end-to-end after the VF fixes, with identical reported metrics.**
**Refactor pass:** 2026-07-14, implementing the safe subset of `defect report/refactor suggestion.md` (items 1, 2-limited, 3, 4, 5, 7, 8). Its byte-diff verification exposed the pipeline's last two unseeded components — `LinearSVR` in the score zoo and SHAP's permutation order — whose metrics drifted between identical runs; both are now pinned to SEED=42. Post-refactor, two consecutive `run_all.py` executions differ **only in the timestamp line** (dashboard JSON byte-identical), and every deterministic value matches the pre-refactor output exactly.

## Executive summary

The original report was accurate: 10 of 12 defects were confirmed exactly as written, and verification made two of them *stronger* than claimed — DEF-007 turned out to be an active bug (59 of 1,169 matches were processed out of chronological order), and DEF-003's hidden override was masking a missing winner label in the source data. Two defects needed reframing: DEF-001's leakage mechanism is real but could not have inflated the reported test metrics (those were already computed on chronologically disjoint matches), and DEF-004's "wrong label" is actually a consistent pre-ball labeling convention that needed documentation, not a code change.

The fix pass introduced remediation for all 12 original defects. A subsequent
post-fix verification identified four remaining issues, recorded below in
**Post-fix verification findings** — each of those now carries its own
Assessment and Resolution and all four are closed (VF-001 was a
false alarm about the CI file's required location; VF-002/003/004 were genuine
latent gaps, none of which affected the reported numbers). The most
consequential outcome of the fix pass remains: with a correct timeline and
honest statistics, **the pre-match model's 2026 external accuracy (63.5%,
47/74) is exactly equal to the naive majority baseline (exact McNemar
p = 1.0)** — the previously reported +1.4-point margin over the baseline was
an artifact of the defects.

## Defects

### DEF-001: Delivery-level calibration splits can leak match-level information

- **Severity:** High (original) → **Medium (assessed)**
- **File:** `src/pipeline.py`
- **Function/class:** `train_dynamic_internal`, `phase_specific_eval`
- **Description:** `CalibratedClassifierCV(cv=5)` is fitted on individual delivery rows. Deliveries from one match can therefore be in both the base-model/calibration training partition and the held-out calibration partition.
- **Assessment — valid mechanism, overstated impact.** Confirmed: sklearn's default stratified folds split delivery rows randomly, so one match's deliveries did sit on both sides of every calibration fold. However, the claim that this made "the resulting dynamic metrics appear stronger than they would be for genuinely unseen matches" does not hold: all reported dynamic and phase metrics were computed on a test set that is chronologically disjoint at the match level (train ≤ 2020, test > 2020), so no test match was ever seen in training. The real harm was confined to the *quality of the isotonic calibration mapping* (fitted on optimistically-scored rows), not to test-set integrity. Downgraded to Medium.
- **Resolution.** Added `match_grouped_cv()` in `src/pipeline.py` (GroupKFold keyed by `match_id`) and used it for every delivery-level `CalibratedClassifierCV`: `train_dynamic_internal` (both innings models), `phase_specific_eval`, and the isotonic comparator in `compare_calibration_methods_dyn2`. Pre-match models keep `cv=5` — they operate on one-row-per-match data, so folds cannot split a match. Regression test: `tests/test_defect_fixes.py::TestMatchGroupedCV` proves no match ever appears on both sides of a fold. Metric impact after refit: negligible (dyn2 Brier 0.1445 → 0.1450, AUC 0.8782 → 0.8771), which is consistent with the assessment that test metrics were never inflated.

### DEF-002: Score-regression metrics are not evaluated at stated horizons

- **Severity:** High — **confirmed**
- **File:** `src/pipeline.py`
- **Function/class:** `train_score_zoo_and_save`
- **Description:** The score-regression model is trained and evaluated on every delivery in an innings, including final or near-final balls where current score nearly determines final score. The `make_score_zoo` documentation states evaluation at 5, 10, 15, and 18-over horizons, but this filtering is absent.
- **Assessment — valid, and slightly understated.** Confirmed, with an aggravating detail: on the final ball of an innings the feature `team_runs` *equals* the regression target (`final_score` is `max(team_runs)`), so the old aggregate metrics included rows where the answer was literally in the input. The documented 5/10/15/18-over protocol was entirely absent from the code.
- **Resolution.** Added `SCORE_HORIZONS = (5, 10, 15, 18)` and `horizon_snapshot()` to `src/pipeline.py`: one snapshot row per innings at exactly h completed overs (`team_balls == h*6`, last such row); innings that ended earlier are excluded rather than scored trivially. Zoos are still *trained* on all deliveries but *evaluated* only at the horizons, reporting MAE, RMSE, R², and n per horizon (`_regression_row`); the aggregate all-delivery numbers are gone. `run_all.py` and the dashboard print the per-horizon tables; the saved bundle now records the horizons and feature schema. Regression test: `TestHorizonSnapshot`. **Impact — this changes the headline story:** the old aggregate said inn1 MAE ≈ 18.4; the honest per-horizon numbers are MAE ≈ 24–26 at 5 overs, ≈ 19–21 at 10, ≈ 12 at 15, ≈ 8 at 18. Early-innings projection was genuinely being flattered by late-innings rows, exactly as the report predicted.

### DEF-003: External evaluation contains an undocumented hard-coded result override

- **Severity:** High — **confirmed**
- **File:** `src/pipeline.py`
- **Function/class:** `evaluate_2026_pre_match`
- **Description:** The function overrides `actual_winner` for `match_id == 12` directly in code.
- **Assessment — valid; root cause identified.** Confirmed, and the verification found *why* the override exists: the source row for match 12 (KKR vs PBKS, Eden Gardens, 06 Apr 2026) has an **empty `match_winner`**, and the repo's in-game data for that match is truncated at 3.4 overs of innings 1, so the result cannot be re-derived from repository data. The override silently pinned the winner with no provenance, no logging, and — as the report warned — would have silently corrupted a different match if the CSV were ever re-numbered.
- **Resolution.** The hard-coded line is gone. Corrections now live in a versioned table, `data/external_2026/label_corrections.csv` (`match_id, column, value, reason, source`), applied by the new `apply_label_corrections()` which (a) refuses to run if a correction references a match absent from the data (stale-table protection), and (b) returns every applied change so `run_all.py` logs it in the run report with full provenance. `actual_winner` is now derived from the corrected `match_winner` column and validated: any remaining null winner raises with an instruction to add a documented correction, and every row is cross-checked against `bat_first_won` (a mismatch raises). Regression tests: `TestApplyLabelCorrections`. The correction record for match 12 documents that the value is inherited from the original project's fix and cannot be independently re-derived from in-repo data — that honest limitation is now visible instead of buried in an expression.

### DEF-004: Transformer auxiliary label is current-ball, not next-ball

- **Severity:** High (original) → **Low, documentation defect (assessed)**
- **File:** `src/alt_transformer_data.py`
- **Function/class:** `_encode_next_ball`, `build_innings_sequences`
- **Description:** The label is calculated from the same delivery row as the input state, while model documentation describes the head as predicting the next-ball outcome.
- **Assessment — factually accurate, but the labeling is correct; the documentation was wrong.** The label does come from the same delivery row — but the game-state features at position t are strictly *pre-ball* (`score_before`, `wickets_before`, `legal_balls_before`, …), so from the model's point of view delivery t **is** the next ball to be bowled. Shifting labels to t+1, as suggested, would actually have introduced an off-by-one error under this convention. The one genuine data problem — feature index 1 leaking the current delivery's legality into the "pre-ball" state — is DEF-005's finding, and fixing it makes this labeling clean. Two further mitigations: `LAMBDA_NEXT_BALL = 0.0` (the head is disabled in the default, recommended configuration) and the docs said so. High severity was not warranted for an inactive auxiliary head.
- **Resolution.** Documentation fixed to match the (correct) code rather than the other way around: `src/transformer_model.py`'s module docstring now defines `next_ball[t]` as "the outcome of delivery t itself — the imminent ball given the strictly pre-ball state at t," `_encode_next_ball` in `src/alt_transformer_data.py` carries a docstring explaining why no one-ball shift is wanted, and `docs/codebase_walkthrough.md` / `CODEBASE_DEEP_DIVE.md` were updated to say "imminent-delivery outcome." Combined with the DEF-005 fix below, the pre-ball state now contains no information about the delivery being labeled.

### DEF-005: Pre-ball game state includes current delivery legality

- **Severity:** Medium — **confirmed**
- **File:** `src/game_state.py`
- **Function/class:** `build_game_state_matrix`
- **Description:** `legal_balls_total` includes the current delivery's legal/wide/no-ball status, while score, wickets, and other state features are calculated before the delivery.
- **Assessment — valid.** Confirmed: feature index 1 used `legal_balls_total`, a counter that includes the current ball's legality, while every other state feature followed the "before this ball" convention. The code comments show this was a deliberate choice (to match the source data's `team_balls` semantics for validation), but deliberate or not, a live pre-ball model must not know whether the imminent delivery will be a wide — and since a wide forces `runs_batter = 0`, this also leaked label information into DEF-004's auxiliary task.
- **Resolution.** Feature index 1 now uses `legal_balls_before / 120` (strictly pre-ball). `legal_balls_total` is still computed, but only as a validation column — the existing test that cross-checks it 1:1 against the source's `team_balls` (`tests/test_game_state.py`) still passes, so the useful convention check was kept without polluting the feature vector. The feature-layout docstring documents the change. Regression test: `TestWicketWindowUsesLegalBalls::test_feature_index_1_is_pre_ball`.

### DEF-006: "Last five overs" wicket feature uses delivery rows, not legal balls

- **Severity:** Medium — **confirmed**
- **File:** `src/game_state.py`
- **Function/class:** `build_game_state_matrix`
- **Description:** `wk_last_5_overs` uses a 30-row rolling window, although the comment defines it as 30 legal balls.
- **Assessment — valid.** Confirmed: `.rolling(window=30)` counted 30 *rows*, so any wide or no-ball inside the span silently shrank the window below five overs of actual play.
- **Resolution.** The window is now computed over the last 30 **legal** balls: within each innings, `searchsorted` on the non-decreasing `legal_balls_before` finds the span start, and the wicket count is the difference of cumulative pre-ball wicket counters (wickets that fell on illegal deliveries inside the span — e.g. run-outs off wides — are correctly included; the current ball is correctly excluded). Regression test: `TestWicketWindowUsesLegalBalls` constructs an innings where a wicket sits 31 rows but only 26 legal balls in the past — the old code dropped it, the fix retains it, and it correctly expires after the 30th legal ball.

### DEF-007: Historical feature ordering relies on match ID instead of verified date

- **Severity:** Medium (original) → **High (assessed — this was an active bug)**
- **Files:** `src/data.py`, `src/pipeline.py`
- **Functions/classes:** `build_match_table`, `load_and_prepare`
- **Description:** Match history is sorted by `match_id`, and the match table does not retain a date. Delivery aggregation also relies on input row order for `first` and `last` values.
- **Assessment — valid, and worse than reported.** The report framed this as a fragile assumption; verification showed the assumption was already broken: the match IDs are Cricsheet-style (e.g. 1082591) and **59 of 1,169 matches were out of date order** under the `match_id` sort. Every walk-forward computation — Elo, form, head-to-head — was running on a partially scrambled timeline. The source data has a clean, fully populated ISO `date` column that was simply never used. Upgraded to High.
- **Resolution.** `build_match_table` now (a) sorts deliveries deterministically by `(match_id, innings, over, ball)` before any `first`/`last` aggregation, (b) retains a parsed `date` column, and (c) validates it — more than one date within a match, or an unparseable/missing date, raises immediately. `load_and_prepare` sorts the match table by `(date, match_id)` before Elo/form/H2H. `src/elo.py`'s docstrings now state the date-order contract explicitly. Regression tests: `TestBuildMatchTable` (correct `score1` even from fully scrambled row order; date retained and parsed; conflicting dates raise). This fix (with DEF-008) is why the reported numbers moved — see "Impact on reported results" below.

### DEF-008: Franchise normalisation conflates distinct teams

- **Severity:** Medium — **confirmed**
- **File:** `src/data.py`
- **Function/class:** `NAME_MAP`
- **Description:** Deccan Chargers are normalised to Sunrisers Hyderabad.
- **Assessment — valid.** Confirmed. Deccan Chargers folded after 2012; Sunrisers Hyderabad entered 2013 as a new franchise that merely took over the Hyderabad slot. Mapping one onto the other handed SRH five seasons of another franchise's Elo, form, and head-to-head history. The merge *was* disclosed in the walkthrough docs (tagged DEF-L01), so this was a documented-but-unjustified policy rather than a hidden bug — the report's "define a justified policy" ask was the right one, and the justifiable policy is separation.
- **Resolution.** Removed the `Deccan Chargers → Sunrisers Hyderabad` entry (and the no-op `Pune Warriors` self-mapping) from `NAME_MAP`; the map now covers same-franchise renames only, and says so. SRH now starts at the initial Elo rating in 2013, which is historically correct; the Elo history now tracks 15 teams instead of 14. Updated: `src/elo.py` docstring, `docs/codebase_walkthrough.md`, `CODEBASE_DEEP_DIVE.md`, and `tests/test_features.py::test_name_map_deccan_chargers` (which had pinned the old behavior and now pins the new policy).

### DEF-009: Reported external p-value uses an unsuitable baseline

- **Severity:** Medium — **confirmed**
- **File:** `src/pipeline.py`
- **Function/class:** `evaluate_2026_pre_match`
- **Description:** The p-value is a normal-approximation test against 50%, but the report also shows a 63.5% naive majority baseline.
- **Assessment — valid, and the fix proved the point.** Confirmed on both counts: the test was a normal approximation (not exact), and 50% was not the relevant null. The suggested comparison against the naive baseline turned out to be decisive — see Resolution.
- **Resolution.** The p-value vs. chance is now an exact binomial test (`scipy.stats.binomtest`), and the model is additionally compared against the naive majority-side predictor with an **exact McNemar paired test** on the same 74 matches (paired is the right design: both predictors are evaluated match-by-match). `run_all.py` reports both, plus the discordant-pair counts, with a note that only the McNemar line is the relevant claim. **Post-fix result: model 63.5% vs naive 63.5%, discordant pairs 1–1, McNemar p = 1.000** — the model is statistically and numerically indistinguishable from "always pick the historically favoured side," while still being "significant vs 50/50" (exact p = 0.027). This is precisely the misleading-readers scenario the defect described, now made impossible to miss. README and walkthrough claims of a "+1.4-point margin over the baseline" were corrected.

### DEF-010: Multiple components are unconsumed or not deployable

- **Severity:** Medium — **confirmed**
- **Files:** `src/models.py`, `src/player_features.py`, `src/win_probability_engine.py`, `run_alt_transformer.py`
- **Functions/classes:** `make_zoo`, `compute_rolling_player_stats`, `compute_matchup_features`, `WinProbabilityEngine`
- **Description:** The five-classifier zoo and player-level features are not used by the default pipeline. The alternative Transformer training entry point does not persist a model checkpoint, while `WinProbabilityEngine` requires a trained model to be provided.
- **Assessment — valid on every item.** Confirmed by grep: `make_zoo` had zero importers anywhere; `player_features` and `WinProbabilityEngine` were exercised only by their own tests; `run_alt_transformer.py` trained a model and let it die with the process, making the "live engine" unusable in practice.
- **Resolution.** Per the report's "integrate or retire" framing, each item got the honest treatment: (1) `make_zoo` **retired** — deleted from `src/models.py` with its now-unused imports (the classifiers actually used are built in `src/pipeline.py`). (2) Transformer checkpoint **persisted** — `run_alt_transformer.py` now saves `models/alt_transformer.pt` containing the selected weights *plus* the metadata needed to rebuild identical inputs (player registry, embedding seed, year splits, epochs, val/test metrics, feature-layout description). (3) Engine made **loadable** — new `WinProbabilityEngine.from_checkpoint(path)` reconstructs a working MC-Dropout engine from that file; round-trip covered by `TestEngineFromCheckpoint`. (4) `player_features.py` **kept, explicitly statused** — its docstring now states it is a tested reference implementation not consumed by any default path, tied to improvement priority 6, with an instruction to archive it if still unconsumed when that work lands. Full integration would be a modeling project, not a defect fix, and pretending otherwise in code would be worse than saying so.

### DEF-011: Dashboard update is brittle and can retain stale data

- **Severity:** Low — **confirmed**
- **File:** `src/dashboard_export.py`
- **Function/class:** `update_dashboard_data`
- **Description:** The exporter identifies an embedded JSON object using exact string markers and performs a shallow merge.
- **Assessment — valid on both halves.** Confirmed: the end-of-blob search for `";\n"` breaks if any JS follows the object on the same line (it would swallow that code into the JSON slice), and `run_all.py` was only pushing 4 of the 12 computed sections, so the dashboard's headline numbers (pre-match, dynamic, score regression, external 2026) silently went stale relative to the run report.
- **Resolution.** Three changes: (1) the end of the embedded object is now found by *parsing it* (`json.JSONDecoder.raw_decode`) instead of pattern-matching a sentinel, with a numpy-safe encoder for the write-back; (2) `run_all.py` now pushes **every section it computes** (overview, pre-match, dynamic, phases, score regression, external 2026, SHAP, tournament, plus the four it already pushed) and logs which sections were updated vs. retained, so "retained" is a visible, deliberate list of hand-curated visualisations rather than silent staleness; (3) the merged object is additionally written to a standalone, diffable artifact, `reports/dashboard_data.json`. The report's suggestion to have the dashboard *fetch* a separate JSON file was considered and rejected: the dashboard is deliberately a single self-contained HTML file that works over `file://`, where fetch is blocked by CORS. Regression tests: `TestDashboardExportRobustness` (including the same-line-JS case that broke the old parser). The dashboard's score-regression table renders the new per-horizon stages without JS changes (its renderer iterates stage keys generically).

### DEF-012: Reproducibility and deployment controls are incomplete

- **Severity:** Low — **confirmed**
- **Files:** repository root, `requirements.txt`, entry points
- **Description:** No packaging manifest, CI workflow, environment-variable configuration, structured experiment registry, data-schema contract, or complete dependency lock was found. `torch>=2.1` permits dependency drift.
- **Assessment — valid.** Confirmed; verification also found `pytest-cov>=5.0` listed but not installed and not used by any configuration — a phantom dependency.
- **Resolution.** Added `pyproject.toml` (packaging manifest, pinned dependencies, pytest configuration); pinned `torch==2.8.0` (the verified working version) and removed the phantom `pytest-cov`; added GitHub Actions CI (`.github/workflows/ci.yml` at the repo root) that installs pinned requirements and runs the suite on every push/PR — tests requiring the uncommitted 36 MB raw dataset already self-skip, so CI runs the synthetic/unit suite. Partial-scope items, stated honestly: a full experiment registry and formal data-schema contract were **not** built (they are infrastructure projects beyond a fix pass), but the two concrete schema risks the report identified are now enforced in code — date validation in `build_match_table` (DEF-007) and winner-label validation with a corrections table in `evaluate_2026_pre_match` (DEF-003) — and deterministic run inputs are pinned end-to-end. **Addendum (refactor pass):** byte-diff verification of the refactors exposed two remaining unseeded components — `LinearSVR` (no `random_state`, unlike every other zoo member) and SHAP's permutation explainer (seeded background sample, unseeded permutation order) — whose values drifted slightly between identical runs. Both now use SEED=42, making `run_all.py` byte-reproducible: two consecutive runs differ only in the report's timestamp line.

## Impact on reported results

The fixes deliberately changed reported numbers (chronology repaired, franchises separated, honest evaluation protocol). Re-run of `run_all.py` after the fixes:

| Metric | Before | After | Driver |
|---|---|---|---|
| 2026 external accuracy | 64.9% (48/74) | 63.5% (47/74) | DEF-007/008 (corrected features) |
| p-value vs 50/50 | 0.011 (normal approx.) | 0.027 (exact binomial) | DEF-009 |
| Margin over naive baseline | +1.4 pts (claimed) | **0.0 pts, McNemar p = 1.0** | DEF-007/008/009 |
| Dynamic 2nd Brier / AUC | 0.1445 / 0.8782 | 0.1450 / 0.8771 | DEF-001/007/008 (negligible) |
| inn1 score MAE | 18.4–19.1 (aggregate, misleading) | 24–26 @5ov → ~8 @18ov | DEF-002 (honest horizons) |
| Elo team histories | 14 teams | 15 teams (Deccan separate) | DEF-008 |

## Post-fix verification findings

The following issues were identified by a subsequent read-only review of the
updated report and codebase. A second fix pass (same day) verified each one,
resolved it, and re-ran the full suite (172 tests pass) and `run_all.py`
(reported metrics unchanged — confirming VF-002 and VF-003 were latent holes,
not active bugs, in this dataset).

### VF-001: Reported CI workflow is absent

- **Severity:** High
- **Files:** `defect report/DEFECT_REPORT.md`; repository root
- **Function/class:** N/A
- **Description:** DEF-012's resolution says that `.github/workflows/ci.yml`
  was added and runs the test suite on pushes and pull requests. No `.github`
  workflow directory or CI workflow is present in the current repository.
- **Why it matters:** Continuous verification is not available, despite being
  presented as implemented. Regressions can therefore be merged without the
  automated checks described in the report.
- **Suggested improvement:** Add the documented workflow, then revise the
  report only after confirming it is present and configured to install the
  pinned environment and execute the intended test suite.
- **Assessment — false alarm on absence; real discoverability gap.** The
  workflow exists, but one level above where the review looked: at
  `Merged/.github/workflows/ci.yml`, the **git repository root** (confirmed
  via `git rev-parse --show-toplevel`). The project lives in the
  `merged_project/` subdirectory of that repo, and GitHub Actions only reads
  workflows from the repo root — placing the file inside `merged_project/`
  would make it a no-op. The reviewer's underlying point stands, though:
  nothing inside the project directory told an auditor where CI lives.
- **Resolution.** Workflow kept at the repo root (the only functional
  location); its config already uses `defaults.run.working-directory:
  merged_project` to build and test this project. Discoverability fixed:
  `README.md` now states the workflow's path, why it must live at the git
  root, and that data-dependent tests self-skip in CI.

### VF-001 resolution status: **Closed (file verified present; location documented)**

> **Publication update:** when this project was pushed standalone to GitHub
> (`IPL-Prediction-merged`, with this directory as the repository root), the
> workflow was placed at this repo's own `.github/workflows/ci.yml` with the
> `working-directory` indirection removed — the assessment above describes
> the local multi-project workspace, where the git root sits one level up.

### VF-002: Match-date validation permits partially missing dates

- **Severity:** Medium
- **File:** `src/data.py`
- **Function/class:** `build_match_table`
- **Description:** `df.groupby("match_id")["date"].nunique()` ignores null
  values. A match with one populated date and one or more missing dates has a
  unique-count of one; pandas' `first` aggregation may then select the valid
  date and allow the match through validation.
- **Why it matters:** The timeline used by Elo, form, and head-to-head features
  can be accepted even though the source date is incomplete or internally
  inconsistent. This contradicts the report's claim that missing match dates
  are always rejected.
- **Suggested improvement:** For every match, require all delivery rows to
  have non-null dates and require exactly one unique parsed date before
  aggregation.
- **Assessment — valid.** Confirmed: `Series.nunique()` excludes NaN, so a
  match with one populated date and the rest missing passed the
  one-date-per-match check, and the groupby `first` aggregation can pick the
  non-null value, defeating the post-aggregation null check too. Latent
  rather than active: the real dataset has 0 null dates across all 278,205
  rows (verified directly), so no reported number was affected — but the
  validation claim in DEF-007's resolution was stronger than the code.
- **Resolution.** `build_match_table` now rejects any delivery row with a
  missing date *before* the uniqueness check, listing the offending match
  IDs. Regression test:
  `TestBuildMatchTable::test_partially_missing_dates_raise` (one populated
  date + rest missing — the exact case that used to slip through).

### VF-002 resolution status: **Closed (fixed in `src/data.py`, regression-tested)**

### VF-003: Horizon snapshots still depend on input row order

- **Severity:** Medium
- **File:** `src/pipeline.py`
- **Function/class:** `horizon_snapshot`
- **Description:** The function filters rows and returns
  `groupby("match_id").tail(1)` without first sorting delivery order. It is
  currently called with loader-preserved ordering, but the function itself has
  no ordering contract or enforcement.
- **Why it matters:** If row order changes upstream, the selected state at a
  score horizon can be incorrect, invalidating the fixed-horizon metrics that
  DEF-002 introduced.
- **Suggested improvement:** Sort deterministically by match, innings, over,
  and ball before selecting each horizon snapshot, and add a regression test
  with intentionally scrambled delivery rows.
- **Assessment — valid.** Confirmed: `groupby("match_id").tail(1)` inherited
  the caller's row order, so "the last row at this ball count" was only
  correct because the loader happens to preserve delivery order. That is the
  same class of implicit-ordering contract DEF-007 flagged elsewhere, and it
  should have been closed in the same pass.
- **Resolution.** `horizon_snapshot` now sorts by `(match_id, over, ball)`
  (stable mergesort) before selecting each snapshot; callers pass
  per-innings frames, so `innings` is not needed in the key. Regression
  test: `TestHorizonSnapshot::test_snapshot_is_independent_of_input_row_order`
  (fully scrambled rows select the identical state). `run_all.py` re-run:
  every horizon metric is unchanged, confirming this was latent.

### VF-003 resolution status: **Closed (fixed in `src/pipeline.py`, regression-tested, metrics unchanged)**

### VF-004: Checkpoint loader does not rebuild usable feature inputs

- **Severity:** Medium
- **Files:** `src/win_probability_engine.py`, `run_alt_transformer.py`
- **Function/class:** `WinProbabilityEngine.from_checkpoint`
- **Description:** The checkpoint stores the player registry and embedding
  seed, but `from_checkpoint()` discards that metadata and returns an engine
  that accepts only precomputed feature arrays. The round-trip test supplies
  random arrays, not features generated from real delivery data.
- **Why it matters:** A user cannot use the returned engine directly for a
  real match without separately loading and reconstructing the exact feature
  pipeline. The claimed deployment readiness is therefore incomplete.
- **Suggested improvement:** Expose checkpoint metadata through the engine or
  provide a loader/factory that rebuilds the registry, embedding lookup, and
  game-state-to-feature transformation; test a real data-to-prediction
  round-trip.
- **Assessment — valid.** Confirmed: the checkpoint carried the registry and
  seed, but `from_checkpoint()` threw them away, so "deployment ready" meant
  "ready if you rebuild the feature pipeline yourself" — and the round-trip
  test proved only that random arrays flow through the network.
- **Resolution.** Three pieces: (1) `from_checkpoint()` now retains
  everything except the weights on `engine.metadata`; (2) new
  `WinProbabilityEngine.features_from_deliveries(df)` rebuilds the embedding
  lookup from the checkpoint's registry + seed (bit-identical to training,
  same `torch.manual_seed` path) and converts raw delivery rows into
  `{(match_id, innings): (T, 120)}` feature sequences via the new
  `build_features_for_innings()` in `src/alt_transformer_data.py` — which is
  the training path itself refactored (`_stack_features` is now shared with
  `build_innings_sequences`), not a re-implementation that could drift;
  an engine built without a checkpoint refuses raw deliveries with a clear
  error. (3) Tests cover the full chain:
  synthetic deliveries → features → prediction, the no-metadata rejection,
  and the requested **real-data round-trip**
  (`test_real_data_to_prediction_roundtrip`: raw xlsx deliveries of a real
  match → checkpoint-reconstructed features → per-ball win probabilities;
  self-skips where the dataset is absent, runs locally).

### VF-004 resolution status: **Closed (engine + data-builder changes, synthetic and real-data round-trips tested)**

The headline takeaway is unchanged in direction but sharper: the dynamic in-game models are genuinely strong and essentially unaffected; the pre-match model has no demonstrated skill over the naive baseline, and the reporting now says so explicitly.

## Testing gaps

Status after the fix pass:

- ~~No test verifies grouped, chronological calibration behavior.~~ **Covered:** `tests/test_defect_fixes.py::TestMatchGroupedCV`.
- ~~No real-data schema/order/duplicate validation test was identified.~~ **Partially covered:** `TestBuildMatchTable` (order-independence, date retention, conflicting-date rejection) plus runtime validation in `build_match_table` and `evaluate_2026_pre_match`. A full schema contract for the xlsx remains open.
- ~~No test validates score metrics at exact horizons.~~ **Covered:** `TestHorizonSnapshot`.
- ~~No test protects against hard-coded external-label corrections.~~ **Covered:** `TestApplyLabelCorrections`; the code path that allowed silent overrides no longer exists (missing winners now raise).
- ~~No end-to-end reproducibility test rebuilds artifacts from source data.~~ **Partially covered:** CI runs the suite on every push (workflow verified present at the git root, `Merged/.github/workflows/ci.yml` — VF-001 was looking inside the project subdirectory, where GitHub would not read it). A full artifact-rebuild test still requires the uncommitted raw dataset and remains open.
- ~~Dashboard rendering and dashboard-data schema are not tested.~~ **Partially covered:** `TestDashboardExportRobustness` tests the export/merge layer and the JSON artifact; in-browser rendering remains untested.
- Edge cases still needing explicit coverage: ties, abandoned/DLS matches, super overs, unknown teams, single-class folds, and empty external inputs. **Open** (unchanged).

## Improvement priorities

1. ~~Repair the temporal and match-grouped evaluation protocol.~~ **Done** (DEF-001, DEF-007; VF-002's partial-missing-date loophole closed).
2. ~~Remove the hard-coded external label override and establish auditable data provenance.~~ **Done** (DEF-003).
3. ~~Rework score regression to fixed-horizon evaluation.~~ **Done** (DEF-002; VF-003's row-order dependence closed, metrics unchanged).
4. ~~Persist deployable artifacts with feature and preprocessing metadata.~~ **Done** (DEF-010; VF-004 closed — the engine now reconstructs the feature pipeline from checkpoint metadata, with a real-data round-trip test).
5. Integrate or retire dormant model and feature modules. **Done for `make_zoo` (retired) and the engine (now genuinely usable end-to-end); `player_features.py` is explicitly statused as reference-only pending item 6.**
6. Add venue, roster, player availability, season-decayed Elo, rolling form, and toss-by-venue interactions only after the evaluation protocol is reliable. **Now unblocked** — items 1–4 are complete.

## Positive observations

- The source modules have mostly focused responsibilities and readable names.
- The README and limitations document disclose important weaknesses, including poor pre-match performance — and after this fix pass they disclose the (worse) corrected external result rather than the flattering one.
- Utility tests cover Elo, features, metrics, calibration, game-state invariants, and basic Transformer behavior; the suite grew from 150 to 172 tests across the two fix passes, re-executed in full after the VF fixes (172 passed).
- Evaluation uses a season-based internal holdout rather than a simple random split — which is also why DEF-001's leakage never reached the reported test metrics.
- Calibration measures are reported alongside discrimination metrics.
