# Refactor Suggestions

This document collects safe, behavior-preserving refactor ideas for the merged IPL project.

The goal here is not to change model behavior or reported metrics. The focus is on reducing duplication, making the flow easier to follow, and lowering the cost of future changes.

> **Implementation update (2026-07-14):** the suggestions were assessed for
> validity and safety, and the safe subset was implemented. Each item below
> now carries a **Status** block recording exactly what was done (or why it
> was deliberately skipped or narrowed). Behavior preservation was verified
> by byte-diffing full pipeline outputs before and after the refactors — see
> **Verification record** at the end, including the pre-existing
> nondeterminism bug the verification uncovered.

## 1) Split `run_all.py` into section-specific functions

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/run_all.py`

Current issue:

- `main()` orchestrates the entire end-to-end run in one long block.
- It loads data, trains models, prints metrics, runs SHAP, runs tournament simulation, updates the dashboard, and writes the report.

Suggested refactor:

- Extract one function per section, for example:
  - `run_pre_match_section(...)`
  - `run_dynamic_section(...)`
  - `run_phase_section(...)`
  - `run_score_section(...)`
  - `run_external_eval_section(...)`
  - `run_explainability_section(...)`
  - `run_tournament_section(...)`
  - `run_calibration_section(...)`
- Keep `main()` as a lightweight orchestrator that calls those helpers in order.

Why this helps:

- Makes the control flow easier to scan.
- Reduces the chance of accidental coupling between sections.
- Makes it easier to test individual pieces without running the full pipeline.

Priority: High

**Status: Implemented.** `main()` is now a thin orchestrator over ten
section functions (`run_pre_match_section` … `run_calibration_section`,
plus `run_reliability_section` and `run_elo_section` for the two sections
this list omitted) and a final `export_dashboard()` for the DEF-011 push.
One correction to the premise, though: the sections are *not* independent,
and the refactor makes that visible rather than pretending otherwise — the
pre-match results feed SHAP (section 6) and the reliability bins (section
8), the external evaluation feeds the tournament simulation (section 7),
and the dashboard export consumes results from nearly every section. Each
helper therefore returns its results and `main()` threads them explicitly;
the module docstring documents this coupling. Note also that no test covers
`run_all.py`, so this refactor was verified by the output byte-diff
described in the Verification record, not by unit tests.

## 2) Extract repeated model-fit / score / metric patterns in `src/pipeline.py`

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/pipeline.py`

Current issue:

- The same sequence appears multiple times:
  - split train/test
  - scale features
  - fit a classifier
  - predict probabilities
  - compute Brier / AUC / accuracy / ECE
- Similar logic is duplicated across:
  - `train_pre_match_internal`
  - `train_dynamic_internal`
  - `phase_specific_eval`
  - `compare_calibration_methods_dyn2`

Suggested refactor:

- Add small shared helpers such as:
  - `fit_standard_scaled_classifier(...)`
  - `score_probabilities(...)`
  - `build_metric_row(...)`
- Let the public functions keep their current return shape, but delegate the repeated mechanics to helpers.

Why this helps:

- Less copy/paste means fewer places to update when evaluation changes.
- Keeps the model-specific logic visible while hiding the repetitive boilerplate.

Priority: High

**Status: Implemented in narrowed form — the metric row only.** A shared
`classification_metrics(y, p, p_ref=None, with_ece=True)` helper now backs
the pre-match rows (with BSS vs. the climatology baseline) and both dynamic
rows (dyn1 without ECE), preserving the historical key order so the report
and dashboard JSON are unchanged. The proposed
`fit_standard_scaled_classifier(...)` was **deliberately not built**: the
fit mechanics differ across these functions on purpose, and flattening them
into one helper is exactly how the DEF-001 fix could silently regress —
pre-match models use `cv=5` (stratified folds are correct on
one-row-per-match data) while every delivery-level model must use
`match_grouped_cv`; `phase_specific_eval` intentionally reuses the scaler
fitted on the full training frame and then column-slices it; and
`compare_calibration_methods_dyn2` has its own three-way year split. Those
differences stay inline where a reviewer can see them.

## 3) Break `build_game_state_matrix()` into smaller helpers

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/game_state.py`

Current issue:

- `build_game_state_matrix()` currently does everything:
  - legal-ball accounting
  - score/wicket deltas
  - over history
  - wicket windows
  - partnership tracking
  - batter stats
  - bowler stats
  - dot/boundary rates
  - chase-state features
  - final feature matrix assembly

Suggested refactor:

- Split the function into clearly named steps, for example:
  - `_add_legal_ball_features(...)`
  - `_add_over_history_features(...)`
  - `_add_wicket_window_features(...)`
  - `_add_partnership_features(...)`
  - `_add_batter_features(...)`
  - `_add_bowler_features(...)`
  - `_add_chase_state_features(...)`
  - `_assemble_game_state_matrix(...)`
- Keep the same output and the same validation rules.

Why this helps:

- The function becomes much easier to reason about.
- Each step can be reviewed and tested independently.
- Future feature additions will be less risky.

Priority: High

**Status: Implemented.** `build_game_state_matrix()` is now a sort/reset
followed by seven `_add_*` steps plus `_assemble_matrix()`:
`_add_legal_ball_state`, `_add_last_over_runs`, `_add_wicket_window`,
`_add_partnership_state`, `_add_batter_bowler_state` (batter and bowler
share one helper — same pattern, same groupby shape),
`_add_boundary_dot_state`, `_add_chase_state`. Every expression and every
DEF-005/DEF-006 comment moved verbatim into the relevant helper. Two
constraints the suggestion didn't mention are now documented in the module
docstring because they are load-bearing: the helper *order* matters
(`legal_balls_before` feeds the run rate, wicket window, partnership,
batter/bowler, and chase-state steps), and `_add_last_over_runs`'s merge
rebuilds the frame, so the wicket-window step's positional-index writes
depend on the clean RangeIndex being preserved. The existing real-data
invariant tests (`tests/test_game_state.py`) and the synthetic DEF-005/006
regressions passed unchanged — this item had the strongest safety net of
the eight.

## 4) Remove or isolate dead / unused variables and imports

Files:

- `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/pipeline.py`
- `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/game_state.py`

Current issue:

- Some values appear to be computed but never used.
- Example: `over_runs` in `src/game_state.py` is assigned but not referenced later.
- `src/pipeline.py` also carries a few imports and constants that are easy to miss unless you read the whole file carefully.

Suggested refactor:

- Remove unused local variables.
- Group imports by purpose.
- If a constant is only there for documentation, label it explicitly or move it next to the code that uses it.

Why this helps:

- Lowers cognitive load.
- Makes actual dependencies easier to spot.
- Reduces the chance that future readers assume a variable has a hidden role.

Priority: Medium

**Status: Implemented, with two corrections to the premise.**
(1) `over_runs` — confirmed dead (assigned once, never referenced; the
actual feature comes from `prev_over_map`) and removed. (2) The
`src/pipeline.py` claim did **not** verify: every import and constant there
is used (`SEED` seeds the GBT classifier, `phase_vec` builds the dynamic
frames, `calibration_bins` feeds the calibration comparison), so nothing
was removed from it; imports were already grouped stdlib/third-party/local.
(3) One important non-removal: `legal_balls_total` in `game_state.py` now
*looks* dead in the feature matrix (the DEF-005 fix switched feature index
1 to `legal_balls_before`), but it is an intentional validation column that
tests cross-check 1:1 against the source's `team_balls` — a naive
dead-code sweep would delete it and break the suite. Its comment says so.
Future cleanups of this kind should be linter-driven (ruff/pyflakes)
rather than by eye; no linter was available in this environment, so each
candidate was verified by grep before touching it.

## 5) Improve the year-by-year aggregation flow in `src/player_features.py`

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/player_features.py`

Current issue:

- `compute_rolling_player_stats()` and `compute_matchup_features()` rebuild cumulative history with repeated `pd.concat(...)` inside a loop.
- That is readable, but it is not ideal for larger datasets.

Suggested refactor:

- Build the season slices once, then accumulate in a more explicit structure.
- If the project may grow, consider a one-pass accumulator or a list-based concat strategy.

Why this helps:

- Better performance as the dataset grows.
- Makes the “history up to year Y” logic more obvious.

Priority: Medium

**Status: Implemented.** Both functions now compute
`history = df[df["year"] < year]` inside the loop instead of rebuilding an
ever-growing `pd.concat` frame — the walk-forward invariant ("year Y sees
exactly the balls from seasons < Y") is now a single literal expression
rather than a property of accumulation order. Equivalence is provable, not
assumed: every downstream aggregation in `_aggregate_player_stats` and
`_aggregate_matchup` is an order-insensitive groupby sum/count/nunique, so
row order and index differences cannot change results, and
`tests/test_player_features.py` passed unchanged. Caveat kept from the
assessment: this module is reference-only (not consumed by any default
pipeline path — see its DEF-010 status note), so the performance benefit is
latent until something integrates it.

## 6) Return a typed result object from `compare_calibration()`

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/temperature_scaling.py`

Current issue:

- `compare_calibration()` returns a wide dictionary with many fields.
- That works, but the shape is easy to mistype and hard to discover from tooling.

Suggested refactor:

- Replace the raw dict with a small dataclass such as `CalibrationComparisonResult`.
- Keep a `.to_dict()` helper if existing callers expect a mapping.

Why this helps:

- Better autocomplete and type safety.
- Easier to see which fields are intentional.
- Less fragile when additional calibration outputs are added.

Priority: Low

**Status: Not implemented — deliberately rejected.** As written, this one
is not behavior-preserving: `compare_calibration()`'s dict flows through
`compare_calibration_methods_dyn2` into `update_dashboard_data`, which
serializes it with `json.dumps` using a `default=` hook that handles numpy
types only — a dataclass would raise `TypeError` at the end of every
`run_all.py` run. `run_all.py` also subscripts the result (`t["T"]`,
`t["brier_raw"]`, …), and both `tests/test_temperature_scaling.py` and
`tests/test_pipeline.py` assert dict keys. Making it work needs a
`.to_dict()` conversion at the `run_all` boundary plus call-site and test
updates — real churn for an autocomplete-only win on a stable, tested
shape. If the calibration output ever grows new fields, revisit then.

## 7) Simplify `report`/logging concerns in `run_all.py`

File: `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/run_all.py`

Current issue:

- `Report` mixes formatting and orchestration concerns.
- The file also directly handles warnings, report writing, console output, and dashboard summary messaging.

Suggested refactor:

- Keep `Report` as a formatting helper only.
- Move setup and execution concerns into separate small functions.
- Consider a helper for repetitive `r.row(...)` and `r.note(...)` patterns.

Why this helps:

- Keeps presentation logic separate from pipeline execution.
- Makes future CLI changes less invasive.

Priority: Medium

**Status: Implemented, folded into item 1.** After the orchestration split,
`Report` contains formatting only (its docstring now states that
contract), `_setup()` keeps the warnings/logging wiring, execution lives in
the `run_*_section` functions, and the dashboard summary messaging moved
into `export_dashboard()`. The suggested wrapper for repetitive
`r.row(...)`/`r.note(...)` calls was **skipped**: the call sites differ in
which metrics they print and how they phrase notes, so a wrapper would just
relocate the differences behind another layer without removing any.

## 8) Consider consolidating repeated “train/test split by year” logic

Files:

- `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/src/pipeline.py`
- `/Users/gnandeepboppudi/Documents/MSC/FInal project/Merged/merged_project/run_all.py`

Current issue:

- The same year-based split logic appears in multiple places.
- The cutoff values are meaningful and should stay visible, but the mechanics are repeated.

Suggested refactor:

- Add a small split helper such as:
  - `split_by_year(df, train_end_year=2020)`
  - `split_2021_holdout(...)`
- Keep the actual cutoff values in one place.

Why this helps:

- Fewer copy/paste mistakes.
- Makes it easier to change the evaluation protocol intentionally later.

Priority: Medium

**Status: Implemented.** `split_by_year(df, train_end_year=2020)` lives in
`src/pipeline.py` and is used by `train_pre_match_internal`,
`train_dynamic_internal` (both innings), `train_score_zoo_and_save`
(pre-match table and both innings zoos), and `run_all.py`'s SHAP section
(which previously re-derived the same split by hand). The separate
`split_2021_holdout(...)` was unnecessary — the default argument covers it.
One deliberate exception, documented in the helper's docstring:
`compare_calibration_methods_dyn2` keeps its explicit three-way
2018/2020 split, because temperature scaling needs its own validation
window carved out of the training years and hiding that inside a generic
helper would obscure the one place the protocol intentionally differs.

## Recommended order

If we want the biggest payoff with the least risk, I’d do the refactors in this order:

1. `run_all.py` orchestration split
2. Shared helpers in `src/pipeline.py`
3. `build_game_state_matrix()` decomposition
4. Cleanup of dead variables / imports
5. `player_features.py` performance cleanup
6. `temperature_scaling.py` typed result object

That sequence gives the biggest readability win first, while keeping behavior stable.

**Order actually used (risk-first, not readability-first):** the split
helper (#8) and metric-row helper (#2, narrowed) went in first as small,
mechanically verifiable wins; then the `game_state` decomposition (#3,
protected by the strongest test coverage); then the dead-variable removal
(#4) and `player_features` cleanup (#5); then the `run_all.py` split
(#1/#7, the largest but least logic-bearing change). #6 was rejected — see
its Status block.

## Notes

- These are suggestion-only changes.
- Nothing in this document requires immediate code modification.
- If you want, I can turn any item above into a concrete patch plan next, but I will not edit code unless you approve it first.
- **Update (2026-07-14):** implementation was approved and carried out; the
  Status blocks above and the Verification record below document the result.

## Verification record (2026-07-14)

How "behavior-preserving" was actually checked, not assumed:

1. **Test suite:** all 172 tests pass after every item, including the
   real-data game-state invariants and the DEF-001…DEF-011 regression tests.
2. **Byte-diff protocol:** `reports/run_summary.txt` and
   `reports/dashboard_data.json` were snapshotted before the refactors, and
   `run_all.py` was re-run after. Every deterministic value — all
   classifier metrics, four of the five score regressors at every horizon,
   the external 2026 evaluation, tournament simulation, Elo history,
   reliability bins, and the calibration comparison — was **byte-identical**.
3. **What the diff caught:** the only values that moved were the SVR rows
   and the SHAP importances. Investigation showed this was *pre-existing
   run-to-run nondeterminism*, not refactor damage: `LinearSVR` was the one
   zoo member without a `random_state` (demonstrated in isolation —
   unseeded fits on identical data differ by up to 0.043 in coefficients;
   seeded fits are exactly equal), and SHAP's permutation explainer was
   unseeded (only its background *sample* was seeded). Both are now pinned
   to `SEED=42` — the one deliberate behavior delta of this pass, within
   the noise band those values already occupied, and recorded in
   `DEFECT_REPORT.md` under DEF-012.
4. **Determinism proof:** after seeding, two consecutive full `run_all.py`
   executions differ **only in the report's timestamp line**, and the
   dashboard JSON artifact is byte-identical across runs. That protocol
   (`pytest` + double-run byte-diff) is now the standing way to verify any
   future refactor of this pipeline.
