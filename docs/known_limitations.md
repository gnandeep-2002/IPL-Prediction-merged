# Known Limitations

Honest caveats carried forward from `COMPARISON_REPORT.md` and
`DEEP_COMPARISON.md`, plus new limitations introduced by the merge itself.
Read this before quoting any number from this project.

## Carried forward from project_gagan

- **Pre-match win prediction is barely better than chance.** Internal
  LOSO cross-validation AUC is ~0.51. The 74-match 2026 holdout accuracy
  (64.9%) is nominally better than the "always pick bowl-first" baseline
  (63.5%), but gagan's own code says this plainly: *"pre-match win
  prediction in T20 cricket is inherently limited... the 2026 figure may
  reflect variance over 74 matches rather than genuine model skill."*
  `run_all.py` prints this caveat every time it runs.
- **Pre-match score regression is worse than predicting the mean.** R² is
  negative (-0.17 to -0.48) for every model at the pre-match stage. In-game
  score regression (once a few overs have been bowled) is much more useful
  and should be trusted; pre-match should not.

## Carried forward from project_hrishav

- **The alternative Transformer model (src/transformer_model.py +
  run_alt_transformer.py) is not the recommended win-probability path.**
  hrishav's own bootstrap-CI analysis (2000 resamples) found its
  Transformer does not beat a simple calibrated Logistic Regression at a
  statistically significant level, and this merge's default path
  (project_gagan's calibrated LogReg/GBT zoo) has independently verified
  stronger metrics on its own data (see `COMPARISON_REPORT.md`). The
  Transformer is kept available for comparison, clearly labeled as
  secondary, per the approved merge decision -- not because it's the
  better model.
- **hrishav's CatBoost Brier and XGBoost-tuned AUC had unresolved
  cross-source discrepancies** (CLAUDE.md vs. `task7_comparison_with_cis.csv`
  vs. the live rerun in this session -- see `DEEP_COMPARISON.md`'s "Notable
  discrepancies"). Neither hrishav's tuned baselines nor native XGBoost
  were ported into this merged project at all (see below), so this
  specific discrepancy does not affect any number reported here -- noted
  for completeness only.

## New limitations introduced by this merge

- **PSM-adjusted Impact Player analysis was NOT ported.** hrishav's
  `impact_player.py` (938 lines, including the propensity-score-matched
  causal analysis) fundamentally depends on `impact_player_in` /
  `impact_player_out` columns, which Cricsheet's JSON populates from a
  match's `replacements` field when `reason == "impact_player"`.
  project_gagan's `ipl_data.xlsx` schema has no equivalent column, and
  there is no reliable way to infer an Impact Player substitution from
  the raw ball-by-ball batter/bowler/non-striker names alone (a bowling
  change or fielding substitution looks identical in that data). Per the
  approved merge decision ("keep gagan's xlsx loader only"), and since
  porting ~1000 lines of code that can never be exercised or tested
  against real data in this repository would violate the "every file
  should trace to a real, used feature" constraint, this analysis was
  left out entirely rather than ported non-functionally. If Cricsheet
  JSON (or another data source with impact-player markers) is ever added
  alongside the xlsx loader, `project_hrishav/impact_player.py` is the
  reference implementation to adapt.
- **project_gagan's LSTM (dynamic 2nd-innings sequence model) was not
  ported.** Its own reported numbers (`DEEP_COMPARISON.md` Table 5) show
  the calibrated Logistic Regression model beats the LSTM at every
  reported over horizon (5/10/15/18) on both accuracy and AUC, so there
  was no accuracy case for carrying the TensorFlow/Keras dependency into
  this merged project.
- **hrishav's Optuna-tuned baselines (LogReg/LightGBM/CatBoost/XGBoost)
  and native XGBoost were not ported.** project_gagan's calibrated
  LogReg/GBT zoo is the approved default win-probability path, and native
  XGBoost requires Homebrew's `libomp` which is unavailable in some
  environments (this was hit directly during Phase 1 investigation) --
  the merged project defaults to gagan's own sklearn `HistGradientBoosting`
  fallback pattern instead, avoiding that dependency entirely.
- **The alternative Transformer's reported metrics are NOT comparable to
  hrishav's own headline numbers.** It is trained here on project_gagan's
  `ipl_data.xlsx`-derived data (different date range, different
  train/val/test season boundaries, different game-state feature
  computation -- see `src/game_state.py`'s module docstring) rather than
  hrishav's original Cricsheet JSON. Treat any number from
  `run_alt_transformer.py` as a sanity-check that the port works, not as
  a reproduction of hrishav's Brier/AUC figures.
- **`src/game_state.py`'s cumulative per-ball features (partnership,
  in-innings batter/bowler form, runs-in-last-over) are a from-scratch
  vectorised reimplementation**, not a direct port of hrishav's
  `data_loader.py` row-by-row accumulation logic (which depended on
  Cricsheet-specific parsing state gagan's data doesn't have). The feature
  *definitions* match; the *exact numeric values* have not been
  cross-validated ball-for-ball against hrishav's original pipeline, since
  the two now run on different underlying data.
