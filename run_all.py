"""
Run the merged IPL pipeline end-to-end: internal evaluation (train 2008-2020,
test 2021-2025), phase-specific evaluation, score regression, and the locked
2026 external holdout evaluation.

Usage: python3 run_all.py
"""
from __future__ import annotations

import sys

import pandas as pd

from src.pipeline import (
    load_and_prepare, train_pre_match_internal, build_dynamic_2nd,
    build_dynamic_1st, train_dynamic_internal, phase_specific_eval,
    train_score_zoo_and_save, retrain_pre_match_full, evaluate_2026_pre_match,
    PRE_FEAT,
)
from src.explainability import shap_importance
from src.tournament import actual_points_table, predicted_points_table, compare_tables

DATA_XLSX = "data/raw/ipl_data.xlsx"
PM26_CSV = "data/external_2026/prematch_dataset.csv"
IG26_CSV = "data/external_2026/ingame_dataset.csv"
SCORE_PIPELINE_PATH = "models/ipl_score_pipeline.pkl"


def main() -> None:
    print("Loading data and computing Elo/form/H2H features...")
    df, match_df = load_and_prepare(DATA_XLSX)
    print(f"  Matches: {len(match_df)}  Deliveries: {len(df):,}")

    print("\n1. PRE-MATCH MODEL (internal: train <=2020, test >2020)")
    pre = train_pre_match_internal(match_df)
    print(f"   Climatology  Brier={pre['climatology']['brier']:.4f}  AUC={pre['climatology']['auc']:.4f}")
    print(f"   Cal. LR      Brier={pre['cal_lr']['brier']:.4f}  AUC={pre['cal_lr']['auc']:.4f}  "
          f"Acc={pre['cal_lr']['acc']:.4f}  BSS={pre['cal_lr']['bss']:.4f}  ECE={pre['cal_lr']['ece']:.4f}")
    print(f"   Cal. GBT     Brier={pre['cal_gbt']['brier']:.4f}  AUC={pre['cal_gbt']['auc']:.4f}  "
          f"Acc={pre['cal_gbt']['acc']:.4f}  BSS={pre['cal_gbt']['bss']:.4f}  ECE={pre['cal_gbt']['ece']:.4f}")

    print("\n2. DYNAMIC 2ND/1ST-INNINGS MODELS (internal)")
    df2 = build_dynamic_2nd(df, match_df)
    df1 = build_dynamic_1st(df, match_df)
    dyn = train_dynamic_internal(df2, df1)
    print(f"   Dynamic 2nd  Brier={dyn['dyn2']['brier']:.4f}  AUC={dyn['dyn2']['auc']:.4f}  "
          f"Acc={dyn['dyn2']['acc']:.4f}  ECE={dyn['dyn2']['ece']:.4f}")
    print(f"   Dynamic 1st  Brier={dyn['dyn1']['brier']:.4f}  AUC={dyn['dyn1']['auc']:.4f}  "
          f"Acc={dyn['dyn1']['acc']:.4f}")

    print("\n3. PHASE-SPECIFIC EVALUATION (2nd innings)")
    phases = phase_specific_eval(dyn["train2"], dyn["test2"], dyn["dsc2"])
    for name, m in phases.items():
        print(f"   {name:22s}  AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  n={m['n']}")

    print("\n4. SCORE REGRESSION ZOO (saved to models/ipl_score_pipeline.pkl)")
    score_metrics = train_score_zoo_and_save(df1, df2, match_df, SCORE_PIPELINE_PATH)
    for stage_name, stage_metrics in score_metrics.items():
        print(f"   {stage_name}:")
        for model_name, m in stage_metrics.items():
            print(f"     {model_name:15s} MAE={m['MAE']:.1f}  R2={m['R2']:.3f}")
    print("   NOTE: pre-match score R2 is expected to be negative (worse than "
          "predicting the mean) -- this mirrors project_gagan's own finding and "
          "is not a bug. See docs/known_limitations.md.")

    print("\n5. EXTERNAL 2026 HOLDOUT EVALUATION (locked, never used above)")
    print("   Retraining pre-match model on full 2008-2025 data first...")
    pre_model_full, sc_pre_full = retrain_pre_match_full(match_df)
    ext = evaluate_2026_pre_match(match_df, pre_model_full, sc_pre_full, PM26_CSV)
    print(f"   Pre-match accuracy: {ext['accuracy']:.1%}  ({ext['k']}/{ext['n']})")
    print(f"   95% CI (Clopper-Pearson): [{ext['ci'][0]:.1%}, {ext['ci'][1]:.1%}]")
    print(f"   p-value vs 50/50: {ext['p_value']:.3f}")
    print(f"   Naive majority-class baseline: {ext['naive_baseline_acc']:.1%}")
    print("   NOTE: pre-match AUC is ~0.51 on internal LOSO (near-random) -- the "
          "2026 accuracy figure may reflect variance over 74 matches rather than "
          "genuine model skill. See docs/known_limitations.md.")

    print("\n6. SHAP EXPLAINABILITY (pre-match Cal. GBT model)")
    train_m = match_df[match_df["year"] <= 2020]
    test_m = match_df[match_df["year"] > 2020]
    X_tr = pre["sc"].transform(train_m[PRE_FEAT])
    X_te = pre["sc"].transform(test_m[PRE_FEAT])
    shap_result = shap_importance(pre["cal_gbt_model"], X_tr, X_te, PRE_FEAT)
    for feat, val in shap_result["importance"].items():
        print(f"   {feat:18s} mean|SHAP|={val:.4f}")

    print("\n7. 2026 TOURNAMENT SIMULATION (actual vs. predicted points table)")
    ig26 = pd.read_csv(IG26_CSV)
    table_actual = actual_points_table(ext["pm26"], ig26)
    table_pred = predicted_points_table(ext["pre_df"])
    cmp = compare_tables(table_actual, table_pred)
    print(f"   Actual table topper   : {cmp['actual_topper']}")
    print(f"   Predicted table topper: {cmp['predicted_topper']}  "
          f"({'CORRECT' if cmp['topper_correct'] else 'WRONG'})")
    print(f"   Top-4 overlap: {cmp['top4_overlap_count']}/4  {cmp['top4_overlap']}")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
