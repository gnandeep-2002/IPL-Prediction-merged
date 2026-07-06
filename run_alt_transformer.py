"""
Train and evaluate the ALTERNATIVE win-probability model (causal
Transformer over ball-by-ball sequences, hrishav-derived architecture)
on project_gagan's data.

This is a secondary, clearly-labeled alternative to the default
win-probability path in run_all.py (calibrated LogReg/GBT on team-level
features). hrishav's own bootstrap-CI analysis found this architecture
does not beat a simple calibrated LogReg baseline at a statistically
significant level -- see docs/known_limitations.md. It is kept available
here for comparison, not as the recommended model.

Usage: python3 run_alt_transformer.py [--epochs N]
"""
from __future__ import annotations

import argparse
import sys

from src.data import load_ball_by_ball
from src.alt_transformer_data import build_player_registry, build_embedding_lookup, build_innings_sequences
from src.alt_transformer_train import train_alt_transformer

DATA_XLSX = "data/raw/ipl_data.xlsx"

TRAIN_YEARS = set(range(2008, 2019))   # 2008-2018
VAL_YEARS = {2019, 2020}
TEST_YEARS = set(range(2021, 2026))    # 2021-2025, matches run_all.py's internal test window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    print("Loading data...")
    df = load_ball_by_ball(DATA_XLSX)

    print("Building player embedding table (fixed random init, not GNN-pretrained --")
    print("see src/transformer_model.py docstring)...")
    registry = build_player_registry(df)
    embed_lookup = build_embedding_lookup(registry)
    print(f"  {len(registry)} unique players")

    print("Building innings sequences (train/val/test)...")
    train_seqs = build_innings_sequences(df, embed_lookup, TRAIN_YEARS)
    val_seqs = build_innings_sequences(df, embed_lookup, VAL_YEARS)
    test_seqs = build_innings_sequences(df, embed_lookup, TEST_YEARS)
    print(f"  train={len(train_seqs)}  val={len(val_seqs)}  test={len(test_seqs)} innings")

    print(f"Training IPLTransformer (win-only, {args.epochs} epochs)...")
    result = train_alt_transformer(train_seqs, val_seqs, test_seqs, epochs=args.epochs)

    print("\nRESULTS (win probability at the final ball of each innings)")
    print(f"  Best val Brier: {result['best_val_brier']:.4f}")
    if "test_brier" in result:
        print(f"  Test Brier: {result['test_brier']:.4f}  AUC: {result['test_auc']:.4f}  n={result['test_n']}")
    print("\nFor comparison, run_all.py's default calibrated LogReg/GBT models")
    print("(team-level features) are the recommended win-probability path.")


if __name__ == "__main__":
    sys.exit(main())
