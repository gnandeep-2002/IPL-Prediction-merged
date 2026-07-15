"""
Builds innings-level training sequences for the alternative Transformer
(src/transformer_model.py) from project_gagan's ball-by-ball data:
24-dim game state (src/game_state.py) concatenated with three 32-dim
player embeddings (batter, bowler, non-striker), looked up from a fixed
random-init embedding table (see transformer_model.PlayerEmbedTable's
docstring for why GNN pretraining was not ported).

Ported/adapted from project_hrishav/train.py's build_innings_sequences()
and player_graph.py's PlayerEmbedLookup.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.game_state import build_game_state_matrix, GAME_STATE_DIM
from src.transformer_model import PlayerEmbedTable, PLAYER_EMBED_DIM, NEXT_BALL_MAP

SEED = 42


def build_player_registry(df: pd.DataFrame) -> dict[str, int]:
    """Deterministic player_name -> int id map (0 reserved for padding/unknown)."""
    players = sorted(set(df["batter"]) | set(df["bowler"]) | set(df["non_striker"]))
    return {name: i + 1 for i, name in enumerate(players)}


def build_embedding_lookup(registry: dict[str, int], seed: int = SEED) -> dict[str, np.ndarray]:
    """Fixed random Xavier-init embedding per player (frozen, not trained further)."""
    torch.manual_seed(seed)
    table = PlayerEmbedTable(num_players=len(registry), embed_dim=PLAYER_EMBED_DIM)
    all_embeds = table.embed.weight.detach().numpy()  # (num_players+1, embed_dim)
    return {name: all_embeds[idx] for name, idx in registry.items()}


def _encode_next_ball(row) -> int:
    """
    Class label for the outcome of THIS delivery row (0/1/2/3/4/6/W).

    DEF-004: the label intentionally comes from the same row as the input
    state. The game-state features at position t are strictly pre-ball
    (score_before/wickets_before/legal_balls_before...), so the imminent
    delivery -- the "next ball" the head is documented to predict -- is
    delivery t itself. No one-ball shift is needed or wanted; see the
    src/transformer_model.py module docstring.
    """
    if row["is_wicket"] == 1:
        return 6
    return NEXT_BALL_MAP.get(int(row["runs_batter"]), 0)


def _stack_features(d: pd.DataFrame, game_state: np.ndarray,
                    embed_lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Game state + batter/bowler/non-striker embeddings -> (N, 120) matrix.
    Unknown players get a zero embedding (same convention as training)."""
    zero = next(iter(embed_lookup.values())) * 0
    bat_emb = np.stack([embed_lookup.get(n, zero) for n in d["batter"]])
    bowl_emb = np.stack([embed_lookup.get(n, zero) for n in d["bowler"]])
    non_emb = np.stack([embed_lookup.get(n, zero) for n in d["non_striker"]])
    features = np.concatenate([game_state, bat_emb, bowl_emb, non_emb], axis=1).astype(np.float32)
    assert features.shape[1] == GAME_STATE_DIM + 3 * PLAYER_EMBED_DIM
    return features


def build_features_for_innings(
    df: pd.DataFrame, embed_lookup: dict[str, np.ndarray],
) -> dict[tuple, np.ndarray]:
    """
    VF-004: label-free twin of build_innings_sequences for INFERENCE -- turns
    raw delivery rows into the exact (T, 120) feature sequences the model was
    trained on, keyed by (match_id, innings). Used by
    WinProbabilityEngine.features_from_deliveries so a persisted checkpoint
    can be applied to real match data without re-deriving the feature
    pipeline by hand.
    """
    game_state, d = build_game_state_matrix(df)
    d = d.reset_index(drop=True)
    features = _stack_features(d, game_state, embed_lookup)
    return {
        key: features[np.asarray(idx)]
        for key, idx in d.groupby(["match_id", "innings"]).groups.items()
    }


def build_innings_sequences(
    df: pd.DataFrame, embed_lookup: dict[str, np.ndarray], years: set[int],
) -> list[dict]:
    """
    One sequence per (match_id, innings) with year in `years`.

    Returns a list of dicts: features (T, 120) float32, win_label (0/1),
    nb_labels (T,) int, score_label (float, final_score/250).
    """
    subset = df[df["year"].isin(years)]
    if subset.empty:
        return []

    game_state, d = build_game_state_matrix(subset)
    d = d.reset_index(drop=True)
    features = _stack_features(d, game_state, embed_lookup)

    d["nb_label"] = d.apply(_encode_next_ball, axis=1)
    final_scores = d.groupby(["match_id", "innings"])["runs_total"].transform("sum")
    win_labels = d.groupby(["match_id", "innings"])["batting_wins"].transform("first")

    sequences = []
    for (mid, inn), idx in d.groupby(["match_id", "innings"]).groups.items():
        idx = np.asarray(idx)
        sequences.append({
            "features": features[idx],
            "win_label": float(win_labels.iloc[idx[0]]),
            "nb_labels": d["nb_label"].values[idx],
            "score_label": float(final_scores.iloc[idx[0]]) / 250.0,
        })
    return sequences
