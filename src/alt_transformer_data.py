from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.game_state import build_game_state_matrix, GAME_STATE_DIM
from src.transformer_model import PlayerEmbedTable, PLAYER_EMBED_DIM, NEXT_BALL_MAP

SEED = 42


def build_player_registry(df: pd.DataFrame) -> dict[str, int]:
    players = sorted(set(df["batter"]) | set(df["bowler"]) | set(df["non_striker"]))
    return {name: i + 1 for i, name in enumerate(players)}


def build_embedding_lookup(registry: dict[str, int], seed: int = SEED) -> dict[str, np.ndarray]:
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        table = PlayerEmbedTable(num_players=len(registry), embed_dim=PLAYER_EMBED_DIM)
        all_embeds = table.embed.weight.detach().numpy()
    finally:
        torch.set_rng_state(rng_state)
    return {name: all_embeds[idx] for name, idx in registry.items()}


def _encode_next_ball(row) -> int:
    if row["is_wicket"] == 1:
        return 6
    return NEXT_BALL_MAP.get(int(row["runs_batter"]), 0)


def _stack_features(d: pd.DataFrame, game_state: np.ndarray,
                    embed_lookup: dict[str, np.ndarray]) -> np.ndarray:
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
