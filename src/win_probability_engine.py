from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.transformer_model import IPLTransformer

MC_SAMPLES = 50


@dataclass
class WinProbResult:
    ball_indices: np.ndarray
    win_prob_mean: np.ndarray
    win_prob_std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    score_mean: np.ndarray
    score_std: np.ndarray
    over_labels: list[str]


@dataclass
class MatchWinProb:
    innings1: WinProbResult | None
    innings2: WinProbResult | None
    batting_team1: str
    batting_team2: str


class WinProbabilityEngine:

    def __init__(self, model: IPLTransformer, mc_samples: int = MC_SAMPLES, device: str = "cpu"):
        self.model = model.to(device)
        self.mc_samples = mc_samples
        self.device = device
        self.metadata: dict = {}
        self._embed_lookup: dict | None = None

    @classmethod
    def from_checkpoint(cls, path: str, mc_samples: int = MC_SAMPLES,
                        device: str = "cpu") -> "WinProbabilityEngine":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = IPLTransformer()
        model.load_state_dict(ckpt["state_dict"])
        engine = cls(model, mc_samples=mc_samples, device=device)
        engine.metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
        return engine

    def features_from_deliveries(self, df) -> dict:
        if not self.metadata.get("player_registry"):
            raise ValueError(
                "features_from_deliveries needs checkpoint metadata "
                "(player_registry, embed_seed) -- build this engine with "
                "WinProbabilityEngine.from_checkpoint()")
        from src.alt_transformer_data import build_embedding_lookup, build_features_for_innings
        if self._embed_lookup is None:
            self._embed_lookup = build_embedding_lookup(
                self.metadata["player_registry"], seed=self.metadata["embed_seed"])
        return build_features_for_innings(df, self._embed_lookup)

    def predict(self, features: np.ndarray) -> WinProbResult:
        T = features.shape[0]
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        wp_samples = np.zeros((self.mc_samples, T), dtype=np.float32)
        score_samples = np.zeros((self.mc_samples, T), dtype=np.float32)

        self.model.train()
        with torch.no_grad():
            for s in range(self.mc_samples):
                out = self.model(x)
                wp_samples[s] = out["win_prob"].squeeze().cpu().numpy()
                score_samples[s] = out["score_proj"].squeeze().cpu().numpy() * 250.0
        self.model.eval()

        mean_wp = wp_samples.mean(axis=0)
        std_wp = wp_samples.std(axis=0)

        return WinProbResult(
            ball_indices=np.arange(T),
            win_prob_mean=mean_wp,
            win_prob_std=std_wp,
            ci_lower=np.clip(mean_wp - 1.96 * std_wp, 0.0, 1.0),
            ci_upper=np.clip(mean_wp + 1.96 * std_wp, 0.0, 1.0),
            score_mean=score_samples.mean(axis=0),
            score_std=score_samples.std(axis=0),
            over_labels=_make_over_labels(T),
        )

    def predict_match(
        self,
        inn1_features: np.ndarray | None,
        inn2_features: np.ndarray | None,
        batting_team1: str = "",
        batting_team2: str = "",
    ) -> MatchWinProb:
        return MatchWinProb(
            innings1=self.predict(inn1_features) if inn1_features is not None else None,
            innings2=self.predict(inn2_features) if inn2_features is not None else None,
            batting_team1=batting_team1,
            batting_team2=batting_team2,
        )

    def update(self, feature_history: np.ndarray) -> tuple[float, float, float]:
        result = self.predict(feature_history)
        t = len(feature_history) - 1
        return float(result.win_prob_mean[t]), float(result.ci_lower[t]), float(result.ci_upper[t])


def _make_over_labels(T: int) -> list[str]:
    labels = []
    legal_ball = 0
    for _ in range(T):
        over = legal_ball // 6
        ball_in = legal_ball % 6 + 1
        labels.append(f"{over}.{ball_in}")
        legal_ball += 1
    return labels


def summarise_uncertainty(result: WinProbResult, every_n_overs: int = 5) -> None:
    checkpoints = list(range(0, len(result.ball_indices), every_n_overs * 6))
    checkpoints.append(len(result.ball_indices) - 1)

    print(f"\n{'Over':>6}  {'P(win)':>8}  {'CI 95%':>18}  {'sigma':>6}")
    print("-" * 46)
    for t in checkpoints:
        over = t // 6
        print(
            f"{over:>6}  "
            f"{result.win_prob_mean[t]:>8.3f}  "
            f"[{result.ci_lower[t]:.3f}, {result.ci_upper[t]:.3f}]  "
            f"{result.win_prob_std[t]:>6.4f}"
        )
