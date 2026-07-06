"""
MC-Dropout win-probability engine: wraps a trained IPLTransformer and
produces a per-ball win-probability trajectory with an uncertainty band.

Ported near-verbatim from project_hrishav/win_probability.py -- schema-
agnostic (operates on precomputed (T, INPUT_DIM) feature matrices from
src/alt_transformer_data.py / src/game_state.py), so no adaptation was
needed beyond the import path and MC_SAMPLES constant (was
config.MC_SAMPLES in the original).

Key concept -- Monte Carlo Dropout
-----------------------------------
During training, Dropout randomly zeroes activations. At inference,
PyTorch normally disables dropout (model.eval()). MC Dropout deliberately
keeps dropout ACTIVE (model.train()) and runs many forward passes. The
variance across passes is a principled measure of epistemic (model)
uncertainty.

  mean = E[p(win)]          <- best estimate
  std  = Std[p(win)]        <- uncertainty
  CI95 = mean +/- 1.96*std  <- 95% confidence interval
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.transformer_model import IPLTransformer

MC_SAMPLES = 50


@dataclass
class WinProbResult:
    """Holds the full per-ball win-probability trajectory for one innings."""
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
    """Win-probability results for both innings of a match."""
    innings1: WinProbResult | None
    innings2: WinProbResult | None
    batting_team1: str
    batting_team2: str


class WinProbabilityEngine:
    """
    Parameters
    ----------
    model      : trained IPLTransformer (weights loaded, on correct device)
    mc_samples : number of stochastic forward passes
    device     : torch device string
    """

    def __init__(self, model: IPLTransformer, mc_samples: int = MC_SAMPLES, device: str = "cpu"):
        self.model = model.to(device)
        self.mc_samples = mc_samples
        self.device = device

    def predict(self, features: np.ndarray) -> WinProbResult:
        """
        Run MC Dropout inference on a single innings feature sequence.

        Parameters
        ----------
        features : (T, INPUT_DIM) float32 array, one row per delivery
        """
        T = features.shape[0]
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)

        wp_samples = np.zeros((self.mc_samples, T), dtype=np.float32)
        score_samples = np.zeros((self.mc_samples, T), dtype=np.float32)

        self.model.train()  # keep dropout active -- this is the MC Dropout trick
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
        """Predict win-probability curves for both innings of a match."""
        return MatchWinProb(
            innings1=self.predict(inn1_features) if inn1_features is not None else None,
            innings2=self.predict(inn2_features) if inn2_features is not None else None,
            batting_team1=batting_team1,
            batting_team2=batting_team2,
        )

    def update(self, feature_history: np.ndarray) -> tuple[float, float, float]:
        """
        Given the feature history so far, return the LATEST win-probability
        estimate (mean, lower, upper). For a live-dashboard loop.
        """
        result = self.predict(feature_history)
        t = len(feature_history) - 1
        return float(result.win_prob_mean[t]), float(result.ci_lower[t]), float(result.ci_upper[t])


def _make_over_labels(T: int) -> list[str]:
    """Generate ball labels like '0.1', '0.2', ..., '19.6' for plotting."""
    labels = []
    legal_ball = 0
    for _ in range(T):
        over = legal_ball // 6
        ball_in = legal_ball % 6 + 1
        labels.append(f"{over}.{ball_in}")
        legal_ball += 1
    return labels


def summarise_uncertainty(result: WinProbResult, every_n_overs: int = 5) -> None:
    """Print a compact table of win-probability at key overs."""
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
