"""
Training loop for the alternative IPLTransformer win-probability model
(src/transformer_model.py) on project_gagan's data.

Ported/adapted from project_hrishav/train.py. Kept intentionally small
in scope: this is the SECONDARY/alternative model per the approved
decision, not the default win-probability path (see
src/transformer_model.py's module docstring for why -- hrishav's own
evidence shows it doesn't beat the calibrated LogReg baseline).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from src.transformer_model import IPLTransformer, MultiTaskLoss, InningsDataset, MAX_SEQ_LEN

SEED = 42


def _final_ball_predictions(model: IPLTransformer, dataset: InningsDataset, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Win-probability prediction at the last real ball of each innings."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            valid = batch["valid_mask"]
            preds = model(x)["win_prob"].squeeze(-1).cpu().numpy()  # (B, T)
            for i in range(x.shape[0]):
                last_idx = int(valid[i].sum().item()) - 1
                if last_idx < 0:
                    continue
                y_pred.append(preds[i, last_idx])
                y_true.append(batch["win_labels"][i, 0].item())
    return np.array(y_true), np.array(y_pred)


def train_alt_transformer(
    train_seqs: list[dict], val_seqs: list[dict], test_seqs: list[dict],
    epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
    lambda_next_ball: float = 0.0, lambda_score: float = 0.0,
    device: str = "cpu",
) -> dict:
    """
    Train IPLTransformer win-only (default) or multi-task (if lambdas > 0)
    on the given train/val/test innings sequences. Returns the trained
    model plus test-set Brier/AUC at the final ball of each innings.
    """
    torch.manual_seed(SEED)

    train_ds = InningsDataset(train_seqs, max_len=MAX_SEQ_LEN)
    val_ds = InningsDataset(val_seqs, max_len=MAX_SEQ_LEN)
    test_ds = InningsDataset(test_seqs, max_len=MAX_SEQ_LEN)

    model = IPLTransformer().to(device)
    loss_fn = MultiTaskLoss(lambda_next_ball=lambda_next_ball, lambda_score=lambda_score)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_brier = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["features"].to(device)
            win_labels = batch["win_labels"].to(device)
            nb_labels = batch["nb_labels"].to(device)
            score_labels = batch["score_labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            optimizer.zero_grad()
            preds = model(x)
            losses = loss_fn(preds, win_labels, nb_labels, score_labels, valid_mask)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        y_val, p_val = _final_ball_predictions(model, val_ds, device)
        if len(y_val) > 0:
            val_brier = brier_score_loss(y_val, p_val)
            if val_brier < best_val_brier:
                best_val_brier = val_brier
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = _final_ball_predictions(model, test_ds, device)
    result = {"model": model, "best_val_brier": best_val_brier}
    if len(y_test) > 0 and len(np.unique(y_test)) > 1:
        result["test_brier"] = brier_score_loss(y_test, p_test)
        result["test_auc"] = roc_auc_score(y_test, p_test)
        result["test_n"] = len(y_test)
    return result
