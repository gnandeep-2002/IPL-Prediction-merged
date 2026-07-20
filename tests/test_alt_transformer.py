import numpy as np
import pytest
import torch

from src.transformer_model import IPLTransformer, MultiTaskLoss, InningsDataset, PlayerEmbedTable, INPUT_DIM
from src.game_state import GAME_STATE_DIM


def test_transformer_output_shapes():
    torch.manual_seed(0)
    model = IPLTransformer()
    x = torch.randn(4, 20, INPUT_DIM)
    out = model(x)
    assert out["win_prob"].shape == (4, 20, 1)
    assert out["next_ball"].shape == (4, 20, 7)
    assert out["score_proj"].shape == (4, 20, 1)


def test_win_prob_in_unit_interval():
    torch.manual_seed(0)
    model = IPLTransformer()
    x = torch.randn(2, 10, INPUT_DIM)
    out = model(x)
    p = out["win_prob"]
    assert (p >= 0).all() and (p <= 1).all()


def test_score_proj_non_negative():
    torch.manual_seed(0)
    model = IPLTransformer()
    x = torch.randn(2, 10, INPUT_DIM)
    out = model(x)
    assert (out["score_proj"] >= 0).all()


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    model = IPLTransformer()
    model.eval()
    x = torch.randn(1, 10, INPUT_DIM)
    with torch.no_grad():
        out1 = model(x)["win_prob"][0, 3, 0].item()
        x2 = x.clone()
        x2[0, 8:] = torch.randn_like(x2[0, 8:]) * 100
        out2 = model(x2)["win_prob"][0, 3, 0].item()
    assert abs(out1 - out2) < 1e-5, "Causal mask leaked future information into an earlier position"


def test_multitask_loss_win_only_ignores_aux_heads():
    torch.manual_seed(0)
    loss_fn = MultiTaskLoss(lambda_win=2.0, lambda_next_ball=0.0, lambda_score=0.0)
    preds = {
        "win_prob": torch.sigmoid(torch.randn(2, 5, 1)),
        "next_ball": torch.randn(2, 5, 7),
        "score_proj": torch.rand(2, 5, 1),
    }
    win_labels = torch.randint(0, 2, (2, 5)).float()
    nb_labels = torch.randint(0, 7, (2, 5))
    score_labels = torch.rand(2, 5)
    valid_mask = torch.ones(2, 5, dtype=torch.bool)

    losses = loss_fn(preds, win_labels, nb_labels, score_labels, valid_mask)
    assert losses["loss"].item() == pytest.approx(2.0 * losses["loss_win"].item())


def test_innings_dataset_padding():
    seqs = [{
        "features": np.random.randn(5, INPUT_DIM).astype(np.float32),
        "win_label": 1,
        "nb_labels": np.zeros(5, dtype=int),
        "score_label": 0.6,
    }]
    ds = InningsDataset(seqs, max_len=10)
    item = ds[0]
    assert item["features"].shape == (10, INPUT_DIM)
    assert item["valid_mask"].sum().item() == 5
    assert (item["features"][5:] == 0).all()


def test_player_embed_table_padding_idx_zero():
    table = PlayerEmbedTable(num_players=10, embed_dim=8)
    padding_vec = table(torch.tensor([0]))
    assert torch.allclose(padding_vec, torch.zeros(1, 8))


def test_game_state_matches_transformer_input_dim():
    from src.transformer_model import GAME_STATE_DIM as MODEL_GS_DIM
    assert GAME_STATE_DIM == MODEL_GS_DIM
