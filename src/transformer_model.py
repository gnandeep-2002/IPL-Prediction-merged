from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GAME_STATE_DIM = 24
PLAYER_EMBED_DIM = 32
INPUT_DIM = GAME_STATE_DIM + 3 * PLAYER_EMBED_DIM

TRANSFORMER_D_MODEL = 64
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DIM_FF = 128
TRANSFORMER_DROPOUT = 0.15
MAX_SEQ_LEN = 280

NEXT_BALL_CLASSES = 7
NEXT_BALL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, "W": 6}

LAMBDA_WIN_PROB = 2.00
LAMBDA_NEXT_BALL = 0.0
LAMBDA_SCORE = 0.0


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class IPLTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        dim_ff: int = TRANSFORMER_DIM_FF,
        dropout: float = TRANSFORMER_DROPOUT,
        num_classes: int = NEXT_BALL_CLASSES,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        head_hidden = d_model // 2
        self.win_head = _mlp(d_model, head_hidden, 1, dropout)
        self.next_ball_head = _mlp(d_model, head_hidden, num_classes, dropout)
        self.score_head = _mlp(d_model, head_hidden, 1, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = x.shape

        h = self.input_proj(x)
        h = self.pos_enc(h)

        causal_mask = self._causal_mask(T, x.device)
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)

        win_prob = torch.sigmoid(self.win_head(h))
        next_ball = self.next_ball_head(h)
        score_proj = F.softplus(self.score_head(h))

        return {"win_prob": win_prob, "next_ball": next_ball, "score_proj": score_proj, "hidden": h}


class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        lambda_win: float = LAMBDA_WIN_PROB,
        lambda_next_ball: float = LAMBDA_NEXT_BALL,
        lambda_score: float = LAMBDA_SCORE,
    ):
        super().__init__()
        self.lw = lambda_win
        self.lnb = lambda_next_ball
        self.ls = lambda_score

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        win_labels: torch.Tensor,
        nb_labels: torch.Tensor,
        score_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid = valid_mask.reshape(-1)
        win_pred = preds["win_prob"].reshape(-1)[valid]
        win_tgt = win_labels.reshape(-1)[valid].float()

        loss_win = F.binary_cross_entropy(win_pred, win_tgt)
        total = self.lw * loss_win
        loss_nb = torch.tensor(0.0, device=win_pred.device)
        loss_sc = torch.tensor(0.0, device=win_pred.device)

        if self.lnb > 0:
            nb_pred = preds["next_ball"].reshape(-1, NEXT_BALL_CLASSES)[valid]
            nb_tgt = nb_labels.reshape(-1)[valid]
            loss_nb = F.cross_entropy(nb_pred, nb_tgt)
            total = total + self.lnb * loss_nb

        if self.ls > 0:
            sc_pred = preds["score_proj"].reshape(-1)[valid]
            sc_tgt = score_labels.reshape(-1)[valid].float()
            loss_sc = F.huber_loss(sc_pred, sc_tgt, delta=0.2)
            total = total + self.ls * loss_sc

        return {"loss": total, "loss_win": loss_win, "loss_nb": loss_nb, "loss_score": loss_sc}


class InningsDataset(torch.utils.data.Dataset):

    def __init__(self, innings_list: list[dict], max_len: int = MAX_SEQ_LEN):
        self.data = innings_list
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        feats = item["features"]
        T = min(len(feats), self.max_len)
        feats = feats[:T]

        win_lbl = item["win_label"]
        nb_lbl = item["nb_labels"][:T]
        sc_lbl = item["score_label"]

        pad = self.max_len - T
        feat_pad = torch.zeros(pad, feats.shape[1], dtype=torch.float32)
        nb_pad = torch.zeros(pad, dtype=torch.long)
        valid = torch.zeros(self.max_len, dtype=torch.bool)
        valid[:T] = True

        features_t = torch.cat([torch.tensor(feats, dtype=torch.float32), feat_pad], dim=0)
        nb_t = torch.cat([torch.tensor(nb_lbl, dtype=torch.long), nb_pad], dim=0)

        win_t = torch.full((self.max_len,), float(win_lbl))
        score_t = torch.full((self.max_len,), float(sc_lbl))

        return {
            "features": features_t,
            "win_labels": win_t,
            "nb_labels": nb_t,
            "score_labels": score_t,
            "valid_mask": valid,
        }


class PlayerEmbedTable(nn.Module):

    def __init__(self, num_players: int, embed_dim: int = PLAYER_EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(num_players + 1, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight.data[1:])

    def forward(self, player_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(player_ids)
