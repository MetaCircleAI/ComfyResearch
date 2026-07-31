"""DETR-style 1D curve breakpoint detector."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class CurveBackbone(nn.Module):
    """1D CNN backbone: (B, C, L) -> (B, d_model, L')."""

    def __init__(self, in_channels: int = 3, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model // 2, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class CurveDETR(nn.Module):
    """
    Set-prediction model for 1D curve partition breakpoints.

    Each query predicts a normalized breakpoint position in [0, 1] plus an
    objectness score. Mechanism types are assigned later via finetize (LMN).
    """

    def __init__(
        self,
        num_queries: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.in_channels = in_channels

        self.backbone = CurveBackbone(in_channels=in_channels, d_model=d_model)
        self.pos_encoder = PositionalEncoding1D(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.bp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, curves: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            curves: (B, C, L) multi-channel curve features

        Returns:
            pred_breakpoints: (B, Q) sigmoid-normalized t in [0, 1]
            pred_scores: (B, Q) objectness in [0, 1]
        """
        b = curves.size(0)
        feat = self.backbone(curves)
        memory = feat.permute(2, 0, 1)
        memory = self.pos_encoder(memory)
        memory = self.encoder(memory)

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)
        hs = self.decoder(queries, memory)
        hs = hs.transpose(0, 1)

        pred_breakpoints = self.bp_head(hs).sigmoid().squeeze(-1)
        pred_scores = self.score_head(hs).sigmoid().squeeze(-1)
        return {"pred_breakpoints": pred_breakpoints, "pred_scores": pred_scores}
