"""MPP-style spatiotemporal ViT-lite: patchify each frame, space-time Transformer, unpatchify.

Inspired by Multiple Physics Pretraining (McCabe et al., NeurIPS 2024; https://arxiv.org/abs/2310.02994)
and the reference AViT layout (PolymathicAI/multiple_physics_pretraining): shared normalization,
patch embedding, global attention over flattened space-time tokens, and pixel decoder.

Forward consumes flattened tensors ``[batch, T*C*H*W]`` matching ComfyResearch regression trainers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.transformer_encoder_custom import StableTransformerEncoder, StableTransformerEncoderLayer


def _scalar_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class MppSpatiotemporalModel(nn.Module):
    def __init__(
        self,
        *,
        context_frames: int,
        channels: int,
        grid_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ff_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_frames < 1 or channels < 1 or grid_size < 4:
            raise HTTPException(status_code=400, detail="MPP model needs contextFrames>=1, channels>=1, gridSize>=4.")
        if patch_size < 1 or grid_size % patch_size != 0:
            raise HTTPException(
                status_code=400,
                detail="MPP patch_size must divide grid_size evenly.",
            )
        if embed_dim % num_heads != 0:
            raise HTTPException(status_code=400, detail="MPP embed_dim must be divisible by num_heads.")

        self.context_frames = int(context_frames)
        self.channels = int(channels)
        self.grid_size = int(grid_size)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.flat_dim = self.context_frames * self.channels * self.grid_size * self.grid_size

        hp = self.grid_size // self.patch_size
        wp = hp
        self._hp = hp
        self._wp = wp
        self.tokens_per_frame = hp * wp
        self.seq_len = self.context_frames * self.tokens_per_frame

        self.patch_embed = nn.Conv2d(self.channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        dim_ff = max(4, int(round(float(ff_ratio) * embed_dim)))
        layer = StableTransformerEncoderLayer(
            int(embed_dim),
            int(num_heads),
            dim_ff,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            qk_norm=False,
            attn_temperature=1.0,
            attn_logit_cap=0.0,
            attn_dropout_p=0.0,
        )
        self.encoder = StableTransformerEncoder(layer, num_layers=max(1, int(depth)))
        patch_pixels = self.patch_size * self.patch_size * self.channels
        self.head = nn.Linear(embed_dim, patch_pixels)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.dim() != 2 or int(x_flat.shape[1]) != self.flat_dim:
            raise ValueError(
                f"MppSpatiotemporalModel expects [batch, {self.flat_dim}], got {tuple(x_flat.shape)}"
            )
        b = int(x_flat.shape[0])
        t = self.context_frames
        c = self.channels
        h = w = self.grid_size
        x = x_flat.reshape(b, t, c, h, w)

        # Per-batch, per-channel normalization over time + space (MPP-style shared scaling).
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True).clamp_min(1e-7)
        x_n = (x - mean) / std

        tokens: list[torch.Tensor] = []
        for ti in range(t):
            z = self.patch_embed(x_n[:, ti])
            z = z.flatten(2).transpose(1, 2)
            tokens.append(z)
        seq = torch.cat(tokens, dim=1)
        seq = seq + self.pos_embed
        enc = self.encoder(seq, is_causal=False)

        hp, wp, p = self._hp, self._wp, self.patch_size
        out_frames: list[torch.Tensor] = []
        patch_pixels = p * p * c
        for ti in range(t):
            sl = enc[:, ti * self.tokens_per_frame : (ti + 1) * self.tokens_per_frame, :]
            pix = self.head(sl)
            pix = pix.reshape(b, hp, wp, c, p, p)
            pix = pix.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(b, c, h, w)
            out_frames.append(pix)
        stacked = torch.stack(out_frames, dim=1)
        out = stacked * std + mean
        return out.reshape(b, self.flat_dim)


def mpp_spatiotemporal_from_canvas_md(md: dict[str, Any]) -> MppSpatiotemporalModel:
    ctx = max(1, _scalar_int(md.get("contextFrames"), 4))
    ch = max(1, _scalar_int(md.get("channels"), 1))
    g = max(4, _scalar_int(md.get("gridSize"), 16))
    patch = max(1, _scalar_int(md.get("patchSize"), 4))
    emb = max(32, _scalar_int(md.get("embedDim"), 128))
    depth = max(1, _scalar_int(md.get("depth"), 4))
    heads = max(1, _scalar_int(md.get("numHeads"), 4))
    ff_r = _scalar_float(md.get("ffRatio"), 4.0)
    drop = _scalar_float(md.get("dropout"), 0.0)

    expected_flat = ctx * ch * g * g
    in_d = _scalar_int(md.get("inputDim"), expected_flat)
    out_d = _scalar_int(md.get("outputDim"), expected_flat)
    if in_d != expected_flat or out_d != expected_flat:
        raise HTTPException(
            status_code=400,
            detail=(
                f"mpp_spatiotemporal_model flat I/O must be contextFrames*channels*gridSize^2 = {expected_flat}; "
                f"got inputDim={in_d}, outputDim={out_d}."
            ),
        )
    return MppSpatiotemporalModel(
        context_frames=ctx,
        channels=ch,
        grid_size=g,
        patch_size=patch,
        embed_dim=emb,
        depth=depth,
        num_heads=heads,
        ff_ratio=ff_r,
        dropout=drop,
    )
