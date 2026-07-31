"""Learned absolute positions and rotary (RoPE) embeddings for atomic layer chains."""

from __future__ import annotations

import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    """Adds a learnable ``[max_seq_len, dim]`` table broadcast along the sequence axis (dim ``-2`` when rank ≥ 3)."""

    def __init__(self, max_seq_len: int, embedding_dim: int) -> None:
        super().__init__()
        if max_seq_len < 1 or embedding_dim < 1:
            raise ValueError("max_seq_len and embedding_dim must be >= 1.")
        self.max_seq_len = int(max_seq_len)
        self.embedding_dim = int(embedding_dim)
        self.pe = nn.Parameter(torch.zeros(self.max_seq_len, self.embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.embedding_dim
        if x.size(-1) != d:
            raise RuntimeError(f"absolute_pos_embed_layer expected last dim {d}, got {x.size(-1)}.")
        if x.dim() == 2:
            return x + self.pe[0]
        if x.dim() < 2:
            raise RuntimeError("absolute_pos_embed_layer expects input rank >= 2.")
        t = x.size(-2)
        if t > self.max_seq_len:
            raise RuntimeError(
                f"absolute_pos_embed_layer sequence length {t} exceeds max_seq_len {self.max_seq_len}."
            )
        pe_b = self.pe[:t]
        for _ in range(x.dim() - 2):
            pe_b = pe_b.unsqueeze(0)
        return x + pe_b


class RotaryEmbedding(nn.Module):
    """Applies rotary position embeddings to the last dimension (must be even); sequence axis is ``-2`` (or length 1 when rank is 2)."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError("rotary_embed_layer rotary_dim must be an even integer >= 2.")
        self.dim = int(dim)
        self.base = float(base)
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.dim
        if x.size(-1) != d:
            raise RuntimeError(f"rotary_embed_layer expected last dim {d}, got {x.size(-1)}.")
        if x.dim() < 2:
            raise RuntimeError("rotary_embed_layer expects input rank >= 2.")
        xh = x.unsqueeze(-2) if x.dim() == 2 else x
        t = xh.size(-2)
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = pos[:, None] * self.inv_freq.to(device=x.device, dtype=pos.dtype)
        cos = freqs.cos().to(dtype=xh.dtype)
        sin = freqs.sin().to(dtype=xh.dtype)
        tail = (t, d // 2)
        br = (1,) * (xh.dim() - 2) + tail
        cos = cos.reshape(br)
        sin = sin.reshape(br)
        x1 = xh[..., 0::2]
        x2 = xh[..., 1::2]
        y = torch.empty_like(xh)
        y[..., 0::2] = x1 * cos - x2 * sin
        y[..., 1::2] = x1 * sin + x2 * cos
        return y.squeeze(-2) if x.dim() == 2 else y
