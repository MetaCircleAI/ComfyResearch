"""Causal depthwise conv + gated pointwise MLP (Hyena-flavored)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class HyenaLikeBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, ff_mult: int = 2) -> None:
        super().__init__()
        ks = int(kernel_size)
        if ks < 3 or ks % 2 == 0:
            ks = ks + (1 - ks % 2)
            if ks < 3:
                ks = 3
        self.ln = nn.LayerNorm(dim)
        self.conv = CausalDepthwiseConv1d(dim, ks)
        self.norm_conv = nn.LayerNorm(dim)
        hidden = max(1, int(dim * ff_mult))
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.ln(x)
        z = z + self.conv(z)
        z = self.norm_conv(z)
        g = torch.sigmoid(self.gate(z))
        z = g * self.ff(z)
        return residual + z


class HyenaLikeConvTokenPredictBundle(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        depth: int = 2,
        kernel_size: int = 7,
        ff_mult: int = 2,
        *,
        local_mixing_kernel: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        dep = max(1, int(depth))
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.context_length = int(context_length)
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        lk = int(local_mixing_kernel)
        if lk >= 3:
            if lk % 2 == 0:
                lk += 1
            self.local_mix: nn.Module | None = CausalDepthwiseConv1d(self.model_dim, lk)
        else:
            self.local_mix = None
        self.blocks = nn.ModuleList(
            [HyenaLikeBlock(model_dim, kernel_size, ff_mult=ff_mult) for _ in range(dep)]
        )
        self.out_ln = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError("seq_len mismatch")
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.lm_head(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "embedding": self.embedding.weight.detach().float().cpu().numpy(),
                "unembed": self.lm_head.weight.detach().float().cpu().numpy(),
            }
