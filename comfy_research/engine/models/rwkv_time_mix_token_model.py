"""Time-mix + channel-mix blocks (RWKV-flavored gated recurrence, pure PyTorch)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class RwkvLiteBlock(nn.Module):
    """Per-step gated recurrence (mix past state with current projection) + gated FF."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.w_k = nn.Linear(dim, dim, bias=True)
        self.w_v = nn.Linear(dim, dim, bias=True)
        self.w_r = nn.Linear(dim, dim, bias=True)
        self.decay = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.ff_gate = nn.Linear(dim, dim * 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        residual = x
        z = self.ln1(x)
        b, t, d = z.shape
        h = torch.zeros(b, d, device=z.device, dtype=z.dtype)
        outs: list[torch.Tensor] = []
        for i in range(t):
            zi = z[:, i, :]
            kk = torch.tanh(self.w_k(zi))
            vv = self.w_v(zi)
            rr = torch.sigmoid(self.w_r(zi))
            dec = torch.sigmoid(self.decay(zi))
            payload = kk * vv
            h = dec * h + (1.0 - dec) * payload
            outs.append(rr * h + (1.0 - rr) * zi)
        tm = torch.stack(outs, dim=1)
        x = residual + self.o_proj(tm)
        residual = x
        z2 = self.ln2(x)
        g, up = self.ff_gate(z2).chunk(2, dim=-1)
        x = residual + torch.sigmoid(g) * F.gelu(up)
        return x


class RwkvTimeMixTokenPredictBundle(nn.Module):
    """Embedding → stacked RwkvLiteBlock → per-step logits."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        depth: int = 2,
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
        self.blocks = nn.ModuleList([RwkvLiteBlock(model_dim) for _ in range(dep)])
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
        return self.lm_head(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "embedding": self.embedding.weight.detach().float().cpu().numpy(),
                "unembed": self.lm_head.weight.detach().float().cpu().numpy(),
            }