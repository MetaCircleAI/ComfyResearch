"""Diagonal input-dependent SSM over sequence (pure PyTorch recurrence)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class DiagonalSsmCore(nn.Module):
    """``h_{t+1} = exp(A_t) ⊙ h_t + B_t ⊙ x_t`` with A_t, B_t from linear projections of x_t."""

    def __init__(self, model_dim: int, context_length: int) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.context_length = int(context_length)
        d = self.model_dim
        self.proj_a = nn.Linear(d, d, bias=True)
        self.proj_b = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be [batch, seq, dim]")
        b, l, d = x.shape
        if l != self.context_length or d != self.model_dim:
            raise ValueError("shape mismatch")
        h = torch.zeros(b, d, device=x.device, dtype=x.dtype)
        outs: list[torch.Tensor] = []
        for t in range(l):
            xt = x[:, t, :]
            a_t = -F.softplus(self.proj_a(xt))
            b_t = self.proj_b(xt)
            h = torch.exp(a_t) * h + b_t * xt
            outs.append(self.out_proj(h))
        return torch.stack(outs, dim=1)


class DiagonalSsmTokenPredictBundle(nn.Module):
    """Stacked diagonal SSM layers + token LM head."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        num_layers: int = 1,
        *,
        local_mixing_kernel: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        nl = max(1, int(num_layers))
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
        self.layers = nn.ModuleList(
            [DiagonalSsmCore(model_dim, context_length) for _ in range(nl)]
        )
        self.ln = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError("seq_len mismatch")
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        for layer in self.layers:
            h = h + layer(h)
        h = self.ln(h)
        return self.lm_head(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "embedding": self.embedding.weight.detach().float().cpu().numpy(),
                "unembed": self.lm_head.weight.detach().float().cpu().numpy(),
            }