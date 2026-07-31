"""Slot attention over token sequence → pooled representation → logits."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class SlotAttention(nn.Module):
    """Iterative attention from slots to input tokens (dot-product softmax)."""

    def __init__(self, dim: int, num_slots: int, iters: int = 3, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.iters = max(1, int(iters))
        self.eps = float(eps)
        self.scale = dim**-0.5
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_input = nn.LayerNorm(dim)
        self.slots_mu = nn.Parameter(torch.randn(1, self.num_slots, dim) * 0.1)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, L, D]
        b, _, d = inputs.shape
        slots = self.slots_mu.expand(b, -1, -1).contiguous()
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            attn = torch.einsum("bqd,bkd->bqk", q, k) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = attn / (torch.sum(attn, dim=-2, keepdim=True) + self.eps)
            updates = torch.einsum("bqk,bkd->bqd", attn, v)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d)).reshape(b, self.num_slots, d)
            slots = slots + self.mlp(slots)
        return slots


class SlotAttentionTokenPredictBundle(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        num_slots: int = 4,
        slot_iters: int = 3,
        *,
        local_mixing_kernel: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
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
        self.slot_attn = SlotAttention(model_dim, num_slots=num_slots, iters=slot_iters)
        self.post = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU())
        self.lm_head = nn.Linear(model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError("seq_len mismatch")
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        slots = self.slot_attn(h)
        pooled = slots.mean(dim=1)
        logits = self.lm_head(self.post(pooled))
        return logits

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "embedding": self.embedding.weight.detach().float().cpu().numpy(),
                "unembed": self.lm_head.weight.detach().float().cpu().numpy(),
            }