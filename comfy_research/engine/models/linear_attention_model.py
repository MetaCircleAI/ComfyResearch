"""Causal linear attention (ELU+1 feature map + cumulative-sum formulation)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class LinearAttentionOnlyModel(nn.Module):
    """Linear self-attention on ``[batch, seq, model_dim]`` (causal or full)."""

    def __init__(
        self,
        model_dim: int,
        context_length: int,
        num_heads: int = 1,
        *,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if model_dim < 1 or context_length < 1 or num_heads < 1:
            raise ValueError("model_dim, context_length, num_heads must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.context_length = int(context_length)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.causal = bool(causal)
        d = self.model_dim
        self.w_q = nn.Linear(d, d, bias=True)
        self.w_k = nn.Linear(d, d, bias=True)
        self.w_v = nn.Linear(d, d, bias=True)
        self.w_o = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("x must be 3D [batch, seq_len, model_dim]")
        b, l, d = x.shape
        if l != self.context_length:
            raise ValueError(f"seq_len {l} != context_length {self.context_length}")
        if d != self.model_dim:
            raise ValueError(f"last dim {d} != model_dim {self.model_dim}")
        h, hd = self.num_heads, self.head_dim
        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)
        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0
        if self.causal:
            kv = torch.cumsum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=2)
            num = torch.einsum("bhld,bhldm->bhlm", phi_q, kv)
            den = torch.einsum("bhld,bhld->bhl", phi_q, torch.cumsum(phi_k, dim=2)).clamp(min=1e-6)
            mixed = num / den.unsqueeze(-1)
        else:
            sum_kv = torch.sum(phi_k.unsqueeze(-1) * v.unsqueeze(-2), dim=2, keepdim=True)
            sum_k = torch.sum(phi_k, dim=2, keepdim=True).clamp(min=1e-6)
            num = torch.einsum("bhld,bh1dm->bhlm", phi_q, sum_kv)
            den = torch.einsum("bhld,bh1d->bhl", phi_q, sum_k).clamp(min=1e-6)
            mixed = num / den.unsqueeze(-1)
        y = mixed.transpose(1, 2).reshape(b, l, d)
        return self.w_o(y)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "w_q": self.w_q.weight.detach().float().cpu().numpy(),
                "w_k": self.w_k.weight.detach().float().cpu().numpy(),
                "w_v": self.w_v.weight.detach().float().cpu().numpy(),
                "w_o": self.w_o.weight.detach().float().cpu().numpy(),
            }


class LinearAttentionTokenPredictBundle(nn.Module):
    """Embed tokens → linear attention → per-step vocab logits."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        num_heads: int,
        *,
        causal: bool = True,
        local_mixing_kernel: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2 for token prediction")
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
        self.block = LinearAttentionOnlyModel(model_dim, context_length, num_heads, causal=causal)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != context_length {self.context_length}",
            )
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        h = self.block(h)
        return self.lm_head(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        out = dict(self.block.observable_numpy_arrays())
        with torch.no_grad():
            out["embedding"] = self.embedding.weight.detach().float().cpu().numpy()
            out["unembed"] = self.lm_head.weight.detach().float().cpu().numpy()
        return out
