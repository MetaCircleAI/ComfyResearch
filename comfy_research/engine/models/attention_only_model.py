"""Multi-head self-attention block (no token embedding / LM head on the canvas class)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


class AttentionOnlyModel(nn.Module):
    """Self-attention on float activations ``[batch, seq, model_dim]`` → same shape."""

    def __init__(
        self,
        model_dim: int,
        context_length: int,
        num_heads: int = 1,
        *,
        causal: bool = True,
        qk_norm: bool = False,
        attn_temperature: float = 1.0,
        attn_logit_cap: float = 0.0,
        attn_dropout_p: float = 0.0,
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
        self.qk_norm = bool(qk_norm)
        self.attn_temperature = max(float(attn_temperature), 1e-6)
        self.attn_logit_cap = float(attn_logit_cap)
        self.attn_dropout_p = float(attn_dropout_p)
        d = self.model_dim
        self.w_q = nn.Linear(d, d, bias=True)
        self.w_k = nn.Linear(d, d, bias=True)
        self.w_v = nn.Linear(d, d, bias=True)
        self.w_o = nn.Linear(d, d, bias=True)

    def _attn_logits_and_probs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)
        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            att = q @ k.transpose(-2, -1)
        else:
            scale = float(hd) ** -0.5 / self.attn_temperature
            att = (q @ k.transpose(-2, -1)) * scale
        if self.attn_logit_cap > 0:
            c = float(self.attn_logit_cap)
            att = att.clamp(min=-c, max=c)
        if self.causal:
            causal = torch.triu(torch.ones(l, l, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal, float("-inf"))
        probs = torch.softmax(att, dim=-1)
        probs = F.dropout(probs, p=self.attn_dropout_p, training=self.training)
        mixed = probs @ v
        y = mixed.transpose(1, 2).reshape(b, l, d)
        return self.w_o(y), probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` float ``[batch, seq_len, model_dim]`` with ``seq_len == context_length``."""
        if x.dim() != 3:
            raise ValueError("x must be 3D [batch, seq_len, model_dim]")
        b, l, d = x.shape
        if l != self.context_length:
            raise ValueError(f"seq_len {l} != context_length {self.context_length}")
        if d != self.model_dim:
            raise ValueError(f"last dim {d} != model_dim {self.model_dim}")
        out, _ = self._attn_logits_and_probs(x)
        return out

    def self_attention_probs_all_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Single attention block → one softmax map ``[batch, heads, seq, seq]``."""
        return [self.self_attention_probs(x)]

    def self_attention_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax attention weights ``[batch, heads, seq, seq]`` (same masking as ``forward``)."""
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
        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            att = q @ k.transpose(-2, -1)
        else:
            scale = float(hd) ** -0.5 / self.attn_temperature
            att = (q @ k.transpose(-2, -1)) * scale
        if self.attn_logit_cap > 0:
            c = float(self.attn_logit_cap)
            att = att.clamp(min=-c, max=c)
        if self.causal:
            causal = torch.triu(torch.ones(l, l, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal, float("-inf"))
        return torch.softmax(att, dim=-1)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {
                "w_q": self.w_q.weight.detach().float().cpu().numpy(),
                "w_k": self.w_k.weight.detach().float().cpu().numpy(),
                "w_v": self.w_v.weight.detach().float().cpu().numpy(),
                "w_o": self.w_o.weight.detach().float().cpu().numpy(),
            }


class AttentionTokenPredictBundle(nn.Module):
    """Trainer-only: embed token ids → :class:`AttentionOnlyModel` → per-step vocab logits (CE)."""

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        context_length: int,
        num_heads: int,
        *,
        causal: bool = True,
        local_mixing_kernel: int = 0,
        qk_norm: bool = False,
        attn_temperature: float = 1.0,
        attn_logit_cap: float = 0.0,
        attn_dropout_p: float = 0.0,
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
        self.block = AttentionOnlyModel(
            model_dim,
            context_length,
            num_heads,
            causal=causal,
            qk_norm=qk_norm,
            attn_temperature=attn_temperature,
            attn_logit_cap=attn_logit_cap,
            attn_dropout_p=attn_dropout_p,
        )
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

    def self_attention_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``[batch, heads, seq, seq]`` softmax weights after embedding (+ optional local mix)."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != context_length {self.context_length}",
            )
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        return self.block.self_attention_probs(h)

    def self_attention_probs_all_layers(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Same path as :meth:`self_attention_probs`; returns a one-element list."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != context_length {self.context_length}",
            )
        h = self.embedding(token_ids.long())
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        return self.block.self_attention_probs_all_layers(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        out = dict(self.block.observable_numpy_arrays())
        with torch.no_grad():
            out["embedding"] = self.embedding.weight.detach().float().cpu().numpy()
            out["unembed"] = self.lm_head.weight.detach().float().cpu().numpy()
        return out
