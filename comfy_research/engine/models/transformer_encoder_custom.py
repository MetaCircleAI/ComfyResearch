"""Pre-LN Transformer encoder stack (GPT / nanoGPT-style: LayerNorm → attention → residual; LayerNorm → MLP → residual).

Uses explicit multi-head attention and two-layer FFN (:class:`StableMultiheadSelfAttention` + linear blocks), not
``torch.nn.TransformerEncoderLayer``. The canvas **Pre-LN encoder (default)** backend uses this stack with vanilla
attention; **Pre-LN + stability knobs** forwards the same modules with optional QK L2 norm, temperature, logit cap,
and attention dropout on probabilities.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_attn_hyperparams_for_encoder_backend(
    encoder_backend: str,
    *,
    stable_qk_norm: bool = False,
    stable_attn_temperature: float = 1.0,
    stable_attn_logit_cap: float = 0.0,
    stable_attn_dropout_p: float = 0.0,
) -> tuple[bool, float, float, float]:
    """Map canvas ``encoderBackend`` to :class:`StableMultiheadSelfAttention` hyperparameters.

    ``pytorch`` (historical id) selects a plain Pre-LN block; ``stable`` exposes the extra stability controls from the UI.
    """
    eb = str(encoder_backend).strip().lower()
    if eb == "stable":
        return (
            bool(stable_qk_norm),
            max(float(stable_attn_temperature), 1e-6),
            max(float(stable_attn_logit_cap), 0.0),
            max(0.0, min(float(stable_attn_dropout_p), 1.0)),
        )
    return (False, 1.0, 0.0, 0.0)


def apply_spectral_norm_to_encoder_linears(encoder: nn.Module) -> None:
    """Apply ``spectral_norm`` to each ``nn.Linear`` under ``encoder`` (encoder stack only)."""
    for child in encoder.modules():
        if isinstance(child, nn.Linear):
            if getattr(child, "parametrizations", None) is not None:
                continue
            try:
                nn.utils.spectral_norm(child, name="weight")
            except ValueError:
                pass


class StableMultiheadSelfAttention(nn.Module):
    """Batch-first MHA: optional QK L2 norm, logit cap, temperature, dropout on attention probs."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qk_norm: bool = False,
        attn_temperature: float = 1.0,
        attn_logit_cap: float = 0.0,
        attn_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.qk_norm = bool(qk_norm)
        self.attn_temperature = max(float(attn_temperature), 1e-6)
        self.attn_logit_cap = float(attn_logit_cap)
        self.attn_dropout_p = float(attn_dropout_p)
        self.w_q = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.w_k = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.w_v = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.w_o = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        v = self.w_v(x).view(b, l, h, hd).transpose(1, 2)
        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            scores = q @ k.transpose(-2, -1)
        else:
            scale = float(hd) ** -0.5 / self.attn_temperature
            scores = (q @ k.transpose(-2, -1)) * scale
        if self.attn_logit_cap > 0:
            c = float(self.attn_logit_cap)
            scores = scores.clamp(min=-c, max=c)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
        mixed = attn @ v
        y = mixed.transpose(1, 2).reshape(b, l, d)
        return self.w_o(y)

    def attention_probs(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Softmax weights ``[batch, heads, seq, seq]`` (same masking as ``forward``; no attention dropout)."""
        b, l, d = x.shape
        h, hd = self.num_heads, self.head_dim
        q = self.w_q(x).view(b, l, h, hd).transpose(1, 2)
        k = self.w_k(x).view(b, l, h, hd).transpose(1, 2)
        if self.qk_norm:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)
            scores = q @ k.transpose(-2, -1)
        else:
            scale = float(hd) ** -0.5 / self.attn_temperature
            scores = (q @ k.transpose(-2, -1)) * scale
        if self.attn_logit_cap > 0:
            c = float(self.attn_logit_cap)
            scores = scores.clamp(min=-c, max=c)
        if attn_mask is not None:
            scores = scores + attn_mask
        return torch.softmax(scores, dim=-1)


class StableTransformerEncoderLayer(nn.Module):
    """Pre-LN: Norm → self-attn → residual; Norm → FFN → residual."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | nn.Module = "gelu",
        *,
        batch_first: bool = True,
        qk_norm: bool = False,
        attn_temperature: float = 1.0,
        attn_logit_cap: float = 0.0,
        attn_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("StableTransformerEncoderLayer only supports batch_first=True")
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = StableMultiheadSelfAttention(
            d_model,
            nhead,
            qk_norm=qk_norm,
            attn_temperature=attn_temperature,
            attn_logit_cap=attn_logit_cap,
            attn_dropout_p=attn_dropout_p,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        if isinstance(activation, nn.Module):
            self.act: nn.Module = activation
        else:
            act_name = str(activation).strip().lower()
            if act_name == "relu":
                self.act = nn.ReLU()
            elif act_name == "silu":
                self.act = nn.SiLU()
            else:
                self.act = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x), attn_mask))
        u = self.norm2(x)
        h = self.linear2(self.dropout_ff(self.act(self.linear1(u))))
        x = x + self.dropout2(h)
        return x


class StableTransformerEncoder(nn.Module):
    """Stack of :class:`StableTransformerEncoderLayer` (batch-first)."""

    def __init__(self, layer: StableTransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        nl = int(num_layers)
        if nl < 1:
            raise ValueError("num_layers must be >= 1")
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(nl)])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_mask = mask
        if is_causal and attn_mask is None:
            lseq = int(x.size(1))
            causal_bool = torch.triu(torch.ones(lseq, lseq, device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = torch.zeros(lseq, lseq, device=x.device, dtype=x.dtype).masked_fill(causal_bool, float("-inf"))
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.view(1, 1, attn_mask.size(0), attn_mask.size(1))
        h = x
        for layer in self.layers:
            h = layer(h, attn_mask)
        return h

    def last_layer_attention_probs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Softmax self-attention at the **final** encoder layer, shape ``[batch, heads, seq, seq]``."""
        attn_mask = mask
        if is_causal and attn_mask is None:
            lseq = int(x.size(1))
            causal_bool = torch.triu(torch.ones(lseq, lseq, device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = torch.zeros(lseq, lseq, device=x.device, dtype=x.dtype).masked_fill(causal_bool, float("-inf"))
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.view(1, 1, attn_mask.size(0), attn_mask.size(1))
        h = x
        for layer in self.layers[:-1]:
            h = layer(h, attn_mask)
        last = self.layers[-1]
        return last.self_attn.attention_probs(last.norm1(h), attn_mask)

    def all_layers_attention_probs(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> list[torch.Tensor]:
        """Softmax self-attention at **each** encoder layer, each ``[batch, heads, seq, seq]``."""
        attn_mask = mask
        if is_causal and attn_mask is None:
            lseq = int(x.size(1))
            causal_bool = torch.triu(torch.ones(lseq, lseq, device=x.device, dtype=torch.bool), diagonal=1)
            attn_mask = torch.zeros(lseq, lseq, device=x.device, dtype=x.dtype).masked_fill(causal_bool, float("-inf"))
        if attn_mask is not None and attn_mask.dim() == 2:
            attn_mask = attn_mask.view(1, 1, attn_mask.size(0), attn_mask.size(1))
        h = x
        probs: list[torch.Tensor] = []
        for layer in self.layers:
            probs.append(layer.self_attn.attention_probs(layer.norm1(h), attn_mask))
            h = layer(h, attn_mask)
        return probs
