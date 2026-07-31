"""Transformer encoder on token ids: embedding + positional bias + encoder + lm_head (per-step logits)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d
from comfy_research.engine.models.numeric_transformer_model import _coerce_nonneg_int, _coerce_yes_no
from comfy_research.engine.models.transformer_encoder_custom import (
    StableTransformerEncoder,
    StableTransformerEncoderLayer,
    apply_spectral_norm_to_encoder_linears,
    stable_attn_hyperparams_for_encoder_backend,
)

_TRANSFORMER_ENCODER_ACTIVATIONS_FFN = frozenset({"gelu", "relu", "silu"})
EncoderBackend = Literal["pytorch", "stable"]


def _coerce_encoder_activation(v: Any, *, default: str = "gelu") -> str:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v if v is not None else default).strip().lower()
    return s if s in _TRANSFORMER_ENCODER_ACTIVATIONS_FFN else default


def _coerce_encoder_backend(v: Any) -> EncoderBackend:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v if v is not None else "pytorch").strip().lower()
    return "stable" if s == "stable" else "pytorch"


def token_transformer_from_canvas_md(md: dict[str, Any]) -> TokenTransformerModel:
    activation = _coerce_encoder_activation(md.get("activation"))
    backend = _coerce_encoder_backend(md.get("encoderBackend"))
    enc_do = float(_scalar_float(md.get("encoderDropout"), 0.0))
    enc_do = max(0.0, min(enc_do, 1.0))
    spectral = _coerce_yes_no(md.get("spectralNormLinears"), default=False)
    lm_scale = float(_scalar_float(md.get("lmLogitScale"), 1.0))
    stable_qk = _coerce_yes_no(md.get("stableQkNorm"), default=False)
    stable_temp = float(_scalar_float(md.get("stableAttnTemperature"), 1.0))
    stable_cap = float(_scalar_float(md.get("stableAttnLogitCap"), 0.0))
    stable_ad = float(_scalar_float(md.get("stableAttnDropout"), 0.0))
    stable_ad = max(0.0, min(stable_ad, 1.0))
    return TokenTransformerModel(
        _coerce_nonneg_int(md.get("vocabSize"), 100),
        _coerce_nonneg_int(md.get("contextLength"), 4),
        _coerce_nonneg_int(md.get("modelDim"), 32),
        _coerce_nonneg_int(md.get("numHeads"), 1),
        _coerce_nonneg_int(md.get("numLayers"), 1),
        _coerce_nonneg_int(md.get("ffDim"), 64),
        tie_embedding_lm_head=_coerce_yes_no(md.get("tieEmbeddingLmHead"), default=True),
        causal=_coerce_yes_no(md.get("causalAttention"), default=True),
        local_mixing_kernel=_coerce_nonneg_int(md.get("localMixingKernel"), 0),
        activation=activation,
        encoder_backend=backend,
        encoder_dropout=enc_do,
        spectral_norm_linears=spectral,
        lm_logit_scale=lm_scale,
        stable_qk_norm=stable_qk,
        stable_attn_temperature=max(stable_temp, 1e-6),
        stable_attn_logit_cap=max(stable_cap, 0.0),
        stable_attn_dropout_p=stable_ad,
    )


def _scalar_float(v: Any, default: float) -> float:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class TokenTransformerModel(nn.Module):
    """Long token ids ``[batch, T]`` with ``T == context_length`` → logits ``[batch, T, vocab_size]``."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        *,
        tie_embedding_lm_head: bool = True,
        causal: bool = True,
        local_mixing_kernel: int = 0,
        activation: str = "gelu",
        encoder_backend: EncoderBackend = "pytorch",
        encoder_dropout: float = 0.0,
        spectral_norm_linears: bool = False,
        lm_logit_scale: float = 1.0,
        stable_qk_norm: bool = False,
        stable_attn_temperature: float = 1.0,
        stable_attn_logit_cap: float = 0.0,
        stable_attn_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if context_length < 1 or vocab_size < 2 or model_dim < 1:
            raise ValueError("context_length must be >= 1, vocab_size >= 2, model_dim >= 1")
        if num_heads < 1 or num_layers < 1 or ff_dim < 1:
            raise ValueError("num_heads, num_layers, and ff_dim must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.context_length = int(context_length)
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.causal = bool(causal)
        self.tie_embedding_lm_head = bool(tie_embedding_lm_head)
        self.encoder_backend = encoder_backend
        self.lm_logit_scale = float(lm_logit_scale)

        lk = int(local_mixing_kernel)
        if lk >= 3:
            if lk % 2 == 0:
                lk += 1
            self.local_mix: nn.Module | None = CausalDepthwiseConv1d(self.model_dim, lk)
        else:
            self.local_mix = None
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.context_length, self.model_dim))

        act_str = _coerce_encoder_activation(activation)
        ed = max(0.0, min(float(encoder_dropout), 1.0))
        qk, atemp, acap, adrop = stable_attn_hyperparams_for_encoder_backend(
            encoder_backend,
            stable_qk_norm=bool(stable_qk_norm),
            stable_attn_temperature=float(stable_attn_temperature),
            stable_attn_logit_cap=float(stable_attn_logit_cap),
            stable_attn_dropout_p=float(stable_attn_dropout_p),
        )
        layer = StableTransformerEncoderLayer(
            self.model_dim,
            int(num_heads),
            int(ff_dim),
            dropout=ed,
            activation=act_str,
            batch_first=True,
            qk_norm=qk,
            attn_temperature=atemp,
            attn_logit_cap=acap,
            attn_dropout_p=adrop,
        )
        self.encoder = StableTransformerEncoder(layer, int(num_layers))

        if spectral_norm_linears:
            apply_spectral_norm_to_encoder_linears(self.encoder)

        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=not self.tie_embedding_lm_head)
        if self.tie_embedding_lm_head:
            self.embedding.weight = self.lm_head.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be 2D [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != model context length {self.context_length}",
            )
        x = self.embedding(token_ids.long()) + self.pos_embed.unsqueeze(0)
        if self.local_mix is not None:
            x = x + self.local_mix(x)
        cast_encoder = self.encoder
        assert isinstance(cast_encoder, StableTransformerEncoder)
        h = cast_encoder(x, is_causal=bool(self.causal))
        logits = self.lm_head(h)
        if self.lm_logit_scale != 1.0:
            logits = logits * self.lm_logit_scale
        return logits

    def self_attention_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Last encoder layer softmax weights ``[batch, heads, seq, seq]`` (same preprocessing as ``forward``)."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be 2D [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != model context length {self.context_length}",
            )
        x = self.embedding(token_ids.long()) + self.pos_embed.unsqueeze(0)
        if self.local_mix is not None:
            x = x + self.local_mix(x)
        cast_encoder = self.encoder
        assert isinstance(cast_encoder, StableTransformerEncoder)
        return cast_encoder.last_layer_attention_probs(x, is_causal=self.causal)

    def self_attention_probs_all_layers(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Per encoder layer: softmax weights ``[batch, heads, seq, seq]`` (same preprocessing as ``forward``)."""
        if token_ids.dim() != 2:
            raise ValueError("token_ids must be 2D [batch, seq_len]")
        if int(token_ids.shape[1]) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(token_ids.shape[1])} != model context length {self.context_length}",
            )
        x = self.embedding(token_ids.long()) + self.pos_embed.unsqueeze(0)
        if self.local_mix is not None:
            x = x + self.local_mix(x)
        cast_encoder = self.encoder
        assert isinstance(cast_encoder, StableTransformerEncoder)
        return cast_encoder.all_layers_attention_probs(x, is_causal=self.causal)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            emb = self.embedding.weight.detach().float().cpu().numpy()
            w = self.lm_head.weight.detach().float().cpu().numpy()
            return {"embedding": emb, "lm_head": w}
