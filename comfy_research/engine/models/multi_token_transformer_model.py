"""Transformer encoder on ``[batch, L, K]`` token ids: fuse K embeddings per timestep, predict K vocab heads at last timestep."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from comfy_research.engine.models.numeric_transformer_model import (
    _coerce_encoder_activation,
    _coerce_encoder_backend,
    _coerce_nonneg_int,
    _coerce_yes_no,
    _scalar_float,
)
from comfy_research.engine.models.transformer_encoder_custom import (
    StableTransformerEncoder,
    StableTransformerEncoderLayer,
    apply_spectral_norm_to_encoder_linears,
    stable_attn_hyperparams_for_encoder_backend,
)

EncoderBackend = Literal["pytorch", "stable"]


def multi_token_transformer_from_canvas_md(md: dict[str, Any]) -> MultiTokenTransformerModel:
    k = max(1, _coerce_nonneg_int(md.get("tokensPerPosition"), 2))
    tie = _coerce_yes_no(md.get("tieEmbeddingLmHead"), default=True)
    act = _coerce_encoder_activation(md.get("activation"))
    backend = _coerce_encoder_backend(md.get("encoderBackend"))
    ed = max(0.0, min(_scalar_float(md.get("encoderDropout"), 0.0), 1.0))
    spectral = _coerce_yes_no(md.get("spectralNormLinears"), default=False)
    lm_scale = float(_scalar_float(md.get("lmLogitScale"), 1.0))
    sqk = _coerce_yes_no(md.get("stableQkNorm"), default=False)
    stemp = max(_scalar_float(md.get("stableAttnTemperature"), 1.0), 1e-6)
    scap = max(_scalar_float(md.get("stableAttnLogitCap"), 0.0), 0.0)
    sad = max(0.0, min(_scalar_float(md.get("stableAttnDropout"), 0.0), 1.0))
    return MultiTokenTransformerModel(
        _coerce_nonneg_int(md.get("vocabSize"), 100),
        _coerce_nonneg_int(md.get("contextLength"), 4),
        k,
        _coerce_nonneg_int(md.get("modelDim"), 32),
        _coerce_nonneg_int(md.get("numHeads"), 1),
        _coerce_nonneg_int(md.get("numLayers"), 1),
        _coerce_nonneg_int(md.get("ffDim"), 64),
        tie_embedding_lm_head=tie,
        causal=_coerce_yes_no(md.get("causalAttention"), default=True),
        activation=act,
        encoder_backend=backend,
        encoder_dropout=ed,
        spectral_norm_linears=spectral,
        lm_logit_scale=lm_scale,
        stable_qk_norm=sqk,
        stable_attn_temperature=stemp,
        stable_attn_logit_cap=scap,
        stable_attn_dropout_p=sad,
    )


class MultiTokenTransformerModel(nn.Module):
    """Token ids ``[batch, L, K]`` with ``L == context_length`` → logits ``[batch, K, vocab_size]`` (last timestep)."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        tokens_per_position: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        *,
        tie_embedding_lm_head: bool = False,
        causal: bool = True,
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
        k = int(tokens_per_position)
        if context_length < 1 or vocab_size < 2 or model_dim < 1 or k < 1:
            raise ValueError(
                "context_length must be >= 1, vocab_size >= 2, model_dim >= 1, tokens_per_position >= 1",
            )
        if num_heads < 1 or num_layers < 1 or ff_dim < 1:
            raise ValueError("num_heads, num_layers, and ff_dim must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.context_length = int(context_length)
        self.vocab_size = int(vocab_size)
        self.tokens_per_position = int(k)
        self.model_dim = int(model_dim)
        self.causal = bool(causal)
        self.tie_embedding_lm_head = bool(tie_embedding_lm_head)
        self.encoder_backend = encoder_backend
        self.lm_logit_scale = float(lm_logit_scale)

        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        fused_in = k * self.model_dim
        self.token_fuse = nn.Linear(fused_in, self.model_dim)
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
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(self.model_dim, self.vocab_size, bias=not self.tie_embedding_lm_head)
                for _ in range(self.tokens_per_position)
            ],
        )
        if self.tie_embedding_lm_head:
            ref = self.lm_heads[0].weight
            self.embedding.weight = ref
            for head in self.lm_heads[1:]:
                head.weight = ref

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 3:
            raise ValueError("token_ids must be 3D [batch, seq_len, tokens_per_position]")
        b, t, k_in = token_ids.shape
        if int(k_in) != self.tokens_per_position:
            raise ValueError(
                f"token_ids last dim {int(k_in)} != model tokens_per_position {self.tokens_per_position}",
            )
        if int(t) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(t)} != model context_length {self.context_length}",
            )
        e = self.embedding(token_ids.long())
        fused = self.token_fuse(e.reshape(b, t, -1))
        x = fused + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        h = cast_enc(x, is_causal=bool(self.causal))
        h_last = h[:, -1, :]
        parts = [head(h_last) for head in self.lm_heads]
        out = torch.stack(parts, dim=1)
        if self.lm_logit_scale != 1.0:
            out = out * self.lm_logit_scale
        return out

    def self_attention_probs(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Last encoder layer softmax weights ``[batch, heads, seq, seq]`` after fused token embeddings."""
        if token_ids.dim() != 3:
            raise ValueError("token_ids must be 3D [batch, seq_len, tokens_per_position]")
        b, t, k_in = token_ids.shape
        if int(k_in) != self.tokens_per_position:
            raise ValueError(
                f"token_ids last dim {int(k_in)} != model tokens_per_position {self.tokens_per_position}",
            )
        if int(t) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(t)} != model context length {self.context_length}",
            )
        e = self.embedding(token_ids.long())
        fused = self.token_fuse(e.reshape(b, t, -1))
        x = fused + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        return cast_enc.last_layer_attention_probs(x, is_causal=self.causal)

    def self_attention_probs_all_layers(self, token_ids: torch.Tensor) -> list[torch.Tensor]:
        """Per encoder layer after fused token embeddings."""
        if token_ids.dim() != 3:
            raise ValueError("token_ids must be 3D [batch, seq_len, tokens_per_position]")
        b, t, k_in = token_ids.shape
        if int(k_in) != self.tokens_per_position:
            raise ValueError(
                f"token_ids last dim {int(k_in)} != model tokens_per_position {self.tokens_per_position}",
            )
        if int(t) != self.context_length:
            raise ValueError(
                f"token_ids seq_len {int(t)} != model context length {self.context_length}",
            )
        e = self.embedding(token_ids.long())
        fused = self.token_fuse(e.reshape(b, t, -1))
        x = fused + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        return cast_enc.all_layers_attention_probs(x, is_causal=self.causal)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            emb = self.embedding.weight.detach().float().cpu().numpy()
            w0 = self.lm_heads[0].weight.detach().float().cpu().numpy()
            return {"embedding": emb, "lm_head": w0}
