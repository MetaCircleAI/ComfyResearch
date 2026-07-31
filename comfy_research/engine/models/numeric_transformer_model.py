from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from comfy_research.engine.models.transformer_encoder_custom import (
    StableTransformerEncoder,
    StableTransformerEncoderLayer,
    apply_spectral_norm_to_encoder_linears,
    stable_attn_hyperparams_for_encoder_backend,
)

_ENCODER_ACT = frozenset({"gelu", "relu", "silu"})
EncoderBackend = Literal["pytorch", "stable"]


def _scalar_float(v: Any, default: float) -> float:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_encoder_activation(v: Any, *, default: str = "gelu") -> str:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v if v is not None else default).strip().lower()
    return s if s in _ENCODER_ACT else default


def _coerce_encoder_backend(v: Any) -> EncoderBackend:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v if v is not None else "pytorch").strip().lower()
    return "stable" if s == "stable" else "pytorch"


def _coerce_nonneg_int(v: Any, default: int) -> int:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        i = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, i)


def _coerce_yes_no(v: Any, *, default: bool) -> bool:
    """Canvas / JSON may store booleans or yes/no style strings (and list-wrapped scalars)."""
    if v is None:
        return default
    if isinstance(v, list) and v:
        return _coerce_yes_no(v[0], default=default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("yes", "true", "1", "on", "y"):
        return True
    if s in ("no", "false", "0", "off", "n"):
        return False
    return default


def read_numeric_transformer_layout_from_md(md: dict[str, Any]) -> tuple[int, int, int]:
    """``(context_length, token_dim, output_token_dim)``. Legacy graphs without ``contextLength`` treat ``inputDim`` as sequence length with scalar tokens."""
    if md.get("contextLength") is not None:
        return (
            _coerce_nonneg_int(md.get("contextLength"), 2),
            _coerce_nonneg_int(md.get("inputDim"), 1),
            _coerce_nonneg_int(md.get("outputDim"), 1),
        )
    seq_len = _coerce_nonneg_int(md.get("inputDim"), 2)
    return seq_len, 1, _coerce_nonneg_int(md.get("outputDim"), 1)


def numeric_transformer_from_canvas_md(md: dict[str, Any]) -> NumericTransformerModel:
    ctx, tin, tout = read_numeric_transformer_layout_from_md(md)
    causal = _coerce_yes_no(md.get("causalAttention"), default=True)
    act = _coerce_encoder_activation(md.get("activation"))
    backend = _coerce_encoder_backend(md.get("encoderBackend"))
    ed = max(0.0, min(_scalar_float(md.get("encoderDropout"), 0.0), 1.0))
    spectral = _coerce_yes_no(md.get("spectralNormLinears"), default=False)
    sqk = _coerce_yes_no(md.get("stableQkNorm"), default=False)
    stemp = max(_scalar_float(md.get("stableAttnTemperature"), 1.0), 1e-6)
    scap = max(_scalar_float(md.get("stableAttnLogitCap"), 0.0), 0.0)
    sad = max(0.0, min(_scalar_float(md.get("stableAttnDropout"), 0.0), 1.0))
    return NumericTransformerModel(
        ctx,
        tin,
        tout,
        _coerce_nonneg_int(md.get("modelDim"), 32),
        _coerce_nonneg_int(md.get("numHeads"), 1),
        _coerce_nonneg_int(md.get("numLayers"), 1),
        _coerce_nonneg_int(md.get("ffDim"), 64),
        causal=causal,
        activation=act,
        encoder_backend=backend,
        encoder_dropout=ed,
        spectral_norm_linears=spectral,
        stable_qk_norm=sqk,
        stable_attn_temperature=stemp,
        stable_attn_logit_cap=scap,
        stable_attn_dropout_p=sad,
    )


class NumericTransformerModel(nn.Module):
    """Transformer encoder on a fixed-length numeric sequence: x [batch, T, D_in] -> y [batch, T, D_out].

    When ``causal`` is True (default), the encoder uses a causal attention mask so position *i* only attends to
    indices ``<= i``. When ``causal`` is False, attention is bidirectional over the window.
    """

    def __init__(
        self,
        context_length: int,
        token_dim: int,
        output_token_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        *,
        causal: bool = True,
        activation: str = "gelu",
        encoder_backend: EncoderBackend = "pytorch",
        encoder_dropout: float = 0.0,
        spectral_norm_linears: bool = False,
        stable_qk_norm: bool = False,
        stable_attn_temperature: float = 1.0,
        stable_attn_logit_cap: float = 0.0,
        stable_attn_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if context_length < 1 or token_dim < 1 or output_token_dim < 1 or model_dim < 1:
            raise ValueError("context_length, token_dim, output_token_dim, and model_dim must be >= 1")
        if num_heads < 1 or num_layers < 1 or ff_dim < 1:
            raise ValueError("num_heads, num_layers, and ff_dim must be >= 1")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.context_length = int(context_length)
        self.token_dim = int(token_dim)
        self.output_token_dim = int(output_token_dim)
        self.model_dim = int(model_dim)
        self.causal = bool(causal)
        self.encoder_backend = encoder_backend
        self.token_proj = nn.Linear(self.token_dim, self.model_dim, bias=True)
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
        self.out_proj = nn.Linear(self.model_dim, self.output_token_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept ``[batch, T, D_in]``, or legacy ``[batch, T]`` when ``token_dim == 1``, or flat ``[batch, T*D_in]``."""
        if x.dim() == 2:
            if int(x.shape[1]) == self.context_length * self.token_dim:
                x = x.reshape(x.shape[0], self.context_length, self.token_dim)
            elif self.token_dim == 1 and int(x.shape[1]) == self.context_length:
                x = x.unsqueeze(-1)
            else:
                raise ValueError(
                    f"x must be [batch, {self.context_length}, {self.token_dim}], "
                    f"[batch, {self.context_length}] with token_dim=1, or "
                    f"[batch, {self.context_length * self.token_dim}]; got {tuple(x.shape)}"
                )
        elif x.dim() != 3:
            raise ValueError(f"x must be rank-2 or rank-3; got dim {x.dim()}")
        if int(x.shape[1]) != self.context_length or int(x.shape[2]) != self.token_dim:
            raise ValueError(
                f"x sequence shape must be [batch, {self.context_length}, {self.token_dim}]; got {tuple(x.shape)}"
            )
        tok = self.token_proj(x) + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        h = cast_enc(tok, is_causal=bool(self.causal))
        return self.out_proj(h)

    def self_attention_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Last encoder layer softmax weights ``[batch, heads, seq, seq]`` (numeric path, same layout as ``forward``)."""
        if x.dim() == 2:
            if int(x.shape[1]) == self.context_length * self.token_dim:
                x = x.reshape(x.shape[0], self.context_length, self.token_dim)
            elif self.token_dim == 1 and int(x.shape[1]) == self.context_length:
                x = x.unsqueeze(-1)
            else:
                raise ValueError(
                    f"x must be [batch, {self.context_length}, {self.token_dim}], "
                    f"[batch, {self.context_length}] with token_dim=1, or "
                    f"[batch, {self.context_length * self.token_dim}]; got {tuple(x.shape)}"
                )
        elif x.dim() != 3:
            raise ValueError(f"x must be rank-2 or rank-3; got dim {x.dim()}")
        if int(x.shape[1]) != self.context_length or int(x.shape[2]) != self.token_dim:
            raise ValueError(
                f"x sequence shape must be [batch, {self.context_length}, {self.token_dim}]; got {tuple(x.shape)}"
            )
        tok = self.token_proj(x) + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        return cast_enc.last_layer_attention_probs(tok, is_causal=self.causal)

    def self_attention_probs_all_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Per encoder layer (numeric path, same layout as ``forward``)."""
        if x.dim() == 2:
            if int(x.shape[1]) == self.context_length * self.token_dim:
                x = x.reshape(x.shape[0], self.context_length, self.token_dim)
            elif self.token_dim == 1 and int(x.shape[1]) == self.context_length:
                x = x.unsqueeze(-1)
            else:
                raise ValueError(
                    f"x must be [batch, {self.context_length}, {self.token_dim}], "
                    f"[batch, {self.context_length}] with token_dim=1, or "
                    f"[batch, {self.context_length * self.token_dim}]; got {tuple(x.shape)}"
                )
        elif x.dim() != 3:
            raise ValueError(f"x must be rank-2 or rank-3; got dim {x.dim()}")
        if int(x.shape[1]) != self.context_length or int(x.shape[2]) != self.token_dim:
            raise ValueError(
                f"x sequence shape must be [batch, {self.context_length}, {self.token_dim}]; got {tuple(x.shape)}"
            )
        tok = self.token_proj(x) + self.pos_embed.unsqueeze(0)
        cast_enc = self.encoder
        assert isinstance(cast_enc, StableTransformerEncoder)
        return cast_enc.all_layers_attention_probs(tok, is_causal=self.causal)
