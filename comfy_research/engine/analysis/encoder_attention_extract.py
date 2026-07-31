"""Helpers to read softmax self-attention from a PyTorch ``nn.TransformerEncoder`` stack (legacy / external use).

ComfyResearch token and numeric transformers use :class:`comfy_research.engine.models.transformer_encoder_custom.StableTransformerEncoder`
instead; prefer its ``last_layer_attention_probs`` / ``all_layers_attention_probs`` methods there.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pytorch_encoder_last_layer_attention_probs(
    encoder: nn.TransformerEncoder,
    x: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    """Return last encoder layer's MHA softmax weights ``[batch, heads, seq, seq]`` (no dropout)."""
    layers = list(encoder.layers)
    if not layers:
        raise ValueError("TransformerEncoder has no layers")
    attn_mask: torch.Tensor | None = None
    if is_causal:
        lseq = int(x.size(1))
        attn_mask = nn.Transformer.generate_square_subsequent_mask(lseq, device=x.device, dtype=torch.float32).to(
            dtype=x.dtype
        )
    h = x
    for layer in layers[:-1]:
        h = layer(h, src_mask=attn_mask, is_causal=is_causal)
    last = layers[-1]
    y = last.norm1(h)
    _attn_out, attn_w = last.self_attn(
        y,
        y,
        y,
        attn_mask=attn_mask,
        need_weights=True,
        average_attn_weights=False,
        is_causal=is_causal,
    )
    return attn_w


def pytorch_encoder_all_layers_attention_probs(
    encoder: nn.TransformerEncoder,
    x: torch.Tensor,
    *,
    is_causal: bool,
) -> list[torch.Tensor]:
    """Return each encoder layer's MHA softmax weights ``[batch, heads, seq, seq]`` (no dropout).

    Runs an extra forward pass per layer (small ``num_layers`` in typical graphs); matches
    :func:`pytorch_encoder_last_layer_attention_probs` on the final layer.
    """
    layers = list(encoder.layers)
    if not layers:
        raise ValueError("TransformerEncoder has no layers")
    attn_mask: torch.Tensor | None = None
    if is_causal:
        lseq = int(x.size(1))
        attn_mask = nn.Transformer.generate_square_subsequent_mask(lseq, device=x.device, dtype=torch.float32).to(
            dtype=x.dtype
        )
    out: list[torch.Tensor] = []
    for i in range(len(layers)):
        h = x
        for j in range(i):
            h = layers[j](h, src_mask=attn_mask, is_causal=is_causal)
        lay = layers[i]
        y = lay.norm1(h)
        _attn_out, attn_w = lay.self_attn(
            y,
            y,
            y,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        out.append(attn_w)
    return out
