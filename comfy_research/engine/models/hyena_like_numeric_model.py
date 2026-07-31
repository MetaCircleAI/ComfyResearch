from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from comfy_research.engine.models.hyena_like_conv_model import HyenaLikeBlock
from comfy_research.engine.models.local_mixing import CausalDepthwiseConv1d


def _coerce_nonneg_int(v: Any, default: int) -> int:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        i = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, i)


def _coerce_zero_or_pos_int(v: Any, default: int) -> int:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        i = int(v)
    except (TypeError, ValueError):
        return max(0, int(default))
    return max(0, i)


class NumericHyenaModel(nn.Module):
    """Hyena-like sequence mixer for numeric tensors: x [batch, T, D_in] -> y [batch, T, D_out]."""

    def __init__(
        self,
        context_length: int,
        token_dim: int,
        output_token_dim: int,
        model_dim: int,
        depth: int = 2,
        kernel_size: int = 7,
        ff_mult: int = 2,
        *,
        local_mixing_kernel: int = 0,
    ) -> None:
        super().__init__()
        if context_length < 1 or token_dim < 1 or output_token_dim < 1 or model_dim < 1:
            raise ValueError("context_length, token_dim, output_token_dim, and model_dim must be >= 1")
        self.context_length = int(context_length)
        self.token_dim = int(token_dim)
        self.output_token_dim = int(output_token_dim)
        self.model_dim = int(model_dim)
        dep = max(1, int(depth))
        self.token_proj = nn.Linear(self.token_dim, self.model_dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(self.context_length, self.model_dim))
        lk = int(local_mixing_kernel)
        if lk >= 3:
            if lk % 2 == 0:
                lk += 1
            self.local_mix: nn.Module | None = CausalDepthwiseConv1d(self.model_dim, lk)
        else:
            self.local_mix = None
        self.blocks = nn.ModuleList([HyenaLikeBlock(self.model_dim, int(kernel_size), ff_mult=int(ff_mult)) for _ in range(dep)])
        self.out_ln = nn.LayerNorm(self.model_dim)
        self.out_proj = nn.Linear(self.model_dim, self.output_token_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        h = self.token_proj(x) + self.pos_embed.unsqueeze(0)
        if self.local_mix is not None:
            h = h + self.local_mix(h)
        for blk in self.blocks:
            h = blk(h)
        h = self.out_ln(h)
        return self.out_proj(h)


def numeric_hyena_from_canvas_md(md: dict[str, Any]) -> NumericHyenaModel:
    ctx = _coerce_nonneg_int(md.get("contextLength"), 8)
    tin = _coerce_nonneg_int(md.get("inputDim"), 2)
    tout = _coerce_nonneg_int(md.get("outputDim"), 2)
    return NumericHyenaModel(
        context_length=ctx,
        token_dim=tin,
        output_token_dim=tout,
        model_dim=_coerce_nonneg_int(md.get("modelDim"), 64),
        depth=_coerce_nonneg_int(md.get("depth"), 2),
        kernel_size=_coerce_nonneg_int(md.get("convKernel"), 7),
        ff_mult=_coerce_nonneg_int(md.get("ffMult"), 2),
        local_mixing_kernel=_coerce_zero_or_pos_int(md.get("localMixingKernel"), 0),
    )


def read_numeric_hyena_layout_from_md(md: dict[str, Any]) -> tuple[int, int, int]:
    return (
        _coerce_nonneg_int(md.get("contextLength"), 8),
        _coerce_nonneg_int(md.get("inputDim"), 2),
        _coerce_nonneg_int(md.get("outputDim"), 2),
    )
