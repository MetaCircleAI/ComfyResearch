"""RMSNorm for atomic layer chains (matches LLaMA-style RMS scale without mean centering)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        self.weight = nn.Parameter(torch.ones(self.dim)) if self.elementwise_affine else None  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expects last dim {self.dim}, got {int(x.shape[-1])}")
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        out = x / rms
        if self.weight is not None:
            out = out * self.weight
        return out
