"""Causal depthwise 1D convolution along sequence (Canon-lite horizontal mixing)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthwiseConv1d(nn.Module):
    """Mix each channel across past time steps only; preserves ``[batch, T, channels]`` layout."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        c = int(channels)
        ks = int(kernel_size)
        if c < 1:
            raise ValueError("channels must be >= 1")
        if ks < 3 or ks % 2 != 1:
            raise ValueError("kernel_size must be odd and >= 3")
        self.channels = c
        self.kernel_size = ks
        self.conv = nn.Conv1d(c, c, ks, groups=c, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, seq, channels], got {tuple(x.shape)}")
        # [B, T, C] -> [B, C, T]
        z = x.transpose(1, 2)
        z = F.pad(z, (self.kernel_size - 1, 0))
        z = self.conv(z)
        return z.transpose(1, 2)


class CausalLocalMixingResidual(nn.Module):
    """Residual causal depthwise mixing: ``y = x + conv(x)``. Supports ``[B, T, C]``; rank-2 ``[B, C]`` is treated as ``T=1``."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.mix = CausalDepthwiseConv1d(int(channels), int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            y = x.unsqueeze(1) + self.mix(x.unsqueeze(1))
            return y.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"local_mixing_layer expects rank 2 or 3, got shape {tuple(x.shape)}")
        return x + self.mix(x)
