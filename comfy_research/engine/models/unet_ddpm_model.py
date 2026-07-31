"""Small time-conditioned UNet used by the CIFAR-10 DDPM node."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException


def _int(value: Any, default: int) -> int:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _channel_mult(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        value = value[0] if value else "1,2,2"
    try:
        parts = tuple(max(1, int(part.strip())) for part in str(value).split(",") if part.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="channelMult must be comma-separated positive integers.") from exc
    return parts or (1, 2, 2)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10_000.0) / max(1, half - 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat((args.sin(), args.cos()), dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.net(emb)


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels))
        while out_channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNetDdpmModel(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 2),
        time_embed_dim: int = 128,
        diffusion_timesteps: int = 1000,
        image_size: int = 32,
    ) -> None:
        super().__init__()
        if image_size % (2 ** (len(channel_mult) - 1)):
            raise HTTPException(status_code=400, detail="imageSize must be divisible by the UNet downsampling factor.")
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.max_timesteps = int(diffusion_timesteps)
        widths = [int(base_channels) * mult for mult in channel_mult]
        self.time = _TimeEmbedding(int(time_embed_dim))
        self.input = nn.Conv2d(self.in_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = widths[0]
        for index, width in enumerate(widths):
            self.down_blocks.append(_ResBlock(prev, width, time_embed_dim))
            prev = width
            if index < len(widths) - 1:
                self.downsamples.append(nn.Conv2d(width, widths[index + 1], 4, stride=2, padding=1))
                prev = widths[index + 1]
        self.mid = _ResBlock(widths[-1], widths[-1], time_embed_dim)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        current = widths[-1]
        for index in range(len(widths) - 1, -1, -1):
            skip = widths[index]
            self.up_blocks.append(_ResBlock(current + skip, skip, time_embed_dim))
            current = skip
            if index > 0:
                self.upsamples.append(nn.ConvTranspose2d(current, widths[index - 1], 4, stride=2, padding=1))
                current = widths[index - 1]
        groups = max(1, min(8, current))
        while current % groups:
            groups -= 1
        self.out = nn.Sequential(nn.GroupNorm(groups, current), nn.SiLU(), nn.Conv2d(current, self.in_channels, 3, padding=1))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise HTTPException(status_code=400, detail="UNet DDPM expects NCHW inputs with the configured channel count.")
        t = self.time(timesteps)
        h = self.input(x)
        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.down_blocks):
            h = block(h, t)
            skips.append(h)
            if index < len(self.downsamples):
                h = self.downsamples[index](h)
        h = self.mid(h, t)
        for index, block in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat((h, skip), dim=1), t)
            if index < len(self.upsamples):
                h = self.upsamples[index](h)
        return self.out(h)


def build_unet_ddpm_from_md(data: dict[str, Any]) -> UNetDdpmModel:
    return UNetDdpmModel(
        in_channels=_int(data.get("inChannels"), 3),
        base_channels=_int(data.get("baseChannels"), 64),
        channel_mult=_channel_mult(data.get("channelMult")),
        time_embed_dim=_int(data.get("timeEmbedDim"), 128),
        diffusion_timesteps=_int(data.get("diffusionTimesteps"), 1000),
        image_size=_int(data.get("imageSize"), 32),
    )
