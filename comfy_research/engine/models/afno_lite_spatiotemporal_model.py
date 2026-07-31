"""AFNO-lite spatiotemporal modules and composite model for flat PDE tensors.

The public constructors in this module all accept flattened tensors:
``[batch, T*C*H*W]`` and return the same shape, so they can be used in
ComfyResearch trainer regression pipelines and in atomic layer chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException


def _scalar_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class AfnoLiteConfig:
    context_frames: int
    channels: int
    grid_size: int
    input_dim: int
    output_dim: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    ff_ratio: float
    dropout: float
    num_spectral_blocks: int
    max_frequency_modes: int
    spectral_shrink_factor: float

    @property
    def flat_dim(self) -> int:
        return int(self.context_frames * self.channels * self.grid_size * self.grid_size)

    @property
    def patch_grid(self) -> tuple[int, int]:
        hp = self.grid_size // self.patch_size
        return hp, hp

    @property
    def tokens_per_frame(self) -> int:
        hp, wp = self.patch_grid
        return int(hp * wp)

    @property
    def seq_len(self) -> int:
        return int(self.context_frames * self.tokens_per_frame)


def afno_lite_config_from_canvas_md(md: dict[str, Any]) -> AfnoLiteConfig:
    ctx = max(1, _scalar_int(md.get("contextFrames"), 4))
    ch = max(1, _scalar_int(md.get("channels"), 1))
    g = max(4, _scalar_int(md.get("gridSize"), 16))
    patch = max(1, _scalar_int(md.get("patchSize"), 4))
    if g % patch != 0:
        raise HTTPException(status_code=400, detail="AFNO-lite patchSize must divide gridSize.")
    emb = max(16, _scalar_int(md.get("embedDim"), 64))
    depth = max(1, _scalar_int(md.get("depth"), 2))
    heads = max(1, _scalar_int(md.get("numHeads"), 4))
    if emb % heads != 0:
        raise HTTPException(status_code=400, detail="AFNO-lite embedDim must be divisible by numHeads.")
    ff_ratio = max(0.25, _scalar_float(md.get("ffRatio"), 2.0))
    dropout = _scalar_float(md.get("dropout"), 0.0)
    if dropout < 0.0 or dropout > 1.0:
        raise HTTPException(status_code=400, detail="AFNO-lite dropout must be in [0, 1].")
    n_spec = max(1, _scalar_int(md.get("numSpectralBlocks"), 1))
    max_modes = max(1, _scalar_int(md.get("maxFrequencyModes"), 4))
    shrink = _scalar_float(md.get("spectralShrinkFactor"), 1.0)
    if shrink <= 0.0:
        raise HTTPException(status_code=400, detail="AFNO-lite spectralShrinkFactor must be > 0.")
    cfg = AfnoLiteConfig(
        context_frames=ctx,
        channels=ch,
        grid_size=g,
        input_dim=_scalar_int(md.get("inputDim"), ctx * ch * g * g),
        output_dim=_scalar_int(md.get("outputDim"), ctx * ch * g * g),
        patch_size=patch,
        embed_dim=emb,
        depth=depth,
        num_heads=heads,
        ff_ratio=ff_ratio,
        dropout=dropout,
        num_spectral_blocks=n_spec,
        max_frequency_modes=max_modes,
        spectral_shrink_factor=shrink,
    )
    if cfg.input_dim != cfg.flat_dim or cfg.output_dim != cfg.flat_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"AFNO-lite flat I/O must be contextFrames*channels*gridSize^2 = {cfg.flat_dim}; "
                f"got inputDim={cfg.input_dim}, outputDim={cfg.output_dim}."
            ),
        )
    return cfg


def _flat_to_btchw(x_flat: torch.Tensor, cfg: AfnoLiteConfig) -> torch.Tensor:
    if x_flat.dim() != 2 or int(x_flat.shape[1]) != cfg.flat_dim:
        raise ValueError(f"AFNO-lite expects [batch, {cfg.flat_dim}], got {tuple(x_flat.shape)}")
    b = int(x_flat.shape[0])
    return x_flat.reshape(b, cfg.context_frames, cfg.channels, cfg.grid_size, cfg.grid_size)


def _btchw_to_flat(x: torch.Tensor, cfg: AfnoLiteConfig) -> torch.Tensor:
    b = int(x.shape[0])
    return x.reshape(b, cfg.flat_dim)


class FieldNormalizer(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(1, 3, 4), keepdim=True)
        std = x.std(dim=(1, 3, 4), keepdim=True).clamp_min(1e-7)
        return (x - mean) / std, mean, std

    def denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class PatchEmbed2D(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Conv2d(cfg.channels, cfg.embed_dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.proj_in = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = int(x.shape[0])
        tokens: list[torch.Tensor] = []
        for ti in range(self.cfg.context_frames):
            z = self.embed(x[:, ti]).flatten(2).transpose(1, 2)
            tokens.append(self.proj_in(z))
        return torch.stack(tokens, dim=1).reshape(b, self.cfg.context_frames, self.cfg.tokens_per_frame, self.cfg.embed_dim)


class PatchDecode2D(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_pixels = int(cfg.patch_size * cfg.patch_size * cfg.channels)
        self.proj_out = nn.Linear(cfg.embed_dim, patch_pixels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = int(tokens.shape[0])
        hp, wp = self.cfg.patch_grid
        p = self.cfg.patch_size
        c = self.cfg.channels
        h = self.cfg.grid_size
        out_frames: list[torch.Tensor] = []
        for ti in range(self.cfg.context_frames):
            sl = tokens[:, ti]
            pix = self.proj_out(sl)
            pix = pix.reshape(b, hp, wp, c, p, p)
            pix = pix.permute(0, 3, 1, 4, 2, 5).contiguous().reshape(b, c, h, h)
            out_frames.append(pix)
        return torch.stack(out_frames, dim=1)


class SpectralMixer2D(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hp, wp = cfg.patch_grid
        self.mh = min(int(cfg.max_frequency_modes), hp)
        self.mw = min(int(cfg.max_frequency_modes), wp // 2 + 1)
        scale = float(cfg.spectral_shrink_factor) / max(1.0, float(cfg.embed_dim))
        self.wr = nn.Parameter(torch.randn(cfg.embed_dim, self.mh, self.mw) * scale)
        self.wi = nn.Parameter(torch.randn(cfg.embed_dim, self.mh, self.mw) * scale)
        self.num_blocks = max(1, int(cfg.num_spectral_blocks))

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        b, t, p, d = x_tokens.shape
        hp, wp = self.cfg.patch_grid
        if p != hp * wp or d != self.cfg.embed_dim:
            raise ValueError("SpectralMixer2D received mismatched token shape.")
        z = x_tokens.reshape(b * t, hp, wp, d).permute(0, 3, 1, 2).contiguous()
        for _ in range(self.num_blocks):
            freq = torch.fft.rfft2(z, s=(hp, wp), dim=(-2, -1), norm="ortho")
            wr = self.wr.to(dtype=freq.real.dtype)[None, :, :, :]
            wi = self.wi.to(dtype=freq.real.dtype)[None, :, :, :]
            low = freq[:, :, : self.mh, : self.mw]
            new_low = torch.complex(low.real * wr - low.imag * wi, low.real * wi + low.imag * wr)
            freq = freq.clone()
            freq[:, :, : self.mh, : self.mw] = new_low
            z = torch.fft.irfft2(freq, s=(hp, wp), dim=(-2, -1), norm="ortho")
        out = z.permute(0, 2, 3, 1).reshape(b, t, p, d)
        return out


class AfnoLiteBlock(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.mix = SpectralMixer2D(cfg)
        hidden = max(8, int(round(cfg.ff_ratio * cfg.embed_dim)))
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(hidden, cfg.embed_dim),
        )
        self.drop = nn.Dropout(float(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mix(self.ln1(x)))
        return x + self.drop(self.ffn(self.ln2(x)))


class AfnoLiteEncoder(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.context_frames, cfg.tokens_per_frame, cfg.embed_dim))
        self.blocks = nn.ModuleList([AfnoLiteBlock(cfg) for _ in range(max(1, cfg.depth))])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x + self.pos_embed
        for blk in self.blocks:
            z = blk(z)
        return z


class _AfnoFlatBase(nn.Module):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = FieldNormalizer()

    def _to_field(self, x_flat: torch.Tensor) -> torch.Tensor:
        return _flat_to_btchw(x_flat, self.cfg)

    def _to_flat(self, x: torch.Tensor) -> torch.Tensor:
        return _btchw_to_flat(x, self.cfg)


class AfnoPatchEmbedLayer(_AfnoFlatBase):
    """Atomic module: patchify + learned patch projection + immediate decode."""

    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__(cfg)
        self.embed = PatchEmbed2D(cfg)
        self.decode = PatchDecode2D(cfg)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self._to_field(x_flat)
        x_n, mean, std = self.norm(x)
        toks = self.embed(x_n)
        out = self.decode(toks)
        return self._to_flat(self.norm.denorm(out, mean, std))


class AfnoSpectralMixerLayer(_AfnoFlatBase):
    """Atomic module: direct low-frequency spectral mixing in field space."""

    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__(cfg)
        g = cfg.grid_size
        mh = min(int(cfg.max_frequency_modes), g)
        mw = min(int(cfg.max_frequency_modes), g // 2 + 1)
        self.mh = mh
        self.mw = mw
        scale = float(cfg.spectral_shrink_factor) / max(1.0, float(cfg.channels))
        self.wr = nn.Parameter(torch.randn(cfg.channels, mh, mw) * scale)
        self.wi = nn.Parameter(torch.randn(cfg.channels, mh, mw) * scale)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self._to_field(x_flat)
        x_n, mean, std = self.norm(x)
        b, t, c, h, w = x_n.shape
        z = x_n.reshape(b * t, c, h, w)
        freq = torch.fft.rfft2(z, s=(h, w), dim=(-2, -1), norm="ortho")
        wr = self.wr.to(dtype=freq.real.dtype)[None, :, :, :]
        wi = self.wi.to(dtype=freq.real.dtype)[None, :, :, :]
        low = freq[:, :, : self.mh, : self.mw]
        mixed = torch.complex(low.real * wr - low.imag * wi, low.real * wi + low.imag * wr)
        freq = freq.clone()
        freq[:, :, : self.mh, : self.mw] = mixed
        out = torch.fft.irfft2(freq, s=(h, w), dim=(-2, -1), norm="ortho").reshape(b, t, c, h, w)
        return self._to_flat(self.norm.denorm(out, mean, std))


class AfnoEncoderBlockLayer(_AfnoFlatBase):
    """Atomic module: one AFNO-lite encoder block in token space."""

    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__(cfg)
        self.embed = PatchEmbed2D(cfg)
        self.block = AfnoLiteBlock(cfg)
        self.decode = PatchDecode2D(cfg)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self._to_field(x_flat)
        x_n, mean, std = self.norm(x)
        toks = self.embed(x_n)
        out = self.decode(self.block(toks))
        return self._to_flat(self.norm.denorm(out, mean, std))


class AfnoPatchDecodeLayer(_AfnoFlatBase):
    """Atomic module: lightweight spatial refinement head in field space."""

    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__(cfg)
        self.refine = nn.Conv2d(cfg.channels, cfg.channels, kernel_size=3, padding=1, groups=1)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self._to_field(x_flat)
        x_n, mean, std = self.norm(x)
        b, t, c, h, w = x_n.shape
        z = x_n.reshape(b * t, c, h, w)
        z = z + self.refine(z)
        out = z.reshape(b, t, c, h, w)
        return self._to_flat(self.norm.denorm(out, mean, std))


class AfnoLiteSpatiotemporalModel(_AfnoFlatBase):
    def __init__(self, cfg: AfnoLiteConfig) -> None:
        super().__init__(cfg)
        self.embed = PatchEmbed2D(cfg)
        self.encoder = AfnoLiteEncoder(cfg)
        self.decode = PatchDecode2D(cfg)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = self._to_field(x_flat)
        x_n, mean, std = self.norm(x)
        toks = self.embed(x_n)
        toks = self.encoder(toks)
        out = self.decode(toks)
        out = self.norm.denorm(out, mean, std)
        return self._to_flat(out)


def afno_lite_spatiotemporal_from_canvas_md(md: dict[str, Any]) -> AfnoLiteSpatiotemporalModel:
    cfg = afno_lite_config_from_canvas_md(md)
    return AfnoLiteSpatiotemporalModel(cfg)


def afno_patch_embed_layer_from_canvas_md(md: dict[str, Any]) -> AfnoPatchEmbedLayer:
    cfg = afno_lite_config_from_canvas_md(md)
    return AfnoPatchEmbedLayer(cfg)


def afno_spectral_mixer_layer_from_canvas_md(md: dict[str, Any]) -> AfnoSpectralMixerLayer:
    cfg = afno_lite_config_from_canvas_md(md)
    return AfnoSpectralMixerLayer(cfg)


def afno_encoder_block_layer_from_canvas_md(md: dict[str, Any]) -> AfnoEncoderBlockLayer:
    cfg = afno_lite_config_from_canvas_md(md)
    return AfnoEncoderBlockLayer(cfg)


def afno_patch_decode_layer_from_canvas_md(md: dict[str, Any]) -> AfnoPatchDecodeLayer:
    cfg = afno_lite_config_from_canvas_md(md)
    return AfnoPatchDecodeLayer(cfg)
