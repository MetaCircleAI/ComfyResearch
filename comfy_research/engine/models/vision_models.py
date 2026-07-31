"""Small ResNet- and ViT-style image classifiers (single-channel, low-res friendly)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException

from comfy_research.schemas.graph import NodeKind


def _scalar_int(v: Any, default: int) -> int:
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _scalar_str(v: Any, default: str) -> str:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v or "").strip()
    return s if s else default


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, *, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        pad = (k - 1) // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=False)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(x)
        return F.relu(out + identity, inplace=False)


class SmallResNet(nn.Module):
    """ResNet-style CNN for 1×H×W inputs (MNIST-scale)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        layers_cfg: tuple[int, int, int, int],
        base_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k > 11 or k % 2 == 0:
            raise HTTPException(
                status_code=400,
                detail="resnet_model kernelSize must be an odd integer from 3 to 11.",
            )
        pad = (k - 1) // 2
        base = int(base_channels)
        if base < 8 or base > 256:
            raise HTTPException(
                status_code=400,
                detail="resnet_model baseChannels must be between 8 and 256.",
            )
        for nb in layers_cfg:
            if nb < 1 or nb > 16:
                raise HTTPException(
                    status_code=400,
                    detail="resnet_model blocks per stage must be between 1 and 16.",
                )
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=k, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        ch = base
        blocks: list[nn.Module] = []
        stage_channels = (base, base * 2, base * 4, base * 8)
        for si, n_blocks in enumerate(layers_cfg):
            out_ch = stage_channels[si]
            stride = 1 if si == 0 else 2
            blocks.append(_BasicBlock(ch, out_ch, stride=stride, kernel_size=k))
            ch = out_ch
            for _ in range(1, n_blocks):
                blocks.append(_BasicBlock(ch, ch, stride=1, kernel_size=k))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.body(h)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.head(h)


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm1(x)
        a, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """Patch ViT for low-resolution single-channel images."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        image_size: int,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise HTTPException(
                status_code=400,
                detail=f"ViT imageSize ({image_size}) must be divisible by patchSize ({patch_size}).",
            )
        n_p = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, n_p + 1, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([_AttentionBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        t = self.patch(x).flatten(2).transpose(1, 2)  # [B, n_p, dim]
        cls = self.cls.expand(t.shape[0], -1, -1)
        h = torch.cat([cls, t], dim=1) + self.pos
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.head(h[:, 0])


def build_resnet_from_md(md: dict[str, Any], *, in_channels: int, num_classes: int) -> nn.Module:
    variant = _scalar_str(md.get("variant"), "resnet18").strip().lower().replace("-", "_")
    if variant == "resnet18":
        layers_cfg = (2, 2, 2, 2)
        base = 32
        k = 3
    elif variant == "resnet34":
        layers_cfg = (2, 3, 2, 2)
        base = 32
        k = 3
    elif variant == "self_defined":
        base = max(8, min(256, _scalar_int(md.get("baseChannels"), 32)))
        k0 = _scalar_int(md.get("kernelSize"), 3)
        k = k0 if k0 % 2 == 1 else k0 + 1
        k = max(3, min(11, k))
        b1 = max(1, min(16, _scalar_int(md.get("blocksStage1"), 2)))
        b2 = max(1, min(16, _scalar_int(md.get("blocksStage2"), 2)))
        b3 = max(1, min(16, _scalar_int(md.get("blocksStage3"), 2)))
        b4 = max(1, min(16, _scalar_int(md.get("blocksStage4"), 2)))
        layers_cfg = (b1, b2, b3, b4)
    else:
        raise HTTPException(
            status_code=400,
            detail="resnet_model variant must be resnet18, resnet34, or self_defined.",
        )
    return SmallResNet(
        in_channels,
        num_classes,
        layers_cfg=layers_cfg,
        base_channels=base,
        kernel_size=k,
    )


def build_vit_from_md(md: dict[str, Any], *, in_channels: int, num_classes: int, image_size: int) -> nn.Module:
    variant = _scalar_str(md.get("variant"), "tiny").lower()
    if variant == "small":
        patch_size = max(2, _scalar_int(md.get("patchSize"), 4))
        dim = _scalar_int(md.get("hiddenDim"), 192)
        depth = _scalar_int(md.get("depth"), 4)
        heads = _scalar_int(md.get("numHeads"), 6)
    else:
        patch_size = max(2, _scalar_int(md.get("patchSize"), 4))
        dim = _scalar_int(md.get("hiddenDim"), 128)
        depth = _scalar_int(md.get("depth"), 3)
        heads = max(1, _scalar_int(md.get("numHeads"), 4))
    return TinyViT(
        in_channels,
        num_classes,
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
    )


def infer_vision_input_channels_height_width(
    ds_kind: NodeKind,
    dd: dict[str, Any],
) -> tuple[int, int, int]:
    size_m = max(8, min(96, _scalar_int(dd.get("imageSize"), 28)))
    size_s = max(16, min(96, _scalar_int(dd.get("imageSize"), 32)))
    size_h = max(24, min(96, _scalar_int(dd.get("imageSize"), 48)))
    if ds_kind == NodeKind.mnist_dataset:
        return 1, 28, 28
    if ds_kind == NodeKind.cifar10_dataset:
        if str(dd.get("preprocessing", "none")).strip() == "center_crop_28_per_image_whiten":
            return 3, 28, 28
        return 3, 32, 32
    if ds_kind == NodeKind.gaussian_blob_dataset:
        return 1, size_m, size_m
    if ds_kind == NodeKind.shape_world_dataset:
        return 1, size_s, size_s
    if ds_kind == NodeKind.hole_counting_dataset:
        return 1, size_h, size_h
    raise HTTPException(status_code=500, detail=f"Internal: vision dims for {ds_kind}")
