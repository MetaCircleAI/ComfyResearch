"""Keskar et al. C1/C2 CNN classifiers for CIFAR (Appendix B.3/B.4; matches official network_zoo)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from fastapi import HTTPException


def _scalar_str(v: Any, default: str) -> str:
    if isinstance(v, list) and v:
        v = v[0]
    s = str(v or "").strip()
    return s if s else default


class KeskarC1(nn.Module):
    """C1 shallow AlexNet-style CIFAR classifier (paper B.3 / shallownet)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(384, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class KeskarC2(nn.Module):
    """C2 deep VGG-style CIFAR classifier (paper B.4 / deepnet)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        def block(ch_in: int, ch_out: int, n: int, *, pool: bool) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(n):
                layers.extend(
                    [
                        nn.Conv2d(ch_in if i == 0 else ch_out, ch_out, kernel_size=3, padding=1),
                        nn.BatchNorm2d(ch_out),
                        nn.ReLU(inplace=True),
                    ]
                )
                if i == 0 and ch_in != ch_out:
                    pass
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(in_channels, 64, 2, pool=True),
            block(64, 128, 2, pool=True),
            block(128, 256, 3, pool=True),
            block(256, 512, 3, pool=True),
            block(512, 512, 3, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_keskar_from_md(md: dict[str, Any], *, in_channels: int, num_classes: int) -> nn.Module:
    arch = _scalar_str(md.get("architecture"), "c1").lower()
    if arch == "c2":
        return KeskarC2(in_channels, num_classes)
    if arch in ("c1", ""):
        return KeskarC1(in_channels, num_classes)
    raise HTTPException(status_code=400, detail="keskar_c1_c2_cnn_model architecture must be 'c1' or 'c2'.")
