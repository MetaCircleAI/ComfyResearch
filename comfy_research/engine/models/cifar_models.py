"""VGG-11 classifier for 32×32 CIFAR inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class VGG11Cifar(nn.Module):
    """VGG-11 style stack adapted for 32×32 RGB."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers: list[nn.Module] = []
        ch = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [
                    nn.Conv2d(ch, int(v), 3, padding=1),
                    nn.BatchNorm2d(int(v)),
                    nn.ReLU(inplace=True),
                ]
                ch = int(v)
        self.features = nn.Sequential(*layers)
        # Paper 1711.04623: FC-512, FC-512, FC-10 (VGG-style Dropout kept).
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.classifier(h.flatten(1))


def build_vgg11_cifar(*, in_channels: int, num_classes: int) -> nn.Module:
    return VGG11Cifar(in_channels, num_classes)



class _ScaleFreeBatchNorm2d(nn.Module):
    """TensorFlow-style batch norm with ``scale=False`` for Small Inception."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm2d(channels, affine=False)
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(x) + self.bias.view(1, -1, 1, 1)


class _SmallInceptionConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            _ScaleFreeBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class _SmallInceptionModule(nn.Module):
    def __init__(self, in_channels: int, channels_1x1: int, channels_3x3: int) -> None:
        super().__init__()
        self.path_1x1 = _SmallInceptionConv(in_channels, channels_1x1, kernel_size=1)
        self.path_3x3 = _SmallInceptionConv(in_channels, channels_3x3, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.path_1x1(x), self.path_3x3(x)), dim=1)


class _SmallInceptionDownsample(nn.Module):
    def __init__(self, in_channels: int, conv_channels: int) -> None:
        super().__init__()
        self.conv = _SmallInceptionConv(in_channels, conv_channels, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.conv(x), self.pool(x)), dim=1)


class SmallInceptionCifar(nn.Module):
    """Small Inception classifier used in Zhang et al.'s CIFAR experiments.

    Adaptive pooling makes the architecture valid for both the application's
    default 32×32 CIFAR tensors and the paper's 28×28 preprocessing protocol.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _SmallInceptionConv(in_channels, 96, kernel_size=3),
            _SmallInceptionModule(96, 32, 32), _SmallInceptionModule(64, 32, 48),
            _SmallInceptionDownsample(80, 80), _SmallInceptionModule(160, 112, 48),
            _SmallInceptionModule(160, 96, 64), _SmallInceptionModule(160, 80, 80),
            _SmallInceptionModule(160, 48, 96), _SmallInceptionDownsample(144, 96),
            _SmallInceptionModule(240, 176, 160), _SmallInceptionModule(336, 176, 160),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(336, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_small_inception_cifar(*, in_channels: int, num_classes: int) -> nn.Module:
    return SmallInceptionCifar(in_channels, num_classes)
