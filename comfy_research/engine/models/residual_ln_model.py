from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str) -> nn.Module:
    n = str(name).strip().lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n == "tanh":
        return nn.Tanh()
    if n == "sigmoid":
        return nn.Sigmoid()
    if n == "leaky_relu":
        return nn.LeakyReLU()
    if n == "silu":
        return nn.SiLU()
    if n == "identity":
        return nn.Identity()
    return nn.ReLU()


class ResidualLNBlock(nn.Module):
    def __init__(self, dim: int, alpha: float, activation: str, *, pre_ln: bool) -> None:
        super().__init__()
        self.pre_ln = bool(pre_ln)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = _activation(activation)
        self.alpha = float(alpha)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.pre_ln:
            u = self.ln(h)
            x = self.fc2(self.act(self.fc1(u)))
            return h + self.alpha * x
        x = self.fc2(self.act(self.fc1(h)))
        return self.ln(h + self.alpha * x)


class ResidualLNModel(nn.Module):
    """Residual stream model that preserves width across layers."""

    def __init__(self, dim: int, depth: int, alpha: float, activation: str, *, pre_ln: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.blocks = nn.ModuleList(
            [
                ResidualLNBlock(
                    self.dim,
                    alpha,
                    activation,
                    pre_ln=pre_ln,
                )
                for _ in range(self.depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        return h


def compute_residual_stream_tensors(model: ResidualLNModel, x: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {"h0": x, "input": x}
    h = x
    for i, blk in enumerate(model.blocks, start=1):
        h = blk(h)
        out[f"h{i}"] = h
    out["output"] = h
    return out


def residual_representation_ids(depth: int) -> set[str]:
    ids: set[str] = {"input", "output", "h0"}
    for i in range(1, int(depth) + 1):
        ids.add(f"h{i}")
    return ids
