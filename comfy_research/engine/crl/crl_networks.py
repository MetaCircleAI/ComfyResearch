"""PyTorch CRL networks aligned with scaling-crl ``train.py`` (residual blocks, φ/ψ, Gaussian actor)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

_CRL_MLP_ACTIVATION_IDS = frozenset(
    {"relu", "gelu", "tanh", "sigmoid", "leaky_relu", "silu", "identity"},
)


def parse_crl_residual_activation(meta: Mapping[str, Any]) -> str:
    """Match MLP node ids; legacy ``useReLU`` maps to relu / silu."""
    raw = meta.get("activation")
    if raw is not None:
        s = str(raw).strip().lower()
        if s in _CRL_MLP_ACTIVATION_IDS:
            return s
    if bool(meta.get("useReLU", False)):
        return "relu"
    return "silu"


def _activation_factory(activation: str) -> Callable[[], nn.Module]:
    a = parse_crl_residual_activation({"activation": activation})
    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(),
        "gelu": lambda: nn.GELU(),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "silu": lambda: nn.SiLU(),
        "identity": lambda: nn.Identity(),
    }
    return factories[a]


class _ResidualBlock(nn.Module):
    """Four Dense→LayerNorm→activation inside a residual (matches Flax ``residual_block``)."""

    def __init__(self, width: int, *, activation: str) -> None:
        super().__init__()
        self.width = width
        act_fn = _activation_factory(activation)
        layers: list[nn.Module] = []
        for _ in range(4):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LayerNorm(width))
            layers.append(act_fn())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SAEncoder(nn.Module):
    """φ(s,a): state+action → embedding (``SA_encoder`` in scaling-crl ``train.py``)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        width: int,
        depth: int,
        embed_dim: int,
        *,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if depth % 4 != 0 or depth < 4:
            raise ValueError("critic_depth must be >=4 and divisible by 4 (total Dense count convention).")
        n_blocks = depth // 4
        act_fn = _activation_factory(activation)
        self.in_proj = nn.Linear(state_dim + action_dim, width)
        self.in_ln = nn.LayerNorm(width)
        self._nonlin = act_fn()
        self.blocks = nn.ModuleList([_ResidualBlock(width, activation=activation) for _ in range(n_blocks)])
        self.out = nn.Linear(width, embed_dim)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        x = self._nonlin(self.in_ln(self.in_proj(x)))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


class GEncoder(nn.Module):
    """ψ(g): goal → embedding."""

    def __init__(self, goal_dim: int, width: int, depth: int, embed_dim: int, *, activation: str = "silu") -> None:
        super().__init__()
        if depth % 4 != 0 or depth < 4:
            raise ValueError("critic_depth must be >=4 and divisible by 4.")
        n_blocks = depth // 4
        act_fn = _activation_factory(activation)
        self.in_proj = nn.Linear(goal_dim, width)
        self.in_ln = nn.LayerNorm(width)
        self._nonlin = act_fn()
        self.blocks = nn.ModuleList([_ResidualBlock(width, activation=activation) for _ in range(n_blocks)])
        self.out = nn.Linear(width, embed_dim)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        x = self._nonlin(self.in_ln(self.in_proj(g)))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


class GaussianActor(nn.Module):
    """Policy on ``concat(state, goal)`` with tanh-squashed Gaussian (``Actor`` in scaling-crl)."""

    def __init__(self, obs_dim: int, action_dim: int, width: int, depth: int, *, activation: str = "silu") -> None:
        super().__init__()
        if depth % 4 != 0 or depth < 4:
            raise ValueError("actor_depth must be >=4 and divisible by 4.")
        n_blocks = depth // 4
        act_fn = _activation_factory(activation)
        self.in_proj = nn.Linear(obs_dim, width)
        self.in_ln = nn.LayerNorm(width)
        self._nonlin = act_fn()
        self.blocks = nn.ModuleList([_ResidualBlock(width, activation=activation) for _ in range(n_blocks)])
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)

    def forward(self, obs_sg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._nonlin(self.in_ln(self.in_proj(obs_sg)))
        for blk in self.blocks:
            x = blk(x)
        return self.mean(x), self.log_std(x)


class CrlResidualAgent(nn.Module):
    """Actor + twin critic encoders + learnable log α (single module for checkpoints / observables)."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        obs_dim_full: int,
        actor_width: int,
        critic_width: int,
        actor_depth: int,
        critic_depth: int,
        embed_dim: int = 64,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.obs_dim_full = obs_dim_full
        self.actor = GaussianActor(obs_dim_full, action_dim, actor_width, actor_depth, activation=activation)
        self.sa_encoder = SAEncoder(state_dim, action_dim, critic_width, critic_depth, embed_dim, activation=activation)
        self.g_encoder = GEncoder(goal_dim, critic_width, critic_depth, embed_dim, activation=activation)
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
