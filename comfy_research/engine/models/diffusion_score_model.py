"""Small MLP score network ε_θ(x_t, t) for DDPM-style noise prediction on vectors."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for integer timesteps ``t`` shape [B]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class DiffusionScoreMLP(nn.Module):
    """Predicts noise ε given noisy ``x_t`` and timestep index."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 128,
        depth: int = 3,
        time_embed_dim: int = 64,
        max_timesteps: int = 100,
    ) -> None:
        super().__init__()
        self.data_dim = int(data_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_timesteps = max(max(2, int(max_timesteps)), 2)
        te = max(8, int(time_embed_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(te, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_in = nn.Linear(te, te)
        layers: list[nn.Module] = [nn.Linear(self.data_dim + self.hidden_dim, self.hidden_dim), nn.SiLU()]
        dep = max(1, int(depth))
        for _ in range(dep - 1):
            layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(self.hidden_dim, self.data_dim))
        self.net = nn.Sequential(*layers)
        self._te_dim = te

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        raw = _timestep_embedding(t.clamp(min=0, max=self.max_timesteps - 1), self._te_dim)
        return self.time_embed(self.time_in(raw))

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_noisy.dim() != 2:
            raise ValueError("x_noisy must be [batch, data_dim]")
        te = self.embed_time(t)
        h = torch.cat([x_noisy, te], dim=-1)
        return self.net(h)

    def observable_numpy_arrays(self) -> dict[str, np.ndarray]:
        return {}


def ddpm_schedule_sqrt_ab(timesteps: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear beta schedule; returns sqrt(alpha_bar) and sqrt(1-alpha_bar) per t in [0, T-1]."""
    t_max = max(int(timesteps), 2)
    beta = torch.linspace(1e-4, 2e-2, t_max, device=device, dtype=dtype)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar.clamp(max=1.0 - 1e-6))


def diffusion_noise_mse_loss(
    model: nn.Module,
    x0: torch.Tensor,
    rng: torch.Generator | None,
    *,
    timesteps: int,
) -> torch.Tensor:
    """Sample t and ε, form x_t, return MSE(model(x_t,t), ε).

    The score model can operate on either vectors ``[B, D]`` or image tensors
    such as CIFAR's ``[B, C, H, W]``.  Only the leading batch dimension is
    special to the DDPM schedule.
    """
    if x0.dim() < 2:
        raise ValueError("x0 must include a batch dimension and at least one feature dimension")
    b = int(x0.shape[0])
    device = x0.device
    dtype = x0.dtype
    t_max = max(int(timesteps), 2)
    t = torch.randint(0, t_max, (b,), device=device, generator=rng)
    eps = torch.randn(x0.shape, device=device, dtype=dtype, generator=rng)
    sa, sb = ddpm_schedule_sqrt_ab(t_max, device, dtype)
    broadcast_shape = (b,) + (1,) * (x0.dim() - 1)
    sqrt_ab = sa[t].reshape(broadcast_shape)
    sqrt_omb = sb[t].reshape(broadcast_shape)
    x_noisy = sqrt_ab * x0 + sqrt_omb * eps
    pred = model(x_noisy, t)
    return nn.functional.mse_loss(pred, eps)


def diffusion_noise_mse_eval_mean(
    model: nn.Module,
    x0: torch.Tensor,
    *,
    timesteps: int,
    num_noise_draws: int = 2,
) -> float:
    """Average MSE over a few random (t, ε) draws (eval mode)."""
    model.eval()
    total = 0.0
    n = max(1, int(num_noise_draws))
    with torch.no_grad():
        for _ in range(n):
            loss = diffusion_noise_mse_loss(model, x0, None, timesteps=timesteps)
            total += float(loss.item())
    return total / n
