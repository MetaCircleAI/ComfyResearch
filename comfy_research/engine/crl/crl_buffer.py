"""HER-style future goal relabeling (``flatten_crl_fn``) in PyTorch — see ``crl_port_map``."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


def flatten_crl_torch(
    obs: torch.Tensor,
    act: torch.Tensor,
    *,
    gamma: float,
    obs_dim: int,
    goal_start: int,
    goal_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match scaling-crl ``TrajectoryUniformSamplingQueue.flatten_crl_fn`` (same-seed trajectory mask).

    ``obs`` is ``[B, L, D]`` (full env vector including current goal tail). ``act`` is ``[B, L-1, A]``.
    Returns ``(new_obs, act)`` with ``new_obs`` ``[B, L-1, D]`` = concat(state, resampled goal).
    """
    if obs.dim() != 3 or act.dim() != 3:
        raise ValueError("obs must be [B,L,D], act must be [B,L-1,A]")
    b, ell, _d = obs.shape
    if act.shape[0] != b or act.shape[1] != ell - 1:
        raise ValueError("act length must be L-1")
    device = obs.device
    rows = torch.arange(ell, device=device, dtype=torch.float32).view(1, ell, 1)
    cols = torch.arange(ell, device=device, dtype=torch.float32).view(1, 1, ell)
    valid = (cols > rows).float()
    disc = (gamma ** (cols - rows).clamp(min=0.0)).float()
    probs = valid * disc + torch.eye(ell, device=device, dtype=torch.float32).unsqueeze(0) * 1e-5
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    logits = torch.log(probs.clamp(min=1e-12))
    # sample future index for each of first L-1 rows
    cat = torch.distributions.Categorical(logits=logits[:, :-1, :])
    gi = cat.sample()  # [B, L-1]
    t_idx = torch.arange(ell - 1, device=device).unsqueeze(0).expand(b, -1)
    b_idx = torch.arange(b, device=device).unsqueeze(1).expand(-1, ell - 1)
    future_full = obs[b_idx, gi]
    goal = future_full[..., goal_start:goal_end]
    state_part = obs[:, :-1, :obs_dim]
    new_obs = torch.cat([state_part, goal], dim=-1)
    return new_obs, act


@dataclass
class TrajectoryChunk:
    obs: np.ndarray  # [T+1, N, D]
    act: np.ndarray  # [T, N, A]
    reward: np.ndarray  # [T, N]
    discount: np.ndarray  # [T, N]
    seed: np.ndarray  # [T+1, N] int32 episode ids


class TrajectoryReplay:
    """FIFO list of fixed-length rollout chunks (``unroll_length`` matches ``obs`` time axis − 1)."""

    def __init__(self, max_chunks: int = 500) -> None:
        self.max_chunks = max_chunks
        self.chunks: list[TrajectoryChunk] = []

    def insert(self, chunk: TrajectoryChunk) -> None:
        self.chunks.append(chunk)
        if len(self.chunks) > self.max_chunks:
            self.chunks.pop(0)

    def __len__(self) -> int:
        return len(self.chunks)

    def sample_windows(
        self,
        rng: random.Random,
        *,
        batch_size: int,
        unroll: int,
    ) -> TrajectoryChunk | None:
        """Stack ``batch_size`` random (chunk, env, time) windows of length ``unroll+1`` / ``unroll``."""
        if not self.chunks:
            return None
        obs_w: list[np.ndarray] = []
        act_w: list[np.ndarray] = []
        rew_w: list[np.ndarray] = []
        disc_w: list[np.ndarray] = []
        seed_w: list[np.ndarray] = []
        for _ in range(batch_size):
            c = rng.choice(self.chunks)
            t_max = c.obs.shape[0] - 1 - unroll
            if t_max < 0:
                continue
            t0 = rng.randint(0, t_max)
            ei = rng.randrange(0, c.obs.shape[1])
            obs_seg = c.obs[t0 : t0 + unroll + 1, ei : ei + 1, :].squeeze(1)  # [U+1, D]
            act_seg = c.act[t0 : t0 + unroll, ei : ei + 1, :].squeeze(1)
            rew_seg = c.reward[t0 : t0 + unroll, ei]
            disc_seg = c.discount[t0 : t0 + unroll, ei]
            seed_seg = c.seed[t0 : t0 + unroll + 1, ei]
            obs_w.append(obs_seg)
            act_w.append(act_seg)
            rew_w.append(rew_seg)
            disc_w.append(disc_seg)
            seed_w.append(seed_seg)
        if len(obs_w) < batch_size:
            return None
        obs_b = np.stack(obs_w, axis=0)  # [B, U+1, D]
        act_b = np.stack(act_w, axis=0)
        rew_b = np.stack(rew_w, axis=0)
        disc_b = np.stack(disc_w, axis=0)
        seed_b = np.stack(seed_w, axis=0)
        return TrajectoryChunk(
            obs=np.transpose(obs_b, (1, 0, 2)),
            act=np.transpose(act_b, (1, 0, 2)),
            reward=np.transpose(rew_b, (1, 0)),
            discount=np.transpose(disc_b, (1, 0)),
            seed=np.transpose(seed_b, (1, 0)),
        )
