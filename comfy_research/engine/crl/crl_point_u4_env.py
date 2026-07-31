"""Lightweight continuous 2D particle in the U4 maze layout from scaling-crl ``envs/ant_maze.py``.

Topology matches ``U4_MAZE`` (``G`` = possible goals, ``R`` = reset region). Full Ant dynamics
require Brax/JAX (see ``crl_port_map``); this env keeps obs/action sizes small for CPU demos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Same layout as scaling-crl envs/ant_maze.py (symbol meanings: 1=wall, R=reset, G=goal)
U4_MAZE = [
    [1, 1, 1, 1, 1],
    [1, 2, 2, 2, 1],  # 2 = G in original; use 2 for goal cells
    [1, 3, 1, 2, 1],  # 3 = R
    [1, 1, 1, 2, 1],
    [1, 2, 1, 2, 1],
    [1, 2, 2, 2, 1],
    [1, 1, 1, 1, 1],
]
RESET = 3
GOAL = 2
WALL = 1


def _cell_centers(layout: list[list[int]], scale: float) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]]]:
    """Return wall AABBs (cx,cy,half), free mask centers, reset and goal positions."""
    h, w = len(layout), len(layout[0])
    walls: list[tuple[float, float, float, float]] = []
    resets: list[tuple[float, float]] = []
    goals: list[tuple[float, float]] = []
    for i in range(h):
        for j in range(w):
            cell = layout[i][j]
            cx = j * scale + scale * 0.5
            cy = (h - 1 - i) * scale + scale * 0.5  # flip so y grows upward visually
            if cell == WALL:
                walls.append((cx, cy, 0.5 * scale, 0.5 * scale))
            elif cell == RESET:
                resets.append((cx, cy))
            elif cell == GOAL:
                goals.append((cx, cy))
    wa = np.array(walls, dtype=np.float64) if walls else np.zeros((0, 4), dtype=np.float64)
    return wa, np.array([], dtype=np.float64), resets, goals


@dataclass
class PointU4State:
    pos: np.ndarray  # [N, 2]
    vel: np.ndarray  # [N, 2]
    goal: np.ndarray  # [N, 2]
    trunc: np.ndarray  # [N] int steps in episode
    seed: np.ndarray  # [N] episode id


class PointU4MazeEnv:
    """Vectorized particle: obs = [x, y, vx, vy, gx, gy], action = force_xy in [-1,1]."""

    def __init__(self, num_envs: int, *, scale: float = 4.0, episode_length: int = 200, rng: np.random.Generator | None = None):
        self.num_envs = int(num_envs)
        self.scale = float(scale)
        self.episode_length = int(episode_length)
        self.rng = rng or np.random.default_rng()
        self.walls, _, self._resets, self._goals = _cell_centers(U4_MAZE, self.scale)
        self.h = len(U4_MAZE)
        self.w = len(U4_MAZE[0])
        self.bounds = np.array(
            [0.5 * self.scale, 0.5 * self.scale, (self.w - 0.5) * self.scale, (self.h - 0.5) * self.scale],
            dtype=np.float64,
        )
        self.obs_dim_state = 4
        self.goal_dim = 2
        self.action_dim = 2
        self._episode_id = 0

    def _sample_spawn(self, n: int) -> np.ndarray:
        if not self._resets:
            raise RuntimeError("U4 maze has no reset cell")
        idx = self.rng.integers(0, len(self._resets), size=n)
        base = np.array([self._resets[i] for i in idx], dtype=np.float64)
        noise = self.rng.normal(0, 0.08 * self.scale, size=base.shape)
        pos = base + noise
        return np.clip(pos, self.bounds[[0, 1]], self.bounds[[2, 3]])

    def _sample_goals(self, n: int) -> np.ndarray:
        if not self._goals:
            raise RuntimeError("U4 maze has no goal cells")
        idx = self.rng.integers(0, len(self._goals), size=n)
        return np.array([self._goals[i] for i in idx], dtype=np.float64)

    def _resolve_walls(self, pos: np.ndarray) -> np.ndarray:
        """Push ``pos`` out of axis-aligned wall boxes (vectorized per env)."""
        if self.walls.shape[0] == 0:
            return pos
        out = pos.copy()
        for k in range(int(self.walls.shape[0])):
            cx, cy, hx, hy = (float(self.walls[k, j]) for j in range(4))
            dx = out[:, 0] - cx
            dy = out[:, 1] - cy
            inside = (np.abs(dx) < hx) & (np.abs(dy) < hy)
            if not np.any(inside):
                continue
            pen_x = hx - np.abs(dx)
            pen_y = hy - np.abs(dy)
            push_x = inside & (pen_x <= pen_y)
            push_y = inside & (pen_y < pen_x)
            out[push_x, 0] = cx + np.sign(dx[push_x]) * (hx + 1e-4)
            out[push_y, 1] = cy + np.sign(dy[push_y]) * (hy + 1e-4)
        out[:, 0] = np.clip(out[:, 0], self.bounds[0], self.bounds[2])
        out[:, 1] = np.clip(out[:, 1], self.bounds[1], self.bounds[3])
        return out

    def reset(self) -> PointU4State:
        self._episode_id += 1
        pos = self._sample_spawn(self.num_envs)
        vel = self.rng.normal(0, 0.01, size=(self.num_envs, 2)).astype(np.float64)
        goal = self._sample_goals(self.num_envs)
        trunc = np.zeros(self.num_envs, dtype=np.int32)
        seed = np.full(self.num_envs, self._episode_id, dtype=np.int32)
        return PointU4State(pos=pos, vel=vel, goal=goal, trunc=trunc, seed=seed)

    def _obs(self, s: PointU4State) -> np.ndarray:
        return np.concatenate([s.pos, s.vel, s.goal], axis=-1).astype(np.float32)

    def observe(self, s: PointU4State) -> np.ndarray:
        return self._obs(s)

    def step(self, s: PointU4State, action: np.ndarray) -> tuple[PointU4State, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (next_state, obs, reward, discount). action float32 [N,2]."""
        a = np.clip(action.astype(np.float64), -1.0, 1.0)
        dt = 0.05
        force_scale = 12.0 * self.scale
        damp = 0.08
        s = PointU4State(
            pos=s.pos.copy(),
            vel=s.vel.copy(),
            goal=s.goal.copy(),
            trunc=s.trunc.copy(),
            seed=s.seed.copy(),
        )
        s.vel = s.vel * (1.0 - damp) + a * force_scale * dt
        s.pos = s.pos + s.vel * dt
        s.pos = self._resolve_walls(s.pos)
        dist = np.linalg.norm(s.pos - s.goal, axis=-1)
        reward = -dist.astype(np.float64)
        success = (dist < 0.35 * self.scale).astype(np.float64)
        s.trunc = s.trunc + 1
        done = (s.trunc >= self.episode_length).astype(np.float64)
        discount = 1.0 - done
        # on done, soft reset positions (keep same goal for HER continuity like truncated episodes)
        reset_mask = done > 0.5
        if np.any(reset_mask):
            s.pos[reset_mask] = self._sample_spawn(int(reset_mask.sum()))
            s.vel[reset_mask] = 0.0
            s.goal[reset_mask] = self._sample_goals(int(reset_mask.sum()))
            s.trunc[reset_mask] = 0
            s.seed[reset_mask] = self._episode_id
            self._episode_id += 1
        obs = self._obs(s)
        return s, obs.astype(np.float32), reward.astype(np.float32), discount.astype(np.float32)
