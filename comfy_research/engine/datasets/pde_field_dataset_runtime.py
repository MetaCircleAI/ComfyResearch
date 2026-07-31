"""Synthetic 2D scalar-field PDE trajectories for regression (flattened space-time windows).

Shapes per sample: input and target are vectors of length T*C*H*W where T=context frames,
C=channels, H=W=grid_size. Input stacks u_0..u_{T-1}; target stacks u_1..u_T (Euler-marched fields).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException

from comfy_research.schemas.graph import NodeKind


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


def pde_field_flat_dims(dd: dict[str, Any]) -> tuple[int, int, int, int]:
    """Returns (context_frames, channels, grid_size, flat_dim)."""
    t_ctx = max(1, _scalar_int(dd.get("contextFrames"), 4))
    c = max(1, _scalar_int(dd.get("channels"), 1))
    g = max(4, _scalar_int(dd.get("gridSize"), 16))
    flat = t_ctx * c * g * g
    return t_ctx, c, g, flat


def _laplacian_periodic(u: np.ndarray) -> np.ndarray:
    """u: (C, H, W) periodic Laplacian via nearest-neighbor rolls."""
    return (
        np.roll(u, -1, axis=-1)
        + np.roll(u, 1, axis=-1)
        + np.roll(u, -1, axis=-2)
        + np.roll(u, 1, axis=-2)
        - 4.0 * u
    )


def _enforce_common_train_test_geometry(dd_train: dict[str, Any], dd_test: dict[str, Any]) -> None:
    t_tr, c_tr, g_tr, _ = pde_field_flat_dims(dd_train)
    t_te, c_te, g_te, _ = pde_field_flat_dims(dd_test)
    if (t_tr, c_tr, g_tr) != (t_te, c_te, g_te):
        raise HTTPException(
            status_code=400,
            detail="Train/test PDE field datasets must match contextFrames, channels, and gridSize.",
        )


def _simulate_episode(
    rng: np.random.Generator,
    *,
    mode: str,
    channels: int,
    grid: int,
    context_frames: int,
    warmup_steps: int,
    dt: float,
    diffusion_coeff: float,
    reaction_rate: float,
    velocity_x: float,
    velocity_y: float,
    ic_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One trajectory: x shaped (T, C, H, W), y shaped (T, C, H, W) one-step-ahead."""
    c, h, w = channels, grid, grid
    u = rng.standard_normal((c, h, w)).astype(np.float32) * float(ic_scale)

    def step_diffusion(cur: np.ndarray) -> np.ndarray:
        return cur + float(dt) * float(diffusion_coeff) * _laplacian_periodic(cur)

    def step_rd(cur: np.ndarray) -> np.ndarray:
        lap = _laplacian_periodic(cur)
        react = float(reaction_rate) * cur * (1.0 - cur)
        return cur + float(dt) * (float(diffusion_coeff) * lap + react)

    def step_advection(cur: np.ndarray) -> np.ndarray:
        ddx = 0.5 * (np.roll(cur, -1, axis=-1) - np.roll(cur, 1, axis=-1))
        ddy = 0.5 * (np.roll(cur, -1, axis=-2) - np.roll(cur, 1, axis=-2))
        return cur - float(dt) * (float(velocity_x) * ddx + float(velocity_y) * ddy)

    if mode == "diffusion":
        step_fn = step_diffusion
    elif mode == "reaction_diffusion":
        step_fn = step_rd
    elif mode == "advection":
        step_fn = step_advection
    else:
        raise HTTPException(status_code=500, detail=f"Internal: unknown PDE mode {mode!r}")

    for _ in range(max(0, int(warmup_steps))):
        u = step_fn(u)

    frames: list[np.ndarray] = []
    for i in range(int(context_frames) + 1):
        frames.append(np.array(u, copy=True))
        if i < int(context_frames):
            u = step_fn(u)
    stack = np.stack(frames, axis=0).astype(np.float32, copy=False)
    x_win = stack[:-1]
    y_win = stack[1:]
    return x_win, y_win


def build_pde_field_arrays(
    ds_kind: NodeKind,
    rng: np.random.Generator,
    dd_train: dict[str, Any],
    dd_test: dict[str, Any],
    train_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    mode_map = {
        NodeKind.diffusion_pde_dataset: "diffusion",
        NodeKind.reaction_diffusion_dataset: "reaction_diffusion",
        NodeKind.advection_dataset: "advection",
    }
    mode = mode_map.get(ds_kind)
    if mode is None:
        raise HTTPException(status_code=400, detail=f"Not a PDE field dataset kind: {ds_kind}")

    t_ctx, c_ch, grid, flat_dim = pde_field_flat_dims(dd_train)
    warmup_tr = max(0, _scalar_int(dd_train.get("warmupSteps"), 40))
    dt_tr = _scalar_float(dd_train.get("dt"), 0.05)
    d_tr = _scalar_float(dd_train.get("diffusionCoeff"), 0.2)
    r_tr = _scalar_float(dd_train.get("reactionRate"), 1.0)
    vx_tr = _scalar_float(dd_train.get("velocityX"), 0.5)
    vy_tr = _scalar_float(dd_train.get("velocityY"), 0.2)
    ic_tr = _scalar_float(dd_train.get("icScale"), 0.5)

    def _batch(n: int, dd: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            z = np.zeros((0, flat_dim), dtype=np.float32)
            return z, z
        g = np.random.default_rng(_scalar_int(dd.get("initSeed"), 0))
        warmup = max(0, _scalar_int(dd.get("warmupSteps"), warmup_tr))
        dt = _scalar_float(dd.get("dt"), dt_tr)
        d_coef = _scalar_float(dd.get("diffusionCoeff"), d_tr)
        react = _scalar_float(dd.get("reactionRate"), r_tr)
        vx = _scalar_float(dd.get("velocityX"), vx_tr)
        vy = _scalar_float(dd.get("velocityY"), vy_tr)
        ic_scale = _scalar_float(dd.get("icScale"), ic_tr)

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for _ in range(n):
            x_w, y_w = _simulate_episode(
                g,
                mode=mode,
                channels=c_ch,
                grid=grid,
                context_frames=t_ctx,
                warmup_steps=warmup,
                dt=dt,
                diffusion_coeff=d_coef,
                reaction_rate=react,
                velocity_x=vx,
                velocity_y=vy,
                ic_scale=ic_scale,
            )
            xs.append(x_w.reshape(-1).astype(np.float32, copy=False))
            ys.append(y_w.reshape(-1).astype(np.float32, copy=False))
        return np.stack(xs, axis=0), np.stack(ys, axis=0)

    x_np, y_np = _batch(train_size, dd_train)
    x_test_np: np.ndarray | None = None
    y_test_np: np.ndarray | None = None
    if test_size > 0:
        _enforce_common_train_test_geometry(dd_train, dd_test)
        x_test_np, y_test_np = _batch(test_size, dd_test)
    return x_np, y_np, x_test_np, y_test_np
