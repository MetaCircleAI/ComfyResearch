"""Runtime helpers for ``random_input_distribution`` graph nodes.

The canvas wires this node into ``teacher_dataset`` (and the trainer resolves it via edges).
Sampling always uses a NumPy ``Generator`` constructed from the node JSON ``seed`` — that is
the "random generator" carried by the distribution node in the backend (there is no separate
Python object passed on the wire; the graph edge selects which node dict seeds ``rng``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import HTTPException


def scalar_int(x: Any, default: int = 0) -> int:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return int(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def scalar_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return float(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


def rng_from_random_input_distribution_data(dd_rid: dict[str, Any]) -> np.random.Generator:
    """``np.random.default_rng`` seeded from the node's ``seed`` field (list or scalar)."""
    return np.random.default_rng(scalar_int(dd_rid.get("seed"), 0))


def sample_inputs(rng: np.random.Generator, n: int, dim: int, dist: str) -> np.ndarray:
    """Draw ``n`` rows of ``dim``-dim inputs according to ``dist`` id (graph JSON values)."""
    if dim < 1:
        raise HTTPException(status_code=400, detail="Dataset input dimension must be >= 1")
    if dist == "linspace_0_1":
        if dim != 1:
            raise HTTPException(status_code=400, detail="linspace_0_1 input sampling requires inputDim=1")
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32).reshape(n, 1)
    if dist == "linspace_0_1_endpoint_excluded":
        if dim != 1:
            raise HTTPException(
                status_code=400,
                detail="linspace_0_1_endpoint_excluded input sampling requires inputDim=1",
            )
        return np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32).reshape(n, 1)
    if dist == "standard_normal":
        return rng.standard_normal((n, dim)).astype(np.float32)
    if dist == "uniform_neg1_1":
        return rng.uniform(-1.0, 1.0, size=(n, dim)).astype(np.float32)
    if dist == "uniform_0_1":
        return rng.uniform(0.0, 1.0, size=(n, dim)).astype(np.float32)
    raise HTTPException(status_code=400, detail=f"Unknown input distribution: {dist}")


def sample_x_from_random_input_dict(
    dd_rid: dict[str, Any],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``x`` with shape ``(n, input_dim)`` using stored ``inputDistribution`` / noise fields."""
    input_dim = scalar_int(dd_rid.get("inputDim"), 10)
    dist = scalar_str(dd_rid.get("inputDistribution"), "standard_normal")
    x = sample_inputs(rng, n, input_dim, dist)
    noise_dist = scalar_str(dd_rid.get("noiseDistribution"), "deterministic")
    noise_level = scalar_float(dd_rid.get("noiseLevel"), 0.0)
    additive = noise_dist == "additive_gaussian"
    if additive and noise_level > 0:
        x = x + noise_level * rng.standard_normal(x.shape).astype(np.float32)
    return x


def sample_x_from_sampler_dict(
    sampler_data: dict[str, Any],
    dd_rid: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``x`` using sampler ``numSamples`` and a wired random-input distribution config."""
    n = max(1, scalar_int(sampler_data.get("numSamples"), 800))
    g = rng if rng is not None else rng_from_random_input_distribution_data(dd_rid)
    return sample_x_from_random_input_dict(dd_rid, n, g)
