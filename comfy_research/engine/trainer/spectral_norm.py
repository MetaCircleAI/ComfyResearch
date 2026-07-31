"""Deterministic, measurement-only matrix spectral-norm estimators.

These helpers deliberately never read or advance PyTorch's global RNG state.
Observables are instrumentation: enabling them must not change a stochastic
training run.
"""
from __future__ import annotations

import torch


def _start_vector(matrix: torch.Tensor) -> torch.Tensor:
    """A deterministic non-zero vector on the matrix's device and dtype."""
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("weight must be a non-empty matrix")
    # A fixed analytic vector avoids both CPU/CUDA global RNG consumption and
    # generator-device compatibility pitfalls.
    return torch.arange(1, int(matrix.shape[1]) + 1, device=matrix.device, dtype=matrix.dtype).reshape(-1, 1)


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector).clamp_min(torch.finfo(vector.dtype).tiny)


def singular_norm_power_estimate(weight: torch.Tensor, *, iterations: int = 10) -> torch.Tensor:
    """Conventional top singular-value estimate using power iteration on WᵀW."""
    matrix = weight.detach()
    vector = _unit(_start_vector(matrix))
    for _ in range(max(1, int(iterations))):
        vector = _unit(matrix.T @ (matrix @ vector))
    return torch.linalg.vector_norm(matrix @ vector)


def author_figure1_power_estimate(
    weight: torch.Tensor,
    *,
    iterations: int = 10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Released Rahaman Figure 1 notebook estimator, with deterministic start.

    It is intentionally *not* the mathematical spectral norm.  The unusual
    final expression preserves the notebook's operator precedence so plots are
    comparable. Vector-shaped first/last layers use their exact operator norm.
    """
    matrix = weight.detach()
    if matrix.ndim != 2:
        raise ValueError("weight must be a matrix")
    if 1 in (int(matrix.shape[0]), int(matrix.shape[1])):
        return torch.linalg.vector_norm(matrix)
    if int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError("author_figure1 is defined only for square matrices")
    vector = (
        _start_vector(matrix)
        if generator is None
        else torch.randn(
            (int(matrix.shape[1]), 1),
            dtype=matrix.dtype,
            device=matrix.device,
            generator=generator,
        )
    )
    for _ in range(max(1, int(iterations))):
        vector = _unit(matrix @ vector)
    return (((vector.T @ matrix @ vector) / vector.T) @ vector).squeeze().abs()
