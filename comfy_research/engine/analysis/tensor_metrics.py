"""Shared numeric metrics on tensors (used by training, APIs, and user observables)."""

from __future__ import annotations

import numpy as np


def effective_rank_from_matrix(mat: np.ndarray, eps: float = 1e-12) -> float:
    """Entropy-based effective rank from singular values (Shannon entropy of normalized singular values)."""
    h = singular_value_entropy(mat, eps=eps)
    if not np.isfinite(h):
        return 1.0
    return float(np.exp(h))


def singular_value_entropy(mat: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of normalized singular values (2D matrix or flattened to samples × features)."""
    if mat.ndim == 1:
        x = mat.reshape(1, -1)
    else:
        x = mat.reshape(mat.shape[0], -1)
    if x.size == 0:
        return float("nan")
    try:
        s = np.linalg.svd(x, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return float("nan")
    tot = float(s.sum())
    if tot <= eps:
        return float("nan")
    p = s / tot
    p = p[p > eps]
    if p.size == 0:
        return float("nan")
    return float(-np.sum(p * np.log(p)))
