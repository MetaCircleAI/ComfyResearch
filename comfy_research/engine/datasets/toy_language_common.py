"""Shared helpers for toy language dataset nodes (scalar coercion, LM slicing)."""

from __future__ import annotations

from typing import Any

import numpy as np


def scalar_int(x: Any, default: int) -> int:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def scalar_float(x: Any, default: float) -> float:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def scalar_str(x: Any, default: str) -> str:
    if isinstance(x, list):
        if not x:
            return default
        x = x[0]
    if x is None:
        return default
    return str(x)


def dataset_rng_seed(data: dict[str, Any]) -> int:
    return scalar_int(data.get("seed"), scalar_int(data.get("initSeed"), 0))


def slice_last_token_lm(seq: np.ndarray, context_length: int) -> tuple[np.ndarray, np.ndarray]:
    """seq: shape [n, L+1] int64 — split into ``x[:, :L]``, ``y[:, L]`` (``y`` rank-1).

    Legacy packaging: **x** is the first ``L`` tokens and **y** is only the final next token.
    Toy language datasets use :func:`slice_shifted_window_lm` instead (per-position ``y``).
    Still used where a single scalar target per row is intentional (e.g. circle random walk).
    """
    if seq.ndim != 2 or seq.shape[1] != context_length + 1:
        raise ValueError(f"expected seq shape [n, {context_length + 1}], got {seq.shape}")
    x = seq[:, :context_length].copy()
    y = seq[:, context_length].copy()
    return x, y


def slice_shifted_window_lm(seq: np.ndarray, context_length: int) -> tuple[np.ndarray, np.ndarray]:
    """seq: shape ``[n, L+1]`` int64 — causal next-token LM targets aligned with **x** ``[n, L]``.

    Returns **x** = ``seq[:, :-1]`` (first ``L`` tokens) and **y** = ``seq[:, 1:]`` (shape ``[n, L]``),
    so ``y[:, i]`` is the token immediately following ``x[:, i]`` in the same ``(L+1)``-length window.
    """
    if seq.ndim != 2 or int(seq.shape[1]) != int(context_length) + 1:
        raise ValueError(f"expected seq shape [n, {int(context_length) + 1}], got {seq.shape}")
    x = seq[:, :-1].copy()
    y = seq[:, 1:].copy()
    return x, y


def resize_sequence(tokens: list[int], target_len: int, rng: np.random.Generator, vocab_size: int) -> np.ndarray:
    """Pad with random terminals or truncate to ``target_len``."""
    v = max(2, int(vocab_size))
    if len(tokens) >= target_len:
        return np.asarray(tokens[:target_len], dtype=np.int64)
    out = list(tokens)
    while len(out) < target_len:
        out.append(int(rng.integers(0, v)))
    return np.asarray(out[:target_len], dtype=np.int64)
