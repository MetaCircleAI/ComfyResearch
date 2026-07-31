"""Deterministic no-replacement orders shared by graph and paper runtimes."""

from __future__ import annotations

import math

import numpy as np


def seedsequence_epoch_permutation(
    count: int,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Return the exact NumPy SeedSequence permutation for one data epoch."""
    n = int(count)
    if n < 1:
        raise ValueError("count must be >= 1")
    sequence = np.random.SeedSequence([int(seed), int(epoch)])
    return np.random.default_rng(sequence).permutation(n).astype(np.int64, copy=False)


def affine_epoch_positions(
    rng: np.random.Generator,
    *,
    count: int,
    repeats: int = 1,
) -> np.ndarray:
    """Return independent affine permutations ``(a*i+b) mod count``."""
    n = int(count)
    r = int(repeats)
    if n < 1:
        raise ValueError("count must be >= 1")
    if r < 1:
        raise ValueError("repeats must be >= 1")
    if n == 1:
        return np.zeros((r, 1), dtype=np.int64)

    coprimes = np.asarray(
        [value for value in range(1, n) if math.gcd(value, n) == 1],
        dtype=np.int64,
    )
    multipliers = np.asarray(rng.choice(coprimes, size=r), dtype=np.int64)
    offsets = np.asarray(rng.integers(0, n, size=r), dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)
    return (multipliers[:, None] * positions[None, :] + offsets[:, None]) % n
