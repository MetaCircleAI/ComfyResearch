"""Small dense-classification datasets used by paper-reproduction templates.

The builders deliberately avoid modular-addition/Grokking benchmarks.  They
cover the fixed and scalable orthogonal Rank-Collapse toys, the legacy
repeated-prototype controls, and the synthetic classification setting from the
Slingshot paper.
"""
from __future__ import annotations

import math

import numpy as np


FIGURE5_LABELS = np.asarray((0, 0, 1, 1, 2, 3), dtype=np.int64)
FIGURE5_UNIFORM_LABELS = np.asarray((0, 0, 1, 1, 2, 2, 3, 3), dtype=np.int64)


def rank_figure5_arrays(mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed orthogonal samples used by Figure 5 and its controls.

    The paper protocol specifies an orthogonal sample matrix.  Choosing the
    canonical basis makes the feature matrix produced by the first Linear
    factor explicit: for ``S = I``, ``F = S @ theta = theta``.  The rows are
    intentionally kept in paper order (no shuffle and no held-out split).
    """
    if mode == "rank_figure5_main":
        labels = FIGURE5_LABELS
        mse = False
    elif mode == "rank_figure5_uniform":
        labels = FIGURE5_UNIFORM_LABELS
        mse = False
    elif mode == "rank_figure5_mse":
        labels = FIGURE5_LABELS
        mse = True
    else:
        raise ValueError(f"unknown Rank Figure-5 mode {mode!r}")

    x = np.eye(len(labels), dtype=np.float32)
    y = (
        np.eye(4, dtype=np.float32)[labels]
        if mse
        else labels.copy()
    )
    return x, y


def balanced_class_counts(sample_count: int, class_count: int) -> np.ndarray:
    """Return deterministic near-uniform counts summing to ``sample_count``."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    counts = np.full(class_count, sample_count // class_count, dtype=np.int64)
    counts[: sample_count % class_count] += 1
    return counts


def geometric_class_counts(sample_count: int, class_count: int, ratio: float) -> np.ndarray:
    """Integer geometric class frequencies with a one-example floor.

    ``(1029, 16, 2)`` gives exactly
    ``[512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1, 1, 1, 1, 1, 1]``.
    """
    if sample_count < class_count:
        raise ValueError("rank datasets require at least one example per class")
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if not math.isfinite(ratio) or ratio < 1.0:
        raise ValueError("frequencyRatio must be finite and at least 1")
    if ratio == 1.0:
        return balanced_class_counts(sample_count, class_count)

    def from_head(head: int) -> np.ndarray:
        return np.asarray(
            [max(1, int(math.floor(head / (ratio**i)))) for i in range(class_count)],
            dtype=np.int64,
        )

    lo, hi = 1, sample_count
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if int(from_head(mid).sum()) <= sample_count:
            lo = mid
        else:
            hi = mid - 1
    counts = from_head(lo)
    remainder = sample_count - int(counts.sum())
    for i in range(remainder):
        counts[i % class_count] += 1
    return counts


def rank_orthogonal_scaleup_arrays(
    *,
    sample_count: int,
    class_count: int,
    frequency_ratio: float,
    mse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a scalable all-orthogonal Rank-Collapse training set.

    Every sample gets its own canonical coordinate, so ``X = I_N`` and the
    rows remain orthogonal even when several samples share a class.  Class
    labels follow :func:`geometric_class_counts`; consequently a ratio of one
    is the uniform-frequency control.  The caller is responsible for enforcing
    the graph-level protocol ``inputDim=trainSize`` and ``testSize=0``.
    """
    counts = geometric_class_counts(sample_count, class_count, frequency_ratio)
    labels = np.repeat(np.arange(class_count, dtype=np.int64), counts)
    x = np.eye(sample_count, dtype=np.float32)
    y = np.eye(class_count, dtype=np.float32)[labels] if mse else labels
    return x, y


def rank_classification_arrays(
    rng: np.random.Generator,
    *,
    sample_count: int,
    input_dim: int,
    class_count: int,
    frequency_ratio: float,
    uniform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat orthogonal one-hot prototypes according to class frequencies."""
    if input_dim < class_count:
        raise ValueError("rank datasets require inputDim >= outputDim for orthogonal prototypes")
    counts = (
        balanced_class_counts(sample_count, class_count)
        if uniform
        else geometric_class_counts(sample_count, class_count, frequency_ratio)
    )
    labels = np.repeat(np.arange(class_count, dtype=np.int64), counts)
    x = np.zeros((sample_count, input_dim), dtype=np.float32)
    x[np.arange(sample_count), labels] = 1.0
    permutation = rng.permutation(sample_count)
    return x[permutation], labels[permutation]


def slingshot_hypercube_splits(
    rng: np.random.Generator,
    *,
    train_size: int,
    test_size: int,
    input_dim: int,
    class_count: int,
    class_separation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Build the TMLR 2024 Appendix A.9.1 synthetic classification task.

    The task has 128 dimensions in the paper-backed template, three informative
    coordinates, and eight cube-vertex classes.  Features are jointly
    standardized across the deterministic 256+256 split.
    """
    if input_dim < 3:
        raise ValueError("slingshot_hypercube requires inputDim >= 3")
    if class_count != 8:
        raise ValueError("slingshot_hypercube reproduces the paper's fixed 8 classes")
    if not math.isfinite(class_separation) or class_separation < 0:
        raise ValueError("classSeparation must be finite and non-negative")
    total = train_size + test_size
    if total <= 0:
        raise ValueError("slingshot_hypercube requires at least one sample")
    labels = np.tile(np.arange(class_count, dtype=np.int64), math.ceil(total / class_count))[:total]
    rng.shuffle(labels)
    x = rng.normal(0.0, 1.0, size=(total, input_dim)).astype(np.float32)
    bits = ((labels[:, None] >> np.arange(3, dtype=np.int64)) & 1).astype(np.float32)
    x[:, :3] += (2.0 * bits - 1.0) * np.float32(class_separation)
    x = ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)).astype(
        np.float32
    )
    x_train = x[:train_size]
    y_train = labels[:train_size]
    if test_size == 0:
        return x_train, y_train, None, None
    return x_train, y_train, x[train_size:], labels[train_size:]
