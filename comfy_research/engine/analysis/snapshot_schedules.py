"""Reusable measurement schedules for paper-reproduction observables."""

from __future__ import annotations

import numpy as np


def idnns_epoch_snapshots(
    epochs: int = 10_000,
    *,
    samples: int = 1800,
    start_sample: int = 1,
) -> np.ndarray:
    """Mirror the released IDNNs base-2 logspace epoch construction."""
    if epochs < 1 or samples < 1 or start_sample < 1 or start_sample > epochs:
        raise ValueError("invalid IDNNs epoch snapshot configuration")
    indexes = np.unique(
        np.logspace(
            np.log2(start_sample),
            np.log2(epochs),
            samples,
            dtype=int,
            base=2,
        )
    ) - 1
    return indexes[(indexes >= 0) & (indexes < epochs)].astype(np.int64)
