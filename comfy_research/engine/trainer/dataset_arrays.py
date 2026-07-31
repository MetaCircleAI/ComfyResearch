"""Narrow handoff between dataset materialization and model construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DatasetArrays:
    x_np: np.ndarray
    y_np: np.ndarray
    x_test_np: np.ndarray | None
    y_test_np: np.ndarray | None
    input_dim: int
    output_dim: int
    extras: dict[str, Any] = field(default_factory=dict)  # per-family extras, keep small
