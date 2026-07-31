"""JSON helpers: standard json cannot encode NaN/Infinity."""

from __future__ import annotations

import math
from typing import Any


def finite_float(x: float, *, default: float | None = None) -> float | None:
    """Return x if finite, otherwise default (None)."""
    if not isinstance(x, (int, float)):
        return x  # type: ignore[return-value]
    val = float(x)
    if math.isfinite(val):
        return val
    return default


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for JSON serialization."""
    if isinstance(obj, float):
        return finite_float(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
