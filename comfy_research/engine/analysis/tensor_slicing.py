from __future__ import annotations

from typing import Any

import numpy as np


def parse_slice_indices(raw: str) -> list[int] | None:
    parts = [p.strip() for p in str(raw).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except (TypeError, ValueError):
            return None
        out.append(v)
    return out


def resolve_slice_index(i: int, axis_len: int) -> int | None:
    """Map a user index on an axis of length ``axis_len`` to ``[0, axis_len)``.

    Non-negative indices are used as-is; negative indices count from the end
    (``-1`` is last), matching Python/NumPy semantics.
    """
    if axis_len < 1:
        return None
    if i >= 0:
        return i if i < axis_len else None
    j = axis_len + i
    return j if j >= 0 else None


def resolve_slice_indices(indices: list[int], axis_len: int) -> list[int] | None:
    out: list[int] = []
    for i in indices:
        r = resolve_slice_index(i, axis_len)
        if r is None:
            return None
        out.append(r)
    return out


def normalize_slices(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            dim = int(entry.get("dimension"))
        except (TypeError, ValueError):
            continue
        indices = str(entry.get("indices", "")).strip()
        out.append({"dimension": dim, "indices": indices})
    return out


def apply_tensor_slicing_specs(arr: np.ndarray, raw_specs: Any) -> np.ndarray:
    cur = np.asarray(arr)
    specs = normalize_slices(raw_specs)
    for spec in specs:
        dim = int(spec["dimension"])
        indices_s = str(spec["indices"]).strip()
        rank = int(cur.ndim)
        if dim < 0 or dim >= rank:
            raise ValueError(f"slice dimension {dim} out of range for rank {rank}")
        if not indices_s:
            # Empty index list means "all indices" for this row.
            continue
        axis_len = int(cur.shape[dim])
        if axis_len < 1:
            raise ValueError(f"slice dimension {dim} has empty axis")
        idx = parse_slice_indices(indices_s)
        if idx is None:
            raise ValueError(f"invalid index list {indices_s!r}")
        idx_r = resolve_slice_indices(idx, axis_len)
        if idx_r is None:
            raise ValueError(
                f"slice indices {idx!r} out of range for axis length {axis_len} on dimension {dim}"
            )
        if len(idx_r) == 1:
            cur = np.take(cur, indices=idx_r[0], axis=dim)
        else:
            cur = np.take(cur, indices=np.asarray(idx_r, dtype=np.int64), axis=dim)
    return cur
