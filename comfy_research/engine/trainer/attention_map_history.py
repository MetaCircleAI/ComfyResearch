"""Validation and bounded storage for attention-map observable frames."""
from __future__ import annotations

from itertools import product
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException

MAX_ATTENTION_MAP_SLICES_PER_FRAME = 20
MAX_ATTENTION_MAP_FRAMES = 50
MAX_ATTENTION_MAP_DIM = 25


def selected_indices(node_id: str, data: dict[str, Any], key: str) -> list[int]:
    raw = data.get(key, 0)
    values = raw if isinstance(raw, (list, tuple)) else str(raw).replace(",", " ").split()
    try:
        if any(isinstance(value, bool) or str(value).strip() != str(int(str(value).strip())) for value in values):
            raise ValueError
        selected = sorted({int(value) for value in values})
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, f"Attention map node {node_id}: {key} must be non-negative integers.") from exc
    if not selected or any(value < 0 for value in selected):
        raise HTTPException(400, f"Attention map node {node_id}: {key} must be a non-empty list of non-negative integers.")
    return selected


def selected_tuples(node_id: str, data: dict[str, Any]) -> list[tuple[int, int, int]]:
    layers = selected_indices(node_id, data, "attentionLayerIndices")
    batches = selected_indices(node_id, data, "attentionBatchIndices")
    heads = selected_indices(node_id, data, "attentionHeadIndices")
    count = len(layers) * len(batches) * len(heads)
    if count > MAX_ATTENTION_MAP_SLICES_PER_FRAME:
        raise HTTPException(
            400,
            f"Attention map node {node_id}: {len(layers)} layers x {len(batches)} batches x {len(heads)} heads = {count}; limit is {MAX_ATTENTION_MAP_SLICES_PER_FRAME} slices per log tick.",
        )
    return list(product(layers, batches, heads))


def validate_tuple(node_id: str, layer: int, batch: int, head: int, layers: list[Any]) -> Any:
    if layer >= len(layers):
        raise HTTPException(400, f"Attention map node {node_id}: layer {layer} is out of range [0, {len(layers) - 1}].")
    tensor = layers[layer]
    if getattr(tensor, "ndim", 0) != 4:
        raise HTTPException(400, f"Attention map node {node_id}: selected model does not expose [batch, head, query, key] attention maps.")
    batch_count, head_count = int(tensor.shape[0]), int(tensor.shape[1])
    if batch >= batch_count:
        raise HTTPException(400, f"Attention map node {node_id}: batch {batch} is out of range [0, {batch_count - 1}].")
    if head >= head_count:
        raise HTTPException(400, f"Attention map node {node_id}: head {head} is out of range [0, {head_count - 1}].")
    return tensor[batch, head]


def crop_map(attention_map: Any) -> tuple[list[list[float]], list[int], int, int]:
    rows, cols = int(attention_map.shape[0]), int(attention_map.shape[1])
    row_start, col_start = max(0, rows - MAX_ATTENTION_MAP_DIM), max(0, cols - MAX_ATTENTION_MAP_DIM)
    return attention_map[row_start:, col_start:].detach().float().cpu().tolist(), [rows, cols], row_start, col_start


def append_frame(history: list[dict[str, Any]], frame: dict[str, Any]) -> None:
    history.append(frame)
    del history[:-MAX_ATTENTION_MAP_FRAMES]


def sanitize_history(raw: Any) -> list[dict[str, Any]]:
    """Keep only persisted frames that satisfy the bounded transport contract."""
    if not isinstance(raw, list):
        return []
    clean: list[dict[str, Any]] = []
    for frame in raw[-MAX_ATTENTION_MAP_FRAMES:]:
        if not isinstance(frame, Mapping) or not isinstance(frame.get("step"), int) or not isinstance(frame.get("slices"), list):
            continue
        slices = [slice_ for slice_ in frame["slices"][:MAX_ATTENTION_MAP_SLICES_PER_FRAME] if isinstance(slice_, Mapping)]
        clean.append({"step": frame["step"], "slices": [dict(slice_) for slice_ in slices]})
    return clean
