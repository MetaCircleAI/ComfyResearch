"""Evaluate a canvas-only atomic layer chain ending at tensor_constant / tensor_linspace (PCA, etc.)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import SEQUENTIAL_MODEL_TYPES, build_sequential_from_flat_atomic_chain
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.engine.runs.trainer_run import (
    _prepare_x_for_atomic_sequential,
    apply_parameter_tensor_payloads_from_atomic_chain,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _tensor_chain_predecessors(cur_id: str, edges: list[Edge], nmap: dict[str, Node]) -> list[Node]:
    preds: list[Node] = []
    for e in edges:
        if e.target != cur_id:
            continue
        th = (e.targetHandle or "").strip()
        if th not in ("tensor", "tensor_in", "in", ""):
            continue
        sh = (e.sourceHandle or "").strip()
        if sh not in ("tensor", "tensor_out", "model", ""):
            continue
        src = nmap.get(e.source)
        if src is not None:
            preds.append(src)
    return preds


_TENSOR_HEAD_TYPES = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("canvas_tensor_source")
)


def collect_canvas_atomic_chain_and_source(
    tip: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
) -> tuple[list[Node], Node]:
    """Return (ordered atomic layers front→back, tensor source node).

    ``tip`` is the canvas node wired into PCA (e.g. activation_layer); walking left must reach
    ``tensor_constant`` or ``tensor_linspace``.
    """
    if tip.type not in SEQUENTIAL_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Canvas tensor chain tip must be an atomic layer (linear, activation, LayerNorm, …).",
        )
    rev: list[Node] = []
    cur: Node | None = tip
    seen: set[str] = set()
    while cur is not None:
        if cur.id in seen:
            raise HTTPException(status_code=400, detail="Cycle detected in canvas tensor chain.")
        seen.add(cur.id)
        if cur.type not in SEQUENTIAL_MODEL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Canvas chain contains non-atomic node {cur.type!s}.",
            )
        rev.append(cur)
        preds = _tensor_chain_predecessors(cur.id, edges, nmap)
        if len(preds) > 1:
            raise HTTPException(
                status_code=400,
                detail="Each atomic layer accepts at most one incoming tensor chain link.",
            )
        nxt = preds[0] if preds else None
        if nxt is None:
            raise HTTPException(
                status_code=400,
                detail="Canvas tensor chain is not connected to a Tensor constant or Tensor linspace source.",
            )
        if nxt.type not in SEQUENTIAL_MODEL_TYPES:
            if nxt.type in _TENSOR_HEAD_TYPES:
                rev.reverse()
                return rev, nxt
            if nxt.type == NodeKind.combined_model:
                raise HTTPException(
                    status_code=400,
                    detail="PCA canvas chain cannot start from combined_model; use inner tensor outputs.",
                )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported upstream of canvas chain: {nxt.type!s}. "
                    "Use Tensor constant or Tensor linspace, then atomic layers, into PCA."
                ),
            )
        cur = nxt
    raise HTTPException(status_code=500, detail="Internal: empty canvas tensor chain.")


def _tensor_from_source_payload(node: Node) -> torch.Tensor:
    d: dict[str, Any] = node.data or {}
    ot = d.get("outputTensor")
    if not isinstance(ot, dict):
        raise HTTPException(
            status_code=400,
            detail=f"{node.type.value} has no outputTensor — set parameters and ensure values exist in the graph.",
        )
    shape_raw = ot.get("shape")
    values_raw = ot.get("values")
    if not isinstance(shape_raw, list) or not shape_raw:
        raise HTTPException(status_code=400, detail="Tensor source is missing a valid shape.")
    if not isinstance(values_raw, list) or not values_raw:
        raise HTTPException(status_code=400, detail="Tensor source has no values.")
    shape = [int(x) for x in shape_raw]
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 0
    if expected != len(values_raw):
        raise HTTPException(
            status_code=400,
            detail=f"Tensor source shape {shape} expects {expected} values, got {len(values_raw)}.",
        )
    try:
        flat = np.asarray(values_raw, dtype=np.float64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tensor source values are not numeric: {e}") from e
    if not np.isfinite(flat).all():
        raise HTTPException(status_code=400, detail="Tensor source contains non-finite values.")
    return torch.as_tensor(flat, dtype=torch.float32).reshape(*shape)


def forward_canvas_atomic_chain_to_torch(
    tip: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
) -> torch.Tensor:
    chain, source = collect_canvas_atomic_chain_and_source(tip, edges, nmap)
    x0 = _tensor_from_source_payload(source)
    model: nn.Module = build_sequential_from_flat_atomic_chain(chain)
    apply_parameter_tensor_payloads_from_atomic_chain(model, chain)
    model.eval()
    x_t = _prepare_x_for_atomic_sequential(model, x0)
    with torch.no_grad():
        return model(x_t)
