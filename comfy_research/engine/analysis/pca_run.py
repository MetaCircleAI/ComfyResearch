from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from fastapi import HTTPException

from comfy_research.engine.analysis.activation_collect import activation_forward_tensors
from comfy_research.engine.models.atomic_layer_chain import SEQUENTIAL_MODEL_TYPES
from comfy_research.engine.analysis.canvas_tensor_chain import forward_canvas_atomic_chain_to_torch
from comfy_research.engine.analysis.tensor_slicing import apply_tensor_slicing_specs
from comfy_research.engine.analysis.tensor_selector_resolve import (
    tensor_choice_ids_for_selector,
    tensor_selector_key_for_output,
)
from comfy_research.engine.runs.trainer_run import _node_map
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _activation_upstream_of_tensor_selector(
    edges: list[Edge], nmap: dict[str, Node], tensor_selector_id: str
) -> Node | None:
    for e in edges:
        th = e.targetHandle or ""
        if e.target != tensor_selector_id or th not in ("tensor_list", "tensors"):
            continue
        src = nmap.get(e.source)
        if src is None:
            continue
        sh = e.sourceHandle or ""
        if src.type == NodeKind.activation and (e.sourceHandle is None or sh in ("tensor_list", "tensor")):
            return src
    return None


def _incoming_pca_tensor_source(edges: list[Edge], nmap: dict[str, Node], pca_id: str) -> Node | None:
    """PCA has a single tensor input (handle id ``tensor``); accept missing targetHandle (XYFlow default)."""
    for e in edges:
        if e.target != pca_id:
            continue
        th = (e.targetHandle or "").strip()
        if th not in ("tensor", ""):
            continue
        return nmap.get(e.source)
    return None


def _incoming_edge(edges: list[Edge], target_id: str, target_handle: str) -> Edge | None:
    for e in edges:
        if e.target != target_id:
            continue
        eth = (e.targetHandle or "").strip()
        if eth == target_handle or (target_handle == "tensor" and eth == ""):
            return e
    return None


def _as_2d_sample_matrix(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        raise HTTPException(status_code=400, detail="Tensor is scalar; need at least 1-D data.")
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    n0 = int(arr.shape[0])
    rest = int(np.prod(arr.shape[1:], dtype=np.int64))
    return arr.reshape(n0, rest)


def iter_pca_events(
    nodes: list[Node],
    edges: list[Edge],
    pca_node_id: str,
) -> Iterator[dict[str, Any]]:
    """Yield NDJSON-friendly events: several `progress`, then one `complete`."""
    nmap = _node_map(nodes)
    pca_node = nmap.get(pca_node_id)
    if pca_node is None:
        raise HTTPException(status_code=404, detail="PCA node not found")
    if pca_node.type != NodeKind.pca:
        raise HTTPException(status_code=400, detail="Target node is not a PCA node")

    inc = _incoming_pca_tensor_source(edges, nmap, pca_node_id)
    if inc is None:
        raise HTTPException(
            status_code=400,
            detail="Connect a tensor into PCA (Activation, Tensor selector, or canvas chain from Tensor constant).",
        )

    pd: dict[str, Any] = pca_node.data or {}
    selector_key = ""
    selector_data: dict[str, Any] = {}
    arr: np.ndarray
    rep_id: str

    if inc.type == NodeKind.tensor_selector:
        act = _activation_upstream_of_tensor_selector(edges, nmap, inc.id)
        if act is None:
            raise HTTPException(
                status_code=400,
                detail="Tensor selector must be connected to an Activation node's tensor output.",
            )
        tsd: dict[str, Any] = inc.data or {}
        selector_data = tsd
        inc_edge = _incoming_edge(edges, pca_node_id, "tensor")
        src_handle = inc_edge.sourceHandle if inc_edge is not None else None
        choice_ids = tensor_choice_ids_for_selector(nmap, edges, inc.id)
        selector_key = tensor_selector_key_for_output(tsd, choice_ids, src_handle)
        if not selector_key:
            raise HTTPException(
                status_code=400,
                detail="Select a tensor in the Tensor selector dropdown.",
            )
        tensors, _meta = activation_forward_tensors(nodes, edges, act.id)
        rep_id = str(pd.get("representationId") or "").strip()
        if selector_key:
            rep_id = selector_key
        keys = sorted(str(k) for k in tensors.keys())
        if not keys:
            raise HTTPException(status_code=400, detail="No activation tensors were produced for this graph.")
        if rep_id not in tensors:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown representation {rep_id!r}. Select a valid tensor in the Tensor selector.",
            )
        try:
            arr = tensors[rep_id].detach().cpu().float().numpy()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read representation tensor: {e}") from e
        try:
            arr = apply_tensor_slicing_specs(arr, selector_data.get("slices"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid tensor selector slicing: {e}") from e

    elif inc.type == NodeKind.activation:
        tensors, _meta = activation_forward_tensors(nodes, edges, inc.id)
        rep_id = str(pd.get("representationId") or "").strip()
        keys = sorted(str(k) for k in tensors.keys())
        if not keys:
            raise HTTPException(status_code=400, detail="No activation tensors were produced for this graph.")
        if not rep_id or rep_id not in tensors:
            rep_id = keys[0]
        try:
            arr = tensors[rep_id].detach().cpu().float().numpy()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read representation tensor: {e}") from e

    elif inc.type in SEQUENTIAL_MODEL_TYPES:
        t_out = forward_canvas_atomic_chain_to_torch(inc, edges, nmap)
        arr = t_out.detach().cpu().float().numpy()
        rep_id = str(pd.get("representationId") or "").strip() or "canvas_chain"

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Tensor input must come from an Activation node, a Tensor selector (after Activation), "
                "or a canvas chain (Tensor constant / Tensor linspace → atomic layers)."
            ),
        )

    X = _as_2d_sample_matrix(arr)
    n_samples, n_features = X.shape
    if n_features < 1:
        raise HTTPException(status_code=400, detail="Need at least one feature per sample.")

    yield {"type": "progress", "step": 1, "total": 10}
    yield {"type": "progress", "step": 2, "total": 10}

    Xc = X - X.mean(axis=0, keepdims=True)
    max_k = min(n_samples, n_features)
    req = int(pd.get("nComponents", 0) or 0)
    if req <= 0:
        k = max_k
    else:
        k = min(req, max_k)
    k = max(1, k)

    yield {"type": "progress", "step": 4, "total": 10}

    # Full SVD then truncate to k components (k <= min(n_samples, n_features)).
    try:
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=400, detail=f"SVD failed: {e}") from e

    yield {"type": "progress", "step": 7, "total": 10}

    s_k = s[:k]
    vt_k = vt[:k, :]
    denom = float(n_samples - 1) if n_samples > 1 else 1.0
    ev = (s_k**2) / denom
    total_ev = float(ev.sum()) if ev.size else 0.0
    if total_ev <= 0:
        ratios = [1.0 / k] * k
    else:
        ratios = (ev / total_ev).tolist()

    flat = vt_k.reshape(-1).astype(np.float64, copy=False)
    components_payload = {"shape": list(vt_k.shape), "values": flat.tolist()}

    # Sample coordinates in reduced space: rows of X (centered) projected onto k components.
    Z = Xc @ vt_k.T
    transformed_payload = {"shape": list(Z.shape), "values": Z.reshape(-1).astype(np.float64, copy=False).tolist()}

    summary = (
        f"PCA on {rep_id!r}: {n_samples}×{n_features} matrix, {k} component(s), "
        f"explained variance (first): {ratios[0] * 100:.2f}%."
    )

    yield {"type": "progress", "step": 9, "total": 10}

    yield {
        "type": "complete",
        "representation_id": rep_id,
        "transformed_tensor": transformed_payload,
        "principal_components": components_payload,
        "explained_variance_ratio": ratios,
        "summary": summary,
    }
