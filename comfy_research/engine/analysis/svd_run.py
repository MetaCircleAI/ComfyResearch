from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from fastapi import HTTPException

from comfy_research.engine.analysis.pca_run import (
    _activation_upstream_of_tensor_selector,
    _as_2d_sample_matrix,
    _incoming_edge,
    _incoming_pca_tensor_source,
)
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


def _incoming_svd_tensor_source(edges: list[Edge], nmap: dict[str, Node], svd_id: str) -> Node | None:
    return _incoming_pca_tensor_source(edges, nmap, svd_id)


def iter_svd_events(
    nodes: list[Node],
    edges: list[Edge],
    svd_node_id: str,
) -> Iterator[dict[str, Any]]:
    """Yield NDJSON-friendly events: several `progress`, then one `complete`."""
    nmap = _node_map(nodes)
    svd_node = nmap.get(svd_node_id)
    if svd_node is None:
        raise HTTPException(status_code=404, detail="SVD node not found")
    if svd_node.type != NodeKind.svd:
        raise HTTPException(status_code=400, detail="Target node is not an SVD node")

    inc = _incoming_svd_tensor_source(edges, nmap, svd_node_id)
    if inc is None:
        raise HTTPException(
            status_code=400,
            detail="Connect a tensor into SVD (Activation, Tensor selector, or canvas chain from Tensor constant).",
        )

    sd: dict[str, Any] = svd_node.data or {}
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
        inc_edge = _incoming_edge(edges, svd_node_id, "tensor")
        src_handle = inc_edge.sourceHandle if inc_edge is not None else None
        choice_ids = tensor_choice_ids_for_selector(nmap, edges, inc.id)
        selector_key = tensor_selector_key_for_output(tsd, choice_ids, src_handle)
        if not selector_key:
            raise HTTPException(
                status_code=400,
                detail="Select a tensor in the Tensor selector dropdown.",
            )
        tensors, _meta = activation_forward_tensors(nodes, edges, act.id)
        rep_id = str(sd.get("representationId") or "").strip()
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
        rep_id = str(sd.get("representationId") or "").strip()
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
        rep_id = str(sd.get("representationId") or "").strip() or "canvas_chain"

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

    yield {"type": "progress", "step": 1, "total": 8}

    remove_mean = bool(sd.get("removeMean", False))
    Xw = X - X.mean(axis=0, keepdims=True) if remove_mean else X.astype(np.float64, copy=False)

    yield {"type": "progress", "step": 3, "total": 8}

    try:
        u, s, vh = np.linalg.svd(Xw, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=400, detail=f"SVD failed: {e}") from e

    yield {"type": "progress", "step": 6, "total": 8}

    u_payload = {"shape": list(u.shape), "values": u.reshape(-1).astype(np.float64, copy=False).tolist()}
    s_payload = {"shape": list(s.shape), "values": s.astype(np.float64, copy=False).tolist()}
    v_payload = {"shape": list(vh.shape), "values": vh.reshape(-1).astype(np.float64, copy=False).tolist()}

    mean_note = "column mean removed" if remove_mean else "mean not removed"
    summary = (
        f"SVD on {rep_id!r}: {n_samples}×{n_features} matrix ({mean_note}); "
        f"U {u.shape[0]}×{u.shape[1]}, {s.size} singular values, Vh {vh.shape[0]}×{vh.shape[1]}."
    )

    yield {"type": "progress", "step": 7, "total": 8}

    yield {
        "type": "complete",
        "representation_id": rep_id,
        "u": u_payload,
        "s": s_payload,
        "v": v_payload,
        "summary": summary,
    }
