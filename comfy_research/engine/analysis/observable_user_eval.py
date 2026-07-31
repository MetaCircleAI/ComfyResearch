"""Evaluate metrics for user-defined observables tied to tensor viz (selector ± PCA ± Statistics)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.attention_only_model import AttentionOnlyModel, AttentionTokenPredictBundle
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.engine.analysis.tensor_slicing import apply_tensor_slicing_specs, normalize_slices
from comfy_research.engine.analysis.tensor_metrics import effective_rank_from_matrix
from comfy_research.engine.analysis.tensor_selector_resolve import (
    tensor_choice_ids_for_selector,
    tensor_selector_key_for_output,
    tensor_selector_out_handle_from_chain,
    tensor_selector_source_handle_ok,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _nmap(nodes: list[Node]) -> dict[str, Node]:
    return {n.id: n for n in nodes}


def _incoming_tensor_feed(
    edges: list[Edge], nmap: dict[str, Node], consumer_id: str, target_handle: str
) -> Node | None:
    for e in edges:
        if e.target != consumer_id:
            continue
        if (e.targetHandle or "") != target_handle:
            continue
        return nmap.get(e.source)
    return None


def _tensor_list_upstream_of_list_picker(
    edges: list[Edge], nmap: dict[str, Node], picker_id: str
) -> tuple[Node, str] | None:
    """First edge feeding a Tensor selector ``tensor_list`` (or legacy ``tensors``)."""
    for e in edges:
        th = e.targetHandle or ""
        if e.target != picker_id or th not in ("tensor_list", "tensors"):
            continue
        src = nmap.get(e.source)
        if src is None:
            continue
        return (src, e.sourceHandle or "")
    return None


def _activation_upstream_of_tensor_selector(
    edges: list[Edge], nmap: dict[str, Node], tensor_selector_id: str
) -> Node | None:
    up = _tensor_list_upstream_of_list_picker(edges, nmap, tensor_selector_id)
    if up is None:
        return None
    src, sh = up
    if src.type == NodeKind.activation and (not sh or sh in ("tensor_list", "tensor")):
        return src
    return None


def _edge_source_to_target(edges: list[Edge], src_id: str, tgt_id: str) -> Edge | None:
    for e in edges:
        if e.source == src_id and e.target == tgt_id:
            return e
    return None


def _as_2d_sample_matrix(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 0:
        raise HTTPException(status_code=400, detail="Tensor is scalar; need at least 1-D data for PCA.")
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    n0 = int(arr.shape[0])
    rest = int(np.prod(arr.shape[1:], dtype=np.int64))
    return arr.reshape(n0, rest)


def _pca_fit_project(
    X: np.ndarray,
    n_components_req: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Match comfy_research.engine.analysis.pca_run SVD convention."""
    Xc = X - X.mean(axis=0, keepdims=True)
    n_samples, n_features = Xc.shape
    if n_features < 1:
        raise HTTPException(status_code=400, detail="Need at least one feature per sample for PCA.")
    max_k = min(n_samples, n_features)
    if n_components_req <= 0:
        k = max_k
    else:
        k = min(n_components_req, max_k)
    k = max(1, k)
    try:
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=400, detail=f"SVD failed: {e}") from e
    s_k = s[:k]
    vt_k = vt[:k, :]
    denom = float(n_samples - 1) if n_samples > 1 else 1.0
    ev = (s_k**2) / denom
    total_ev = float(ev.sum()) if ev.size else 0.0
    if total_ev <= 0:
        ratios = [1.0 / k] * k
    else:
        ratios = (ev / total_ev).tolist()
    z = Xc @ vt_k.T
    return z, vt_k, ratios


def _stat_reduce_1d(slice_1d: np.ndarray, op: str) -> float:
    xs = slice_1d.astype(np.float64, copy=False).reshape(-1)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    if op == "mean":
        return float(xs.mean())
    if op == "median":
        return float(np.median(xs))
    if op == "max":
        return float(xs.max())
    if op == "min":
        return float(xs.min())
    if op in ("l2_norm", "norm"):
        return float(np.sqrt(np.sum(xs * xs)))
    if op == "l1_norm":
        return float(np.sum(np.abs(xs)))
    if op in ("std", "std_dev"):
        return float(xs.std(ddof=0))
    if op == "entropy":
        ax = np.abs(xs.astype(np.float64, copy=False))
        s = float(ax.sum())
        if s <= 0:
            return float("nan")
        p = ax / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
    raise HTTPException(status_code=400, detail=f"Unknown statistics reduction op: {op!r}")


def _parse_single_einstein(expr: str) -> tuple[str, str]:
    s = re.sub(r"\s+", "", expr)
    if "->" not in s:
        raise HTTPException(
            status_code=400,
            detail='Statistics einsum must contain "->" (e.g. ij->j).',
        )
    lhs, rhs = s.split("->", 1)
    if not lhs or not re.fullmatch(r"[a-zA-Z]+", lhs):
        raise HTTPException(
            status_code=400,
            detail="Statistics einsum left side must be letters a–z / A–Z, one per axis.",
        )
    if len(set(lhs)) != len(lhs):
        raise HTTPException(
            status_code=400,
            detail="Statistics einsum: duplicate axis labels on the left are not supported.",
        )
    if rhs and not re.fullmatch(r"[a-zA-Z]*", rhs):
        raise HTTPException(status_code=400, detail="Statistics einsum right side must be letters only.")
    for c in rhs:
        if c not in lhs:
            raise HTTPException(
                status_code=400,
                detail=f"Statistics einsum output label {c!r} does not appear on the left.",
            )
    return lhs, rhs


def _legacy_statistics_axes_to_einsum(rank: int, axes: list[int]) -> str:
    letters = [chr(ord("a") + i) for i in range(rank)]
    drop = {int(a) for a in axes if 0 <= int(a) < rank}
    rhs = "".join(letters[i] for i in range(rank) if i not in drop)
    return f"{''.join(letters)}->{rhs}"


def _single_tensor_einstein_reduce_np(arr: np.ndarray, expr: str, op: str) -> np.ndarray:
    """Apply mean / max / … over axes removed from Einstein notation (not NumPy sum)."""
    lhs, rhs = _parse_single_einstein(expr)
    if len(lhs) != arr.ndim:
        raise HTTPException(
            status_code=400,
            detail=f"Statistics left side has {len(lhs)} axes but tensor rank is {arr.ndim}.",
        )
    lmap = {lhs[i]: i for i in range(len(lhs))}
    out_letters = list(rhs)
    reduced = sorted({c for c in lhs if c not in rhs})
    sh = tuple(int(x) for x in arr.shape)
    st = np.empty(arr.ndim, dtype=np.int64)
    st[-1] = 1
    for d in range(arr.ndim - 2, -1, -1):
        st[d] = st[d + 1] * sh[d + 1]

    def coord_lin(coords: list[int]) -> int:
        return int(sum(coords[i] * int(st[i]) for i in range(arr.ndim)))

    vals = np.asarray(arr, dtype=np.float64).reshape(-1)
    out_shape = [sh[lmap[c]] for c in out_letters]
    out_size = int(np.prod(out_shape)) if out_shape else 1
    if out_shape:
        st_out = np.empty(len(out_shape), dtype=np.int64)
        st_out[-1] = 1
        for d in range(len(out_shape) - 2, -1, -1):
            st_out[d] = st_out[d + 1] * out_shape[d + 1]
    else:
        st_out = np.array([1], dtype=np.int64)

    out_flat = np.empty(out_size, dtype=np.float64)
    pos: dict[str, int] = {}
    out_multi = [0] * len(out_letters)

    def out_lin() -> int:
        if not out_letters:
            return 0
        return int(sum(out_multi[k] * int(st_out[k]) for k in range(len(out_letters))))

    def walk_red(di: int, bucket: list[float]) -> None:
        if di == len(reduced):
            co = [pos[lhs[i]] for i in range(len(lhs))]
            bucket.append(float(vals[coord_lin(co)]))
            return
        L = reduced[di]
        ax = lmap[L]
        for v in range(sh[ax]):
            pos[L] = v
            walk_red(di + 1, bucket)

    def walk_out(d: int) -> None:
        if d == len(out_letters):
            for k, c in enumerate(out_letters):
                pos[c] = out_multi[k]
            ol = out_lin()
            if not reduced:
                co = [pos[lhs[i]] for i in range(len(lhs))]
                out_flat[ol] = float(_stat_reduce_1d(np.asarray([vals[coord_lin(co)]], dtype=np.float64), op))
                return
            sl: list[float] = []
            walk_red(0, sl)
            out_flat[ol] = float(_stat_reduce_1d(np.asarray(sl, dtype=np.float64), op))
            return
        c = out_letters[d]
        ax = lmap[c]
        for i in range(sh[ax]):
            out_multi[d] = i
            walk_out(d + 1)

    walk_out(0)
    if not out_shape:
        return np.asarray(out_flat[0], dtype=np.float64).reshape(())
    return out_flat.reshape(tuple(out_shape))


def _reduce_tensor_along_axis_numpy(arr: np.ndarray, axis: int, op: str) -> np.ndarray:
    if axis < 0 or axis >= arr.ndim:
        raise HTTPException(
            status_code=400,
            detail=f"Statistics axis {axis} out of range for rank {arr.ndim}.",
        )
    return np.apply_along_axis(lambda sl: _stat_reduce_1d(sl, op), axis, arr)


def _normalize_pca_source_handle(raw: str | None) -> str:
    """React Flow often uses 'tensor' for the default PCA output wire; treat as transformed space."""
    if not raw or raw == "tensor":
        return "transformed_tensor"
    return raw


def _apply_pca_output(z: np.ndarray, vt_k: np.ndarray, ratios: list[float], selection: str) -> np.ndarray:
    sel = _normalize_pca_source_handle(selection)
    if sel == "transformed_tensor":
        return z
    if sel == "principal_components":
        return vt_k
    if sel == "explained_variance_ratio":
        return np.asarray(ratios, dtype=np.float64)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown PCA output handle {sel!r} (use transformed_tensor, principal_components, explained_variance_ratio).",
    )


def apply_pca_training(
    arr: Any,
    *,
    n_components: int = 0,
    output_handle: str | None = None,
) -> np.ndarray:
    """Fit PCA on the current tensor and return the selected output (same as canvas PCA node).

    Call once per evaluation step: during training this runs on every log step when the observable is
    computed, so the subspace is refit from the **current** activations / loss history — not frozen
    from an earlier snapshot.
    """
    req = int(n_components or 0)
    x = _as_2d_sample_matrix(np.asarray(arr))
    z, vt_k, ratios = _pca_fit_project(x, req)
    return _apply_pca_output(z, vt_k, ratios, output_handle)


def _apply_pca_node(arr: np.ndarray, node: Node, pca_out_handle: str | None) -> np.ndarray:
    pd: dict[str, Any] = node.data or {}
    req = int(pd.get("nComponents", 0) or 0)
    return apply_pca_training(arr, n_components=req, output_handle=pca_out_handle)


def _statistics_axes_from_node_data(sd: dict[str, Any]) -> list[int]:
    raw = sd.get("reductionAxes")
    if isinstance(raw, list) and len(raw) > 0:
        out: list[int] = []
        for x in raw:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))
    return [int(sd.get("reductionAxis", 0))]


def _apply_statistics_node(arr: np.ndarray, node: Node) -> np.ndarray:
    sd: dict[str, Any] = node.data or {}
    op = str(sd.get("reductionOp", "mean"))
    raw_expr = sd.get("einsumSubscripts") or sd.get("einsteinNotation")
    if isinstance(raw_expr, str) and raw_expr.strip():
        return _single_tensor_einstein_reduce_np(arr, raw_expr, op)
    axes = _statistics_axes_from_node_data(sd)
    if not axes:
        raise HTTPException(status_code=400, detail="Statistics node has no reduction axes.")
    expr = _legacy_statistics_axes_to_einsum(arr.ndim, axes)
    return _single_tensor_einstein_reduce_np(arr, expr, op)


def _apply_effective_rank_node(arr: np.ndarray) -> np.ndarray:
    v = effective_rank_from_matrix(arr)
    return np.asarray([v], dtype=np.float64)


def _apply_series_endpoint_gap_node(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 1:
        raise HTTPException(
            status_code=400,
            detail="Series endpoint gap requires a 1D tensor (after prior transforms).",
        )
    if a.size == 0:
        raise HTTPException(status_code=400, detail="Empty series for endpoint gap.")
    gap = float(a.flat[-1] - a.flat[0])
    return np.asarray([gap], dtype=np.float64)


def _node_kinds_with_capability(capability: str) -> frozenset[NodeKind]:
    return frozenset(NodeKind(node_type) for node_type in node_types_with_capability(capability))


def _observable_user_transform_types() -> frozenset[NodeKind]:
    return _node_kinds_with_capability("observable_user_tensor_transform")


def _tensor_viz_kind_types() -> frozenset[NodeKind]:
    return _node_kinds_with_capability("observable_user_tensor_viz_display")


def _observable_user_tensor_viz_anchor_types() -> frozenset[NodeKind]:
    return _node_kinds_with_capability("observable_user_tensor_viz_anchor")


def _downstream_toward_display(
    edges: list[Edge], nmap: dict[str, Node], cur_id: str
) -> Node | None:
    """Next hop along tensor selector → optional Effective rank → PCA / Statistics → tensor viz."""
    cur = nmap.get(cur_id)
    if cur is None:
        return None
    tv_t = _tensor_viz_kind_types()
    transform_t = _observable_user_transform_types()
    for e in edges:
        if e.source != cur_id:
            continue
        tgt = nmap.get(e.target)
        if tgt is None:
            continue
        sh = e.sourceHandle or ""
        if cur.type == NodeKind.tensor_selector:
            if tensor_selector_source_handle_ok(sh) and (tgt.type in tv_t or tgt.type in transform_t):
                return tgt
        elif cur.type == NodeKind.effective_rank:
            if sh in ("tensor", "") and (tgt.type in tv_t or tgt.type in (NodeKind.pca, NodeKind.statistics)):
                return tgt
        elif cur.type == NodeKind.series_endpoint_gap:
            if sh in ("tensor", "") and (tgt.type in tv_t or tgt.type in (NodeKind.pca, NodeKind.statistics)):
                return tgt
        elif cur.type == NodeKind.pca:
            if tgt.type in tv_t or tgt.type == NodeKind.statistics:
                return tgt
        elif cur.type == NodeKind.statistics:
            if tgt.type in tv_t:
                return tgt
    return None


@dataclass(frozen=True)
class ParsedObservablePath:
    """chain_rev: nodes from display-side inward (closest to optional viz first); ends with tensor_selector.

    ``tensor_viz_id`` is a legacy name: it stores the downstream anchor id used when resolving PCA output
    handles (typically the tensor viz, or the last transform node when no viz is present).
    """

    tensor_viz_id: str
    chain_rev: list[Node]


def parse_observable_user_path_from_list_picker(
    nodes: list[Node],
    edges: list[Edge],
    picker_id: str,
) -> ParsedObservablePath:
    """Same chain semantics as starting from a tensor viz, but without requiring a viz node."""
    nmap = _nmap(nodes)
    sel = nmap.get(picker_id)
    if sel is None:
        raise HTTPException(status_code=400, detail="Path anchor node not found in graph.")
    if sel.type != NodeKind.tensor_selector:
        raise HTTPException(status_code=400, detail="Anchor must be a Tensor selector.")
    middle_forward: list[Node] = []
    cur_id = picker_id
    seen: set[str] = set()
    for _ in range(64):
        if cur_id in seen:
            raise HTTPException(status_code=400, detail="Observable transform chain contains a cycle.")
        seen.add(cur_id)
        nxt = _downstream_toward_display(edges, nmap, cur_id)
        if nxt is None:
            break
        if nxt.type in _tensor_viz_kind_types():
            break
        if nxt.type not in _observable_user_transform_types():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported node between list picker and display: {nxt.type!s}. "
                    "Use Effective rank, Series endpoint gap, PCA, and/or Statistics."
                ),
            )
        middle_forward.append(nxt)
        cur_id = nxt.id
    chain_rev: list[Node] = list(reversed(middle_forward)) + [sel]
    sink = middle_forward[-1].id if middle_forward else picker_id
    return ParsedObservablePath(tensor_viz_id=sink, chain_rev=chain_rev)


def _tensor_selector_key_from_observable_path(
    nmap: dict[str, Node], edges: list[Edge], path: ParsedObservablePath
) -> str:
    picker = path.chain_rev[-1]
    tsd: dict[str, Any] = picker.data or {}
    choice_ids = tensor_choice_ids_for_selector(nmap, edges, picker.id)
    out_h = tensor_selector_out_handle_from_chain(edges, picker.id, path.chain_rev)
    return tensor_selector_key_for_output(tsd, choice_ids, out_h)


def _tensor_selector_slices_from_node(picker: Node) -> list[dict[str, Any]]:
    return normalize_slices(((picker.data or {}).get("slices")))


def _tensor_selector_slices_for_display(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [s for s in specs if str(s.get("indices", "")).strip()]


def parse_observable_user_path(
    nodes: list[Node],
    edges: list[Edge],
    tensor_viz_node_id: str,
) -> ParsedObservablePath:
    nmap = _nmap(nodes)
    tv = nmap.get(tensor_viz_node_id)
    if tv is None:
        raise HTTPException(status_code=400, detail="Tensor viz node not found in graph.")
    if tv.type not in _observable_user_tensor_viz_anchor_types():
        raise HTTPException(
            status_code=400,
            detail="User observable requires a 0D or General tensor viz node.",
        )
    chain_rev: list[Node] = []
    cur_consumer = tensor_viz_node_id
    seen: set[str] = set()
    for _ in range(64):
        if cur_consumer in seen:
            raise HTTPException(status_code=400, detail="Observable tensor path contains a cycle.")
        seen.add(cur_consumer)
        src = _incoming_tensor_feed(edges, nmap, cur_consumer, "tensor")
        if src is None:
            raise HTTPException(
                status_code=400,
                detail="Connect a tensor path to the tensor viz (tensor selector, optionally via PCA / Statistics).",
            )
        if src.type == NodeKind.tensor_selector:
            chain_rev.append(src)
            break
        if src.type in _observable_user_transform_types():
            chain_rev.append(src)
            cur_consumer = src.id
            continue
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported node type in observable path: {src.type!s}. "
                "Use Tensor selector, Effective rank, Series endpoint gap, PCA, and/or Statistics."
            ),
        )
    else:
        raise HTTPException(status_code=400, detail="Observable tensor path is too long or malformed.")
    if not chain_rev:
        raise HTTPException(status_code=400, detail="Empty observable path.")
    if chain_rev[-1].type != NodeKind.tensor_selector:
        raise HTTPException(
            status_code=400,
            detail="Observable path must end at a Tensor selector.",
        )
    return ParsedObservablePath(tensor_viz_id=tensor_viz_node_id, chain_rev=chain_rev)


def parse_observable_user_path_auto(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
) -> ParsedObservablePath:
    """Resolve path from either a tensor viz (legacy) or a tensor selector (viz not required)."""
    nmap = _nmap(nodes)
    an = nmap.get(path_anchor_id)
    if an is None:
        raise HTTPException(status_code=400, detail="Observable path anchor node not found in graph.")
    if an.type == NodeKind.tensor_selector:
        return parse_observable_user_path_from_list_picker(nodes, edges, path_anchor_id)
    if an.type in _observable_user_tensor_viz_anchor_types():
        return parse_observable_user_path(nodes, edges, path_anchor_id)
    raise HTTPException(
        status_code=400,
        detail=(
            "User observable path anchor must be a 0D or General tensor viz or a Tensor selector."
        ),
    )


def _finalize_metric_value(arr: np.ndarray) -> float:
    """Single training log scalar: mean absolute value (matches scalar |x| when rank-0)."""
    flat = arr.reshape(-1).astype(np.float64, copy=False)
    if flat.size == 0:
        return float("nan")
    return float(np.mean(np.abs(flat)))


def _validate_user_observable_list_source(
    edges: list[Edge], nmap: dict[str, Node], src: Node, src_handle: str
) -> None:
    """Training-time user metrics support the same tensor-list sources as the canvas Tensor selector."""
    sh = src_handle or ""
    if src.type == NodeKind.activation:
        if sh not in ("", "tensor_list", "tensor"):
            raise HTTPException(
                status_code=400,
                detail="Connect Activation to the Tensor selector using its tensor list output.",
            )
        from comfy_research.engine.analysis.activation_collect import resolve_model_for_activation

        if resolve_model_for_activation(nmap, edges, src.id) is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation must be connected to a supported model (MLP or residual LN model) "
                    "for training when used as a user observable source."
                ),
            )
        return
    if src.type == NodeKind.training_visualization:
        if sh not in ("", "out_tensor_list"):
            raise HTTPException(
                status_code=400,
                detail="Connect Training visualization to the Tensor selector via the “tensor list” output.",
            )
        return
    if src.type == NodeKind.trainer:
        if sh not in ("", "loss_results", "observable_results"):
            raise HTTPException(
                status_code=400,
                detail="Connect the Trainer’s train/test loss or observables handle to the Tensor selector’s tensor list input.",
            )
        return
    if src.type == NodeKind.model_weight_tensors:
        if sh not in ("", "tensor_list", "tensor"):
            raise HTTPException(
                status_code=400,
                detail="Connect Model weight tensors via its tensor list output.",
            )
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "User observables evaluated during training only support the list picker fed from "
            "Activation (layer tensors), Training visualization (loss curves), the Trainer "
            "(loss or observable histories), or Model weight tensors. "
            f"Connected source node type was {src.type!s}."
        ),
    )


def _base_array_for_user_observable(
    *,
    src: Node,
    src_handle: str,
    key: str,
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
    loss_history: list[float] | None,
    test_loss_history: list[float] | None,
    observable_metric_histories: dict[str, list[float]] | None,
) -> np.ndarray:
    sh = src_handle or ""
    if src.type == NodeKind.activation:
        from comfy_research.engine.analysis.activation_collect import _compute_activation_tensors

        tensors = _compute_activation_tensors(model, x, depth)
        t = tensors.get(key)
        if t is None:
            raise HTTPException(
                status_code=400,
                detail=f"Representation {key!r} not found for current MLP depth.",
            )
        return t.detach().cpu().float().numpy()

    lh = list(loss_history or [])
    tlh = list(test_loss_history or [])
    omh = observable_metric_histories or {}

    if src.type == NodeKind.training_visualization:
        if key == "train_loss":
            return np.asarray(lh, dtype=np.float32)
        if key == "test_loss":
            return np.asarray(tlh, dtype=np.float32)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown training visualization tensor key {key!r} (expected train_loss or test_loss).",
        )
    if src.type == NodeKind.trainer:
        if sh in ("", "loss_results"):
            if key == "train_loss":
                return np.asarray(lh, dtype=np.float32)
            if key == "test_loss":
                return np.asarray(tlh, dtype=np.float32)
            raise HTTPException(
                status_code=400,
                detail=f"Unknown trainer loss tensor key {key!r} (expected train_loss or test_loss).",
            )
        if sh == "observable_results":
            raw = omh.get(key)
            if not isinstance(raw, list) or len(raw) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No observable history for key {key!r} yet.",
                )
            return np.asarray([float(v) for v in raw], dtype=np.float32)

    if src.type == NodeKind.model_weight_tensors:
        for name, p in model.named_parameters():
            if name == key:
                return p.detach().cpu().float().numpy()
        raise HTTPException(
            status_code=400,
            detail=f"Parameter {key!r} not found on the current model.",
        )

    raise HTTPException(status_code=400, detail="Unsupported tensor list source for user observable evaluation.")


def validate_observable_user_tensor_path(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
) -> None:
    """Validate selector → (optional PCA/Stats) → (optional viz) with a supported tensor-list source."""
    nmap = _nmap(nodes)
    path = parse_observable_user_path_auto(nodes, edges, path_anchor_id)
    picker = path.chain_rev[-1]
    up = _tensor_list_upstream_of_list_picker(edges, nmap, picker.id)
    if up is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Connect the list picker’s tensor list input to a supported source: "
                "Activation, Training visualization, the Trainer (loss / observables), or Model weight tensors."
            ),
        )
    src, sh = up
    key = _tensor_selector_key_from_observable_path(nmap, edges, path)
    if not key:
        raise HTTPException(
            status_code=400,
            detail="Select a tensor (or weight parameter) in the list picker for this observable.",
        )
    _validate_user_observable_list_source(edges, nmap, src, sh)


def _eval_observable_from_persisted_definition(
    definition_code: str,
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
    *,
    loss_history: list[float] | None = None,
    test_loss_history: list[float] | None = None,
    observable_metric_histories: dict[str, list[float]] | None = None,
    representation_tensors: dict[str, torch.Tensor] | None = None,
) -> float:
    """Execute repo-persisted definition code (PCA/Statistics inlined as NumPy); no graph anchor required."""
    from comfy_research.engine.analysis.activation_collect import _compute_activation_tensors

    code = definition_code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Empty user observable definition_code.")

    lh = list(loss_history or [])
    tlh = list(test_loss_history or [])
    omh = observable_metric_histories or {}

    def reduce_along_axis(arr: Any, axis: int, op: str) -> np.ndarray:
        return _reduce_tensor_along_axis_numpy(np.asarray(arr), int(axis), str(op))

    def flat_stat_reduce(arr: Any, op: str) -> float:
        from comfy_research.engine.analysis.observable_algebra import flat_stat_reduce as _flat_stat_reduce

        return _flat_stat_reduce(np.asarray(arr), str(op))

    def activation_representation_as_numpy(rep_id: str) -> np.ndarray:
        from comfy_research.engine.analysis.representation_specs import fetch_representation_numpy

        return fetch_representation_numpy(
            model,
            x,
            depth,
            str(rep_id),
            representation_tensors=representation_tensors,
        )

    def singular_value_entropy(arr: Any) -> float:
        from comfy_research.engine.analysis.tensor_metrics import singular_value_entropy as _sve

        return _sve(np.asarray(arr))

    def apply_tensor_slicing(arr: Any, specs: Any) -> np.ndarray:
        try:
            return apply_tensor_slicing_specs(np.asarray(arr), specs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid tensor selector slicing: {e}") from e

    def named_parameter_as_numpy(name: str) -> np.ndarray:
        for n, p in model.named_parameters():
            if n == name:
                return p.detach().cpu().float().numpy()
        raise HTTPException(status_code=400, detail=f"Parameter {name!r} not found on model.")

    training_train_loss = np.asarray(lh, dtype=np.float32)
    training_test_loss = np.asarray(tlh, dtype=np.float32)
    # Matches ``format_observable_python_definition`` trainer handle naming when key is train_loss / test_loss.
    training_train_loss_series = training_train_loss
    training_test_loss_series = training_test_loss

    activation_tensors: dict[str, np.ndarray] = {}
    if "activation_tensors" in code:
        from comfy_research.engine.analysis.activation_collect import _compute_activation_tensors
        from comfy_research.engine.analysis.representation_specs import collect_representation_tensors

        if representation_tensors is not None:
            raw_tensors = representation_tensors
        else:
            raw_tensors = collect_representation_tensors(model, x, depth)
            if not raw_tensors and isinstance(model, nn.Sequential):
                raw_tensors = _compute_activation_tensors(model, x, depth)
        activation_tensors = {k: v.detach().cpu().float().numpy() for k, v in raw_tensors.items()}

    ns: dict[str, Any] = {
        "np": np,
        "torch": torch,
        "reduce_along_axis": reduce_along_axis,
        "flat_stat_reduce": flat_stat_reduce,
        "activation_representation_as_numpy": activation_representation_as_numpy,
        "singular_value_entropy": singular_value_entropy,
        "apply_tensor_slicing_specs": apply_tensor_slicing,
        "as_2d_samples": _as_2d_sample_matrix,
        "apply_pca_training": apply_pca_training,
        "training_train_loss": training_train_loss,
        "training_test_loss": training_test_loss,
        "training_train_loss_series": training_train_loss_series,
        "training_test_loss_series": training_test_loss_series,
        "observable_metric_histories": omh,
        "named_parameter_as_numpy": named_parameter_as_numpy,
        "effective_rank_from_matrix": effective_rank_from_matrix,
    }
    if activation_tensors:
        ns["activation_tensors"] = activation_tensors

    if isinstance(model, (AttentionOnlyModel, AttentionTokenPredictBundle)):
        ns["attention_weight_arrays"] = model.observable_numpy_arrays()

    ns["__builtins__"] = {
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "bool": bool,
        "str": str,
        "min": min,
        "max": max,
        "abs": abs,
        "enumerate": enumerate,
        "zip": zip,
        "getattr": getattr,
        "isinstance": isinstance,
        "tuple": tuple,
        "list": list,
        "dict": dict,
        "set": set,
        "round": round,
        "sum": sum,
        "print": lambda *_a, **_k: None,
    }
    try:
        exec(compile(code, "<user_observable definition_code>", "exec"), ns, ns)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"User observable definition_code failed at training time: {e}",
        ) from e
    metric = ns.get("metric")
    if metric is None:
        raise HTTPException(
            status_code=400,
            detail="User observable definition_code did not assign a numeric 'metric'.",
        )
    try:
        return float(metric)
    except (TypeError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"User observable 'metric' must be a float-compatible value, got {type(metric)!r}.",
        ) from e


def eval_observable_user_mean_abs(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
    *,
    loss_history: list[float] | None = None,
    test_loss_history: list[float] | None = None,
    observable_metric_histories: dict[str, list[float]] | None = None,
    definition_code: str | None = None,
    representation_tensors: dict[str, torch.Tensor] | None = None,
) -> float:
    """Mean |·| over the final tensor after selector → optional PCA / Statistics (same convention as canvas).

    Loss / observable series from Training visualization or the Trainer use the histories passed in from the
    training loop (same values as the canvas would show).

    When ``definition_code`` is set (persisted server-side), the graph anchor is not used; the string is
    executed with training-time bindings (``training_train_loss``, ``reduce_along_axis``, etc.).
    """
    if definition_code is not None and str(definition_code).strip():
        return _eval_observable_from_persisted_definition(
            str(definition_code).strip(),
            model,
            x,
            depth,
            loss_history=loss_history,
            test_loss_history=test_loss_history,
            observable_metric_histories=observable_metric_histories,
            representation_tensors=representation_tensors,
        )
    nmap = _nmap(nodes)
    path = parse_observable_user_path_auto(nodes, edges, path_anchor_id)
    picker = path.chain_rev[-1]
    up = _tensor_list_upstream_of_list_picker(edges, nmap, picker.id)
    if up is None:
        raise HTTPException(
            status_code=400,
            detail="Connect the list picker’s tensor list input (activation, training viz, trainer, or model weights).",
        )
    src, src_handle = up
    key = _tensor_selector_key_from_observable_path(nmap, edges, path)
    if not key:
        raise HTTPException(status_code=400, detail="List picker has no selected tensor key.")

    arr = _base_array_for_user_observable(
        src=src,
        src_handle=src_handle,
        key=key,
        model=model,
        x=x,
        depth=depth,
        loss_history=loss_history,
        test_loss_history=test_loss_history,
        observable_metric_histories=observable_metric_histories,
    )
    try:
        arr = apply_tensor_slicing_specs(arr, _tensor_selector_slices_from_node(picker))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tensor selector slicing: {e}") from e
    fwd = list(reversed(path.chain_rev))
    tv_id = path.tensor_viz_id
    for j in range(1, len(fwd)):
        tn = fwd[j]
        down_id = fwd[j + 1].id if j + 1 < len(fwd) else tv_id
        if tn.type == NodeKind.pca:
            e_out = _edge_source_to_target(edges, tn.id, down_id)
            if down_id == tn.id:
                e_out = None
            handle = _normalize_pca_source_handle(e_out.sourceHandle if e_out is not None else None)
            arr = _apply_pca_node(arr, tn, handle)
        elif tn.type == NodeKind.statistics:
            arr = _apply_statistics_node(arr, tn)
        elif tn.type == NodeKind.effective_rank:
            arr = _apply_effective_rank_node(arr)
        elif tn.type == NodeKind.series_endpoint_gap:
            arr = _apply_series_endpoint_gap_node(arr)
        else:
            raise HTTPException(status_code=400, detail="Invalid transform in observable path.")
    return _finalize_metric_value(arr)


def _human_label_for_representation(rep: str) -> str:
    """Map MLP representation ids (e.g. h2_preact) to short UI labels."""
    if rep == "input":
        return "input"
    if rep == "output":
        return "output"
    m = re.match(r"^h(\d+)_preact$", rep)
    if m:
        return f"layer {m.group(1)} preact"
    m = re.match(r"^h(\d+)_postact$", rep)
    if m:
        return f"layer {m.group(1)} postact"
    return rep


def _human_label_tensor_list_key(key: str) -> str:
    """Labels for Tensor selector keys (activations, loss series, observables)."""
    if key == "train_loss":
        return "train loss"
    if key == "test_loss":
        return "test loss"
    return _human_label_for_representation(key)


def _human_label_pca_output(handle: str) -> str:
    h = _normalize_pca_source_handle(handle)
    if h == "transformed_tensor":
        return "transformed"
    if h == "principal_components":
        return "principal components"
    if h == "explained_variance_ratio":
        return "explained variance"
    return h


def format_observable_human_chain(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
) -> str:
    """One-line pipeline, e.g. ``layer 2 preact → max (dim = 0) → mean(|·|)``."""
    path = parse_observable_user_path_auto(nodes, edges, path_anchor_id)
    nmap_h = _nmap(nodes)
    key = _tensor_selector_key_from_observable_path(nmap_h, edges, path) or "?"
    parts: list[str] = [_human_label_tensor_list_key(key)]
    slice_specs = _tensor_selector_slices_for_display(_tensor_selector_slices_from_node(path.chain_rev[-1]))
    if slice_specs:
        sl = ", ".join(
            f"dim {int(s.get('dimension', 0))}: {str(s.get('indices', '')).strip()}" for s in slice_specs
        )
        parts.append(f"slice ({sl})")
    fwd = list(reversed(path.chain_rev))
    tv_id = path.tensor_viz_id
    for j in range(1, len(fwd)):
        tn = fwd[j]
        down_id = fwd[j + 1].id if j + 1 < len(fwd) else tv_id
        e_out = _edge_source_to_target(edges, tn.id, down_id)
        if down_id == tn.id:
            e_out = None
        if tn.type == NodeKind.pca:
            pd: dict[str, Any] = tn.data or {}
            nc = int(pd.get("nComponents", 0) or 0)
            handle = _normalize_pca_source_handle(e_out.sourceHandle if e_out is not None else None)
            k_txt = "full rank" if nc <= 0 else f"k={nc}"
            parts.append(f"PCA ({k_txt}, {_human_label_pca_output(handle)})")
        elif tn.type == NodeKind.statistics:
            sd: dict[str, Any] = tn.data or {}
            op = str(sd.get("reductionOp", "mean"))
            ex = sd.get("einsumSubscripts") or sd.get("einsteinNotation")
            if isinstance(ex, str) and ex.strip():
                ex_compact = re.sub(r"\s+", " ", ex.strip())
                parts.append(f"{op} (einsum {ex_compact})")
            else:
                axes = _statistics_axes_from_node_data(sd)
                dim_txt = ", ".join(str(a) for a in axes) if axes else "?"
                parts.append(f"{op} (dims = {dim_txt})")
        elif tn.type == NodeKind.effective_rank:
            parts.append("effective rank (entropy over singular values)")
        elif tn.type == NodeKind.series_endpoint_gap:
            parts.append("series endpoint gap (last − first)")
    parts.append("mean(|·|) (training metric)")
    return " → ".join(parts)


def format_observable_python_definition(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
) -> str:
    """Executable Python sketch for the training-time pipeline (PCA / Statistics as NumPy; viz not required)."""
    nmap = _nmap(nodes)
    path = parse_observable_user_path_auto(nodes, edges, path_anchor_id)
    sel = path.chain_rev[-1]
    up = _tensor_list_upstream_of_list_picker(edges, nmap, sel.id)
    key = _tensor_selector_key_from_observable_path(nmap, edges, path) or "?"
    slice_specs = _tensor_selector_slices_from_node(sel)
    slice_specs_effective = _tensor_selector_slices_for_display(slice_specs)
    lines: list[str] = [
        "# Training-time user observable (tensor transforms; viz panels are display-only)",
        f"# Path anchor: {path_anchor_id!r}  # tensor viz id or tensor selector id",
    ]
    if up is None:
        lines.append("# Connect Tensor selector tensor list (activation, training viz, or trainer)")
        lines.append(f"t = ?  # selected key {key!r}")
    else:
        src, _sh = up
        if src.type == NodeKind.activation:
            # Keep ``t = …`` on its own short line so viewers (Prism wrapLongLines) don’t split ``activation_tensors``.
            lines.append("# Batch from MLP forward on trainer batch; depth matches trainer")
            lines.append(f"t = activation_tensors[{key!r}]")
            lines.append(f"# Activation node: {src.id!r}")
        elif src.type == NodeKind.training_visualization:
            lines.append(
                f"# Series from Training visualization (same arrays as the loss plot): key {key!r}",
            )
            lines.append(f"t = np.asarray(training_{key})  # train_loss / test_loss histories")
            lines.append(f"# Training visualization node: {src.id!r}")
        elif src.type == NodeKind.trainer:
            lines.append(f"# Series from Trainer handles; key {key!r}")
            lines.append(f"t = np.asarray(training_{key}_series)  # aligned with logging")
            lines.append(f"# Trainer node: {src.id!r}")
        elif src.type == NodeKind.model_weight_tensors:
            lines.append("# Current model parameter tensor (matches Tensor selector selection)")
            lines.append(f"t = named_parameter_as_numpy({key!r})")
            lines.append(f"# Model weight tensors node: {src.id!r}")
        else:
            lines.append(f"t = ...  # source type {src.type!s}")
    if slice_specs_effective:
        lines.append("")
        lines.append("# --- Tensor selector slicing (same semantics as canvas tensor slicing)")
        lines.append(f"t = apply_tensor_slicing_specs(t, {slice_specs!r})")
    fwd = list(reversed(path.chain_rev))
    tv_id = path.tensor_viz_id
    for j in range(1, len(fwd)):
        tgt_n = fwd[j]
        down_id = fwd[j + 1].id if j + 1 < len(fwd) else tv_id
        e_out = _edge_source_to_target(edges, tgt_n.id, down_id)
        if down_id == tgt_n.id:
            e_out = None
        if tgt_n.type == NodeKind.pca:
            pd: dict[str, Any] = tgt_n.data or {}
            nc = int(pd.get("nComponents", 0) or 0)
            handle = _normalize_pca_source_handle(e_out.sourceHandle if e_out is not None else None)
            lines.append("")
            lines.append(
                f"# --- PCA node {tgt_n.id!r} (n_components={nc or 'all'}, output={handle}); "
                "refits on each training eval (each log step)"
            )
            lines.append(
                f"t = apply_pca_training(t, n_components={nc}, output_handle={handle!r})",
            )
        elif tgt_n.type == NodeKind.statistics:
            sd: dict[str, Any] = tgt_n.data or {}
            op = str(sd.get("reductionOp", "mean"))
            ex = sd.get("einsumSubscripts") or sd.get("einsteinNotation")
            lines.append("")
            if isinstance(ex, str) and ex.strip():
                lines.append(
                    f"# --- Statistics node {tgt_n.id!r} (Einstein / einsum {ex.strip()!r}, op={op!r}); "
                    "matches canvas Statistics"
                )
                lines.append(f"t = einstein_reduce(t, {ex.strip()!r}, op={op!r})  # same semantics as the app")
            else:
                axes = sorted(_statistics_axes_from_node_data(sd), reverse=True)
                lines.append(
                    f"# --- Statistics node {tgt_n.id!r} (legacy axes high→low={axes}, op={op!r}); "
                    "matches canvas Statistics"
                )
                lines.append("for _ax in " + repr(axes) + ":")
                lines.append(f"    t = reduce_along_axis(t, axis=_ax, op={op!r})")
        elif tgt_n.type == NodeKind.effective_rank:
            lines.append("")
            lines.append(f"# --- Effective rank node {tgt_n.id!r} (entropy over singular values)")
            lines.append("t = np.asarray(effective_rank_from_matrix(t)).reshape(1)")
        elif tgt_n.type == NodeKind.series_endpoint_gap:
            lines.append("")
            lines.append(f"# --- Series endpoint gap node {tgt_n.id!r} (1D: last − first)")
            lines.append("t = np.asarray(float(t[-1] - t[0])).reshape(1)  # t must be 1D (same as canvas node)")
    lines.append("")
    lines.append("metric = float(np.mean(np.abs(np.asarray(t).reshape(-1))))")
    return "\n".join(lines)


def describe_observable_training_path(
    nodes: list[Node],
    edges: list[Edge],
    path_anchor_id: str,
) -> dict[str, str]:
    """Validate path and return definition text plus a short human-readable pipeline line.

    ``path_anchor_id`` may be a tensor viz id (legacy) or a tensor selector id (viz not required).
    """
    validate_observable_user_tensor_path(nodes, edges, path_anchor_id)
    return {
        "definition": format_observable_python_definition(nodes, edges, path_anchor_id),
        "human_chain": format_observable_human_chain(nodes, edges, path_anchor_id),
    }


def eval_algebra_observable_user(
    *,
    tensor_name: str,
    tensor_scope: str,
    reductions_raw: list[dict[str, Any]],
    definition_code: str,
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
    flatten_mode: str = "none",
    observable_source: str = "weight",
    representation_id: str = "",
    loss_history: list[float] | None = None,
    test_loss_history: list[float] | None = None,
    observable_metric_histories: dict[str, list[float]] | None = None,
    representation_tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[float, dict[str, float] | None]:
    """Evaluate an algebra observable; returns aggregate scalar and optional per-member values."""
    from comfy_research.engine.analysis.representation_specs import collect_representation_tensors
    from comfy_research.engine.analysis.observable_algebra import (
        format_algebra_definition_code,
        matching_parameter_names,
        matching_representation_ids,
        normalize_flatten_mode,
        parse_axis_reductions,
        reduce_global_flatten_algebra,
    )

    mode = normalize_flatten_mode(flatten_mode)
    specs = parse_axis_reductions(reductions_raw, flatten_mode=mode)
    src = (observable_source or "weight").strip().lower()
    scope = (tensor_scope or "single").strip()
    if src == "representation":
        from comfy_research.engine.analysis.representation_specs import fetch_representation_numpy

        rep_id = (representation_id or tensor_name).strip()
        rep_store = representation_tensors
        if rep_store is None:
            rep_store = collect_representation_tensors(model, x, depth)
        all_rep_ids = list(rep_store.keys())
        rep_names = matching_representation_ids(all_rep_ids, rep_id, scope)
        if not rep_names:
            raise HTTPException(
                status_code=400,
                detail=f"No representations match observable {rep_id!r} (scope={scope!r}).",
            )
        if mode == "global" and len(rep_names) > 1:
            arrays: list[np.ndarray] = []
            for rid in rep_names:
                arrays.append(
                    fetch_representation_numpy(
                        model,
                        x,
                        depth,
                        rid,
                        representation_tensors=rep_store,
                    )
                )
            primary = reduce_global_flatten_algebra(arrays, specs)
            return primary, None
        member_vals: dict[str, float] = {}
        for rid in rep_names:
            code = format_algebra_definition_code(
                tensor_name=rid,
                reductions=specs,
                flatten_mode=mode,
                observable_source="representation",
                member_tensor_names=rep_names if mode == "global" else None,
            )
            member_vals[rid] = _eval_observable_from_persisted_definition(
                code,
                model,
                x,
                depth,
                loss_history=loss_history,
                test_loss_history=test_loss_history,
                observable_metric_histories=observable_metric_histories,
                representation_tensors=rep_store,
            )
        if scope != "all_matching" or len(member_vals) <= 1:
            return next(iter(member_vals.values())), None
        vals = list(member_vals.values())
        primary = float(sum(vals) / len(vals)) if vals else float("nan")
        return primary, member_vals
    all_names = [n for n, _ in model.named_parameters()]
    names = matching_parameter_names(all_names, tensor_name, scope)

    if not names:
        raise HTTPException(
            status_code=400,
            detail=f"No model parameters match observable tensor {tensor_name!r} (scope={scope!r}).",
        )

    if mode == "global" and len(names) > 1:
        arrays: list[np.ndarray] = []
        for pname in names:
            for n, p in model.named_parameters():
                if n == pname:
                    arrays.append(p.detach().cpu().float().numpy())
                    break
        primary = reduce_global_flatten_algebra(arrays, specs)
        return primary, None

    member_vals: dict[str, float] = {}
    for pname in names:
        code = format_algebra_definition_code(
            tensor_name=pname,
            reductions=specs,
            flatten_mode=mode,
            member_tensor_names=names if mode == "global" else None,
        )
        member_vals[pname] = _eval_observable_from_persisted_definition(
            code,
            model,
            x,
            depth,
            loss_history=loss_history,
            test_loss_history=test_loss_history,
            observable_metric_histories=observable_metric_histories,
        )

    if scope != "all_matching" or len(member_vals) <= 1:
        return next(iter(member_vals.values())), None

    vals = list(member_vals.values())
    primary = float(sum(vals) / len(vals)) if vals else float("nan")
    return primary, member_vals
