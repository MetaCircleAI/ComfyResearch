"""Resolve Tensor selector output handle → tensor list key (matches frontend ordering)."""

from __future__ import annotations

import re
from typing import Any

from comfy_research.schemas.graph import Edge, Node, NodeKind

_TENSOR_OUT = re.compile(r"^tensor_(\d+)$")


def tensor_selector_output_index(source_handle: str | None) -> int:
    h = (source_handle or "").strip()
    if not h or h == "selected_tensor":
        return 0
    m = _TENSOR_OUT.match(h)
    if m:
        return max(0, int(m.group(1), 10) - 1)
    return 0


def tensor_selector_source_handle_ok(source_handle: str | None) -> bool:
    h = (source_handle or "").strip()
    if not h or h == "selected_tensor":
        return True
    return bool(_TENSOR_OUT.match(h))


def tensor_choice_ids_for_selector(nmap: dict[str, Node], edges: list[Edge], tensor_selector_id: str) -> list[str]:
    """Mirror ``tensorChoicesForTensorsInput`` / ``tensorChoicesFromSourceHandle`` (Python graph JSON)."""
    for e in edges:
        if e.target != tensor_selector_id:
            continue
        th = e.targetHandle or ""
        if th not in ("tensor_list", "tensors"):
            continue
        src = nmap.get(e.source)
        if src is None:
            continue
        sh = e.sourceHandle or ""
        if src.type == NodeKind.activation and sh in ("tensor_list", "tensor", ""):
            ad = src.data or {}
            picks = ad.get("activationWirePicks") or ad.get("activation_wire_picks") or []
            if isinstance(picks, list) and picks:
                out_wp: list[str] = []
                for p in picks:
                    if isinstance(p, dict):
                        tk = str(p.get("tensorKey") or p.get("tensor_key") or "").strip()
                        if tk:
                            out_wp.append(tk)
                return out_wp
            opts = ad.get("representationOptions") or []
            selected = set(ad.get("selectedRepresentationIds") or [])
            out: list[str] = []
            if isinstance(opts, list):
                for o in opts:
                    if isinstance(o, dict):
                        oid = str(o.get("id") or "").strip()
                        if oid and oid in selected:
                            out.append(oid)
            return out
        if src.type == NodeKind.model_weight_tensors and sh == "tensor_list":
            wd = src.data or {}
            w = wd.get("weightTensorPayloads") or {}
            if isinstance(w, dict):
                return sorted(str(k) for k in w.keys())
            return []
        if src.type == NodeKind.training_visualization and sh in ("out_tensor_list", "out_tensor", ""):
            return ["train_loss", "test_loss"]
        if src.type == NodeKind.trainer and sh == "loss_results":
            return ["train_loss", "test_loss"]
        if src.type == NodeKind.trainer and sh == "observable_results":
            td = src.data or {}
            hist = td.get("observableMetricHistories") or {}
            if isinstance(hist, dict):
                return sorted(str(k) for k in hist.keys())
            return []
        if src.type == NodeKind.observable_viz_weight_l2 and sh in ("out_tensor", ""):
            return ["weight_l2"]
        if src.type == NodeKind.observable_viz_weight_l1 and sh in ("out_tensor", ""):
            return ["weight_l1"]
        if src.type == NodeKind.observable_viz_relu_nonlinear and sh in ("out_tensor", ""):
            return ["relu_nonlinear_count"]
        if src.type == NodeKind.observable_viz_user and sh in ("out_tensor", ""):
            return ["user_observable"]
        if src.type == NodeKind.observable_viz_embedding_trajectory and sh in ("out_tensor", ""):
            return []
        if src.type == NodeKind.observable_viz and sh in ("out_tensor", ""):
            vv = str((src.data or {}).get("vizVariant") or "")
            if vv == "weight_l2":
                return ["weight_l2"]
            if vv == "weight_l1":
                return ["weight_l1"]
            if vv == "relu_nonlinear":
                return ["relu_nonlinear_count"]
            if vv == "user":
                return ["user_observable"]
            return []
        break
    return []


def ordered_selected_tensor_keys(tsd: dict[str, Any], choice_ids: list[str]) -> list[str]:
    """Selected keys in ``choice_ids`` top-to-bottom order.

    An explicit empty ``selectedTensorKeys: []`` means no selection (no fallback to first choice).
    Legacy graphs without ``selectedTensorKeys`` still use ``selectedTensorKey`` / first choice.
    """
    id_set = set(choice_ids)
    raw = tsd.get("selectedTensorKeys")
    keys: list[str] = []
    if isinstance(raw, list):
        for k in raw:
            s = str(k).strip()
            if s and s in id_set:
                keys.append(s)
    else:
        one = str(tsd.get("selectedTensorKey") or "").strip()
        if one in id_set:
            keys = [one]
    if not keys and choice_ids and not isinstance(raw, list):
        keys = [choice_ids[0]]
    sel = set(keys)
    return [cid for cid in choice_ids if cid in sel]


def tensor_selector_key_for_output(
    tsd: dict[str, Any],
    choice_ids: list[str],
    source_handle: str | None,
) -> str:
    ordered = ordered_selected_tensor_keys(tsd, choice_ids)
    if not ordered:
        return ""
    idx = tensor_selector_output_index(source_handle)
    if idx < len(ordered):
        return ordered[idx]
    return ordered[0]


def tensor_selector_out_handle_from_chain(edges: list[Edge], picker_id: str, chain_rev: list[Node]) -> str | None:
    """First hop from the list picker toward the viz: ``chain_rev[-2]`` when present."""
    if len(chain_rev) < 2:
        return None
    tgt_id = chain_rev[-2].id
    for e in edges:
        if e.source == picker_id and e.target == tgt_id:
            return e.sourceHandle
    return None
