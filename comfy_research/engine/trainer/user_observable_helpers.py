"""User-observable definition and anchor helpers extracted from trainer_run.

The record getter is INJECTED (keyword-only) rather than imported: the test
seam patches ``comfy_research.engine.runs.trainer_run.get_user_observable_record``
(see scripts/verify_representation_observable_training.py), and callers pass
that module global (or PrepareState/ObservableRecorder forward it) at call
time so the patch keeps working. This module must never import
comfy_research.api.user_observables (guard-enforced).
"""
from typing import Any

from comfy_research.engine.trainer.observable_config import _observable_metrics_log_enabled
from comfy_research.schemas.graph import Node, NodeKind


def _user_observable_path_anchor(od_u: dict[str, Any], *, get_user_observable_record: Any) -> str:
    """Prefer an in-graph tensor selector id; else load stored id from the server repo; else legacy viz id."""
    sel = str(od_u.get("tensorSelectorNodeId") or "").strip()
    if sel:
        return sel
    uid = str(od_u.get("userObservableId") or "").strip()
    if uid:
        rec = get_user_observable_record(uid)
        if rec is not None:
            ts = str(rec.tensor_selector_node_id or "").strip()
            if ts:
                return ts
    return str(od_u.get("tensorVizNodeId") or "").strip()


def _user_observable_definition_code(od_u: dict[str, Any], *, get_user_observable_record: Any) -> str:
    """Non-empty persisted definition from the repo; training eval uses this and does not need a graph anchor."""
    uid = str(od_u.get("userObservableId") or "").strip()
    if not uid:
        return ""
    rec = get_user_observable_record(uid)
    if rec is None:
        return ""
    return str(rec.definition_code or "").strip()


def _user_observable_algebra_rec(od_u: dict[str, Any], *, get_user_observable_record: Any):
    uid = str(od_u.get("userObservableId") or "").strip()
    if not uid:
        return None
    rec = get_user_observable_record(uid)
    if rec is None or str(rec.definition_kind or "") != "algebra":
        return None
    return rec


def _log_step_needs_representation_cache(
    observable_nodes: list[Node],
    disable_extra: bool,
    *,
    get_user_observable_record: Any,
) -> bool:
    """True when any enabled wired observable reads activations during ``record()``."""
    for on in observable_nodes:
        if not _observable_metrics_log_enabled(on, disable_extra):
            continue
        if on.type not in (NodeKind.observable_user, "observable_user"):
            continue
        od_u: dict[str, Any] = on.data or {}
        algebra_rec = _user_observable_algebra_rec(od_u, get_user_observable_record=get_user_observable_record)
        if algebra_rec is not None and (
            str(algebra_rec.observable_source or "weight").strip().lower() == "representation"
        ):
            return True
        def_code = _user_observable_definition_code(od_u, get_user_observable_record=get_user_observable_record)
        if def_code and (
            "activation_representation_as_numpy" in def_code or "activation_tensors" in def_code
        ):
            return True
    return False
