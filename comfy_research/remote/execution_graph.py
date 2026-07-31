"""Minimal executable graph enforced at the remote transport boundary."""

from __future__ import annotations

from typing import Any


_TRAINER_RUNTIME_KEYS = frozenset(
    {
        "hostTrainUi",
        "trainUi",
        "trainPauseCheckpoint",
        "trainSeriesPauseCtx",
        "lossHistory",
        "testLossHistory",
        "regLossHistory",
        "stepTicks",
        "epochTicks",
        "observableMetricHistories",
        "memoryCheckpoint_b64",
        "checkpoint_b64",
        "targetCurveStepTicks",
        "targetCurveLossHistory",
        "lastTrainLoopSeconds",
        "lastAutoTuneSummary",
        "autoTuneComparisonResult",
    }
)


def _sanitize_node(node: dict[str, Any]) -> dict[str, Any]:
    out = dict(node)
    if node.get("type") in {"trainer", "crl_trainer"} and isinstance(node.get("data"), dict):
        out["data"] = {
            key: value
            for key, value in node["data"].items()
            if key not in _TRAINER_RUNTIME_KEYS
        }
    return out


def select_execution_graph(payload: dict[str, Any], target_node_id: str) -> dict[str, Any]:
    """Keep one target, its upstream wires, and combined-model child structure."""
    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return dict(payload)
    nodes = [node for node in raw_nodes if isinstance(node, dict) and isinstance(node.get("id"), str)]
    edges = [
        edge
        for edge in raw_edges
        if isinstance(edge, dict)
        and isinstance(edge.get("source"), str)
        and isinstance(edge.get("target"), str)
    ]
    node_by_id = {str(node["id"]): node for node in nodes}
    included = {target_node_id} if target_node_id in node_by_id else set()

    changed = True
    while changed:
        changed = False
        for edge in edges:
            source = str(edge["source"])
            target = str(edge["target"])
            if (
                target in included
                and node_by_id[target].get("type") != "model_checkpoint"
                and source in node_by_id
                and source not in included
            ):
                included.add(source)
                changed = True
        for node in nodes:
            node_id = str(node["id"])
            parent_id = str(node.get("parentId") or "")
            if node_id in included and parent_id in node_by_id and parent_id not in included:
                included.add(parent_id)
                changed = True
            if parent_id in included and node_by_id[parent_id].get("type") == "combined_model" and node_id not in included:
                included.add(node_id)
                changed = True

    out = dict(payload)
    out["nodes"] = [_sanitize_node(node) for node in nodes if str(node["id"]) in included]
    out["edges"] = [
        edge
        for edge in edges
        if str(edge["source"]) in included and str(edge["target"]) in included
    ]
    return out
