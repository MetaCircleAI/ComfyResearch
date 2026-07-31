"""Graph wiring/lookup helpers for the trainer runtime (extracted from trainer_run)."""

from fastapi import HTTPException

from comfy_research.engine.node_builder_registry import (
    registered_builder_node_types_for,
    registered_trainer_dataset_node_types,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind

_DATASET_NODE_TYPES = frozenset(NodeKind(node_type) for node_type in registered_trainer_dataset_node_types())
_DATASET_NODE_TYPE_LABEL = ", ".join(node_type.value for node_type in sorted(_DATASET_NODE_TYPES, key=lambda t: t.value))

_OPTIMIZER_NODE_TYPES = frozenset(NodeKind(node_type) for node_type in registered_builder_node_types_for("optimizer"))
_OPTIMIZER_NODE_TYPE_LABEL = ", ".join(node_type.value for node_type in sorted(_OPTIMIZER_NODE_TYPES, key=lambda t: t.value))


def _node_map(nodes: list[Node]) -> dict[str, Node]:
    return {n.id: n for n in nodes}


def _incoming(
    edges: list[Edge], nodes: dict[str, Node], trainer_id: str, handle: str
) -> Node | None:
    want = handle or ""
    matches: list[Node] = []
    for e in edges:
        if e.target != trainer_id:
            continue
        th = e.targetHandle or ""
        if th != want:
            continue
        n = nodes.get(e.source)
        if n is not None:
            matches.append(n)
    if len(matches) > 1:
        kinds = ", ".join(str(n.type) for n in matches)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple edges target node {trainer_id!r} handle {handle!r} ({kinds}). "
                "Disconnect duplicate wires so exactly one upstream node feeds this socket."
            ),
        )
    return matches[0] if matches else None


def _incoming_all(
    edges: list[Edge], nodes: dict[str, Node], trainer_id: str, handle: str
) -> list[Node]:
    out: list[Node] = []
    seen: set[str] = set()
    for e in edges:
        if e.target == trainer_id and e.targetHandle == handle:
            n = nodes.get(e.source)
            if n is not None and n.id not in seen:
                seen.add(n.id)
                out.append(n)
    return out


def _incoming_to_target(edges: list[Edge], nodes: dict[str, Node], target_id: str, handle: str) -> Node | None:
    """Single upstream node wired to ``target_id``'s target handle (same duplicate rules as ``_incoming``)."""
    want = handle or ""
    matches: list[Node] = []
    for e in edges:
        if e.target != target_id:
            continue
        th = e.targetHandle or ""
        if th != want:
            continue
        n = nodes.get(e.source)
        if n is not None:
            matches.append(n)
    if len(matches) > 1:
        kinds = ", ".join(str(n.type) for n in matches)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple edges target node {target_id!r} handle {handle!r} ({kinds}). "
                "Disconnect duplicate wires so exactly one upstream node feeds this socket."
            ),
        )
    return matches[0] if matches else None


def _require(node: Node | None, expect_type: str, label: str) -> Node:
    if node is None:
        raise HTTPException(status_code=400, detail=f"Trainer is missing connection: {label}")
    if node.type != expect_type:
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a {expect_type} node, got {node.type}",
        )
    return node


def _require_dataset(node: Node | None, label: str) -> Node:
    if node is None:
        raise HTTPException(status_code=400, detail=f"Trainer is missing connection: {label}")
    if node.type not in _DATASET_NODE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(f"{label} must be one of: {_DATASET_NODE_TYPE_LABEL}; got {node.type}"),
        )
    return node


def _require_optimizer(node: Node | None, label: str) -> Node:
    if node is None:
        raise HTTPException(status_code=400, detail=f"Trainer is missing connection: {label}")
    if node.type not in _OPTIMIZER_NODE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(f"{label} must be one of: {_OPTIMIZER_NODE_TYPE_LABEL}; got {node.type}"),
        )
    return node
