from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from comfy_research.generated.node_kind import GENERATED_NODE_KIND_VALUES
from comfy_research.generated.node_params import validate_node_params


TRANSITIONAL_SCHEMA_ONLY_NODE_KIND_VALUES: tuple[str, ...] = (
    "dataset",
    "loss",
    "model",
    "observable",
    "observable_viz_embedding_trajectory",
    "observable_viz_relu_nonlinear",
    "observable_viz_user",
    "observable_viz_weight_l1",
    "observable_viz_weight_l2",
    "optimizer",
)

NodeKind = Enum(
    "NodeKind",
    {value: value for value in (*GENERATED_NODE_KIND_VALUES, *TRANSITIONAL_SCHEMA_ONLY_NODE_KIND_VALUES)},
    type=str,
)


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class Node(BaseModel):
    id: str
    type: NodeKind
    position: Position = Field(default_factory=Position)
    data: dict[str, Any] = Field(default_factory=dict)
    parentId: str | None = None
    extent: Literal["parent"] | None = None
    hidden: bool | None = None
    style: dict[str, Any] | None = None

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_legacy_node_types(cls, v: Any) -> Any:
        # Removed node kind: weight_extractor → tensor_selector (same data shape).
        if v == "weight_extractor":
            return NodeKind.tensor_selector.value
        return v

    @field_validator("data", mode="after")
    @classmethod
    def _validate_generated_node_params(cls, v: dict[str, Any], info: Any) -> dict[str, Any]:
        raw_type = info.data.get("type") if hasattr(info, "data") else None
        if raw_type is None:
            return v
        node_type = raw_type.value if isinstance(raw_type, NodeKind) else str(raw_type)
        validate_node_params(node_type, v)
        return v


class Edge(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class Viewport(BaseModel):
    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


class GraphDocument(BaseModel):
    version: int = 1
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    viewport: Viewport | None = None


def empty_graph() -> GraphDocument:
    return GraphDocument(version=1, nodes=[], edges=[], viewport=None)


def default_research_graph() -> GraphDocument:
    """Bundled demo pipeline (linear dataset → MLP, Adam, MSE → trainer → viz)."""
    path = Path(__file__).resolve().parent / "default_research_graph.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GraphDocument.model_validate(raw)
