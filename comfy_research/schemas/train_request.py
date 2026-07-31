"""Shared ``TrainRequest`` body for ``POST /api/train`` and ``remote_train_cli``."""

from __future__ import annotations

import math
import numbers
from typing import Any, Literal

from pydantic import BaseModel, Field

from comfy_research.schemas.graph import Edge, Node


class TrainRequest(BaseModel):
    trainer_node_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    resume: dict[str, Any] | None = None
    hessian_oversized_policy: Literal["skip", "force"] | None = None


def sanitize_train_ndjson_value(obj: Any) -> Any:
    """Replace NaN/Inf with None so output is RFC 8259 JSON (browser JSON.parse rejects ``NaN`` tokens)."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, numbers.Integral):
        return int(obj)
    if isinstance(obj, numbers.Real):
        xf = float(obj)
        return xf if math.isfinite(xf) else None
    if isinstance(obj, dict):
        return {str(k): sanitize_train_ndjson_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_train_ndjson_value(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_train_ndjson_value(x) for x in obj]
    return obj
