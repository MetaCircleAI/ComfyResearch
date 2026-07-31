"""Model weight materialization and effective-rank scalar API."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfy_research.engine.analysis.model_weight_materialize import run_collect_model_weights, run_model_weight_specs
from comfy_research.engine.analysis.representation_specs import run_model_representation_specs
from comfy_research.engine.analysis.tensor_metrics import effective_rank_from_matrix
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["weights"])


class CollectModelWeightsBody(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    model_weight_tensors_node_id: str = Field(default="", min_length=0)
    include_upstream_chain: bool = True


class ModelWeightSpecsBody(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    model_node_id: str = Field(min_length=1)
    include_upstream_chain: bool = True


class EffectiveRankValueBody(BaseModel):
    shape: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


@router.post("/collect_model_weights")
def post_collect_model_weights(body: CollectModelWeightsBody) -> dict[str, Any]:
    wid = body.model_weight_tensors_node_id.strip()
    if not wid:
        raise HTTPException(status_code=400, detail="model_weight_tensors_node_id is required.")
    return run_collect_model_weights(
        body.nodes,
        body.edges,
        wid,
        include_upstream_chain=bool(body.include_upstream_chain),
    )


@router.post("/model_weight_specs")
def post_model_weight_specs(body: ModelWeightSpecsBody) -> dict[str, Any]:
    return run_model_weight_specs(
        body.nodes,
        body.edges,
        body.model_node_id.strip(),
        include_upstream_chain=bool(body.include_upstream_chain),
    )


@router.post("/model_representation_specs")
def post_model_representation_specs(body: ModelWeightSpecsBody) -> dict[str, Any]:
    return run_model_representation_specs(
        body.nodes,
        body.edges,
        body.model_node_id.strip(),
        include_upstream_chain=bool(body.include_upstream_chain),
    )


@router.post("/effective_rank_value")
def post_effective_rank_value(body: EffectiveRankValueBody) -> dict[str, Any]:
    sh = [int(x) for x in body.shape]
    vals = [float(x) for x in body.values]
    if not vals:
        return {"value": float("nan")}
    arr = np.asarray(vals, dtype=np.float64).reshape(sh) if sh else np.asarray(vals, dtype=np.float64)
    return {"value": effective_rank_from_matrix(arr)}
