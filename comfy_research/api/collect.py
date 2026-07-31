from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from comfy_research.engine.analysis.activation_collect import run_collect_activations
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["collect"])


class CollectActivationsRequest(BaseModel):
    activation_node_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)


@router.post("/collect_activations")
def post_collect_activations(body: CollectActivationsRequest) -> dict:
    return run_collect_activations(body.nodes, body.edges, body.activation_node_id)
