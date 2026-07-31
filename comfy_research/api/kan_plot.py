"""API: render pykan ``KAN.plot()`` for the visualize_kan node."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from comfy_research.engine.analysis.kan_visualize import run_kan_plot
from comfy_research.schemas.graph import Edge, Node

router = APIRouter(prefix="/api", tags=["kan"])


class KanPlotBody(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    visualize_kan_node_id: str = Field(min_length=1)
    sample_count: int = Field(
        256,
        description="Used only when visualize_kan has no dataset socket; otherwise batch size comes from the dataset.",
    )
    plot_scale: float = 0.35
    plot_metric: Literal["backward", "forward_n", "forward_u"] = "backward"
    dpi: int = 120


@router.post("/kan_plot")
def post_kan_plot(body: KanPlotBody) -> dict[str, Any]:
    return run_kan_plot(
        body.nodes,
        body.edges,
        body.visualize_kan_node_id.strip(),
        sample_count=int(body.sample_count),
        plot_scale=float(body.plot_scale),
        plot_metric=body.plot_metric,
        dpi=int(body.dpi),
    )
