"""tensor_viz_scatter — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_viz_scatter",
        label="Scatter plot viz",
        category="visualization",
        hint="Scatter plot of tensor 1 vs tensor 2 (both effectively 1D).",
        defaults=(),  # emptyDefaults 实况:hasDefaults:true、defaults {}
        frontend=FrontendSpec(component_key="TensorScatterVizNode"),
    )
)
