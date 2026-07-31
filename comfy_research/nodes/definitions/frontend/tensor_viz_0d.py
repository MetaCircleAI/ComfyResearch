"""tensor_viz_0d — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_viz_0d",
        label="0D tensor viz",
        category="visualization",
        hint="Scalar / 0D tensor panel.",
        family=("observable_user_tensor_viz_display", "observable_user_tensor_viz_anchor"),
        defaults=(),  # emptyDefaults 实况:hasDefaults:true、defaults {}
        frontend=FrontendSpec(component_key="TensorViz0dNode"),
    )
)
