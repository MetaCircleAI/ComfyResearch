"""docking_pose_viz — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="docking_pose_viz",
        label="Docking pose viz",
        category="visualization",
        family=("canvas_comment_source",),
        defaults=(),  # emptyDefaults 实况:hasDefaults:true、defaults {}
        frontend=FrontendSpec(component_key="TensorScatterVizNode"),
    )
)
