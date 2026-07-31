"""shape_checker — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="shape_checker",
        label="Shape checker",
        category="analysis",
        hint="Attach to a node output and click Check shape to show the current tensor shape.",
        defaults=(
            ("shapeText", None),
            ("sourceSummary", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="ShapeCheckerNode"),
    )
)
