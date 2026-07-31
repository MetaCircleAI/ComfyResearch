"""agent_trace_viz — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="agent_trace_viz",
        label="Agent trace viz",
        category="visualization",
        family=("canvas_comment_source",),
        fields=(
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
        ),
        defaults=(
            ("plotXParamKey", None),
            ("logScaleX", False),
            ("logScaleY", False),
        ),
        frontend=FrontendSpec(component_key="TableVizNode"),
    )
)
