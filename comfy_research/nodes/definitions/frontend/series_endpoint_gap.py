"""series_endpoint_gap — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="series_endpoint_gap",
        label="Series endpoint gap",
        category="analysis",
        hint="Scalar last−first gap for a rank-1 (1D) upstream tensor (endpoint change of the series).",
        family=("observable_user_tensor_transform",),
        defaults=(
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="SeriesEndpointGapNode"),
    )
)
