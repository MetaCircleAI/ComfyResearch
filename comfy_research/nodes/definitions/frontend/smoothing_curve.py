"""smoothing_curve — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="smoothing_curve",
        label="Smoothing curve",
        category="analysis",
        hint="Apply Gaussian smoothing on a rank-1 (1D) tensor, preview the smoothed curve, and pass it downstream as tensor.",
        fields=(
            IntField(key="sigma", label="Sigma", default=1),
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
        ),
        defaults=(
            ("sigma", 1),
            ("logScaleX", False),
            ("logScaleY", False),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="SmoothingCurveNode"),
    )
)
