"""derivative_curve — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="derivative_curve",
        label="Derivative curve",
        category="analysis",
        hint="Compute 1st–5th finite-difference derivative on a rank-1 (1D) tensor, preview it, and pass it downstream as tensor.",
        fields=(
            EnumField(key="order", label="Order", default="1"),
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
        ),
        defaults=(
            ("order", "1"),
            ("logScaleX", False),
            ("logScaleY", False),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="DerivativeCurveNode"),
    )
)
