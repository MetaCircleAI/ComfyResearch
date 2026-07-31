"""regressor — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="regressor",
        label="Regressor",
        category="analysis",
        hint="Regressor node for analysis workflows.",
        family=("canvas_single_tensor_target",),
        fields=(
            IntField(key="fitNonce", label="Fit Nonce", default=0),
        ),
        defaults=(
            ("fitNonce", 0),
        ),
        frontend=FrontendSpec(component_key="RegressorNode"),
    )
)
