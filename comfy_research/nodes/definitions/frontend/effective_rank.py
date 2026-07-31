"""effective_rank — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="effective_rank",
        label="Effective rank",
        category="analysis",
        hint="Scalar effective rank (entropy over singular values) of an upstream matrix tensor.",
        family=("observable_user_tensor_transform",),
        defaults=(
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="EffectiveRankNode"),
    )
)
