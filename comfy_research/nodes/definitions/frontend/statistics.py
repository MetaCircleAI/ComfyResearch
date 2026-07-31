"""statistics — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="statistics",
        label="Statistics",
        category="analysis",
        hint="Einstein / einsum subscripts on one tensor; choose mean, max, … for collapsed axes.",
        family=("observable_user_tensor_transform", "canvas_single_tensor_target"),
        fields=(
            EnumField(key="einsumSubscripts", label="Einsum Subscripts", default="ab->b"),
            EnumField(key="reductionOp", label="Reduction Op", default="mean"),
        ),
        defaults=(
            ("einsumSubscripts", "ab->b"),
            ("reductionOp", "mean"),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="StatisticsNode"),
    )
)
