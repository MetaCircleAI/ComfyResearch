"""tensor_slicing — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_slicing",
        label="Tensor slicing",
        category="analysis",
        hint="Slice one tensor dimension by selected indices. One index collapses that dimension; multiple keep it with new size.",
        family=("canvas_single_tensor_target", "canvas_comment_source"),
        defaults=(
            ("slices", [{"dimension": 0, "indices": "0"}]),
        ),
        frontend=FrontendSpec(component_key="TensorSlicingNode"),
    )
)
