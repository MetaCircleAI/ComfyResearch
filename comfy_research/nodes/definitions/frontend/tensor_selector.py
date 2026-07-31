"""tensor_selector — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_selector",
        label="Tensor selector",
        category="analysis",
        hint="Pick which tensor from upstream (activations, model weights, losses, …) to visualize or aggregate.",
        fields=(
            EnumField(key="selectedTensorKey", label="Selected Tensor Key", default=""),
            BoolField(key="tensorSelectorSweeping", label="Tensor Selector Sweeping", default=False),
            IntField(key="tensorSelectorSweepSeq", label="Tensor Selector Sweep Seq", default=0),
        ),
        defaults=(
            ("selectedTensorKey", ""),
            ("activationTensorCache", None),
            ("activationTensorCaches", {}),
            ("tensorSelectorSweeping", False),
            ("tensorSelectorSweepSeq", 0),
        ),
        frontend=FrontendSpec(component_key="TensorSelectorNode"),
    )
)
