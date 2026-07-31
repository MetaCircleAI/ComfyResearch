"""model_weight_tensors — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="model_weight_tensors",
        label="Model weight tensors",
        category="analysis",
        hint="Materialize all model parameters (named_parameters); wire tensor list into Tensor selector to pick one.",
        defaults=(
            ("weightTensorPayloads", {}),
            ("scanMessage", None),
            ("scanSummary", None),
        ),
        frontend=FrontendSpec(component_key="ModelWeightTensorsNode"),
    )
)
