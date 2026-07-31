"""tensor_reader — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_reader",
        label="Tensor reader",
        category="analysis",
        hint="Connect a tensor and open View tensor values for nested JSON (read-only), like model parameters.",
        defaults=(),  # emptyDefaults 实况:hasDefaults:true、defaults {}
        frontend=FrontendSpec(component_key="TensorReaderNode"),
    )
)
