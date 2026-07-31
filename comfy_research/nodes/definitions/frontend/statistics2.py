"""statistics2 — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="statistics2",
        label="Statistics 2 (pair)",
        category="analysis",
        hint="Two tensors: NumPy-style einsum pair; dot or cosine over contracted indices.",
        fields=(
            EnumField(key="einsumSubscripts", label="Einsum Subscripts", default="ij,ik->jk"),
            EnumField(key="pairReduction", label="Pair Reduction", default="dot"),
        ),
        defaults=(
            ("einsumSubscripts", "ij,ik->jk"),
            ("pairReduction", "dot"),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="Statistics2Node"),
    )
)
