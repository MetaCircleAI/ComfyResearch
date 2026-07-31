"""symmetrized_mlp_init — InitializationDef-channel thin definition。"""
from __future__ import annotations

from comfy_research.nodes.registry import initialization_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, InitializationDef

DEF = initialization_def(
    InitializationDef(
        type="symmetrized_mlp_init",
        label="Symmetrized MLP init",
        fields=(FloatField(key="tau", label="Tau", default=1, min=0, step=0.01),),
        frontend=FrontendSpec(component_key="SymmetrizedMlpInitNode", codegen_key="symmetrized_mlp_init"),
    )
)
