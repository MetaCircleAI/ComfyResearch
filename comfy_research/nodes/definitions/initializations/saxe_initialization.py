"""saxe_initialization — InitializationDef-channel thin definition。"""
from __future__ import annotations

from comfy_research.nodes.registry import initialization_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, InitializationDef

DEF = initialization_def(
    InitializationDef(
        type="saxe_initialization",
        label="Orthogonal Small-Scale Initialization",
        fields=(FloatField(key="amplitude", label="Amplitude", default=0.01, min=0, step=0.01),),
        frontend=FrontendSpec(component_key="SaxeInitializationNode", codegen_key="saxe_initialization"),
    )
)
