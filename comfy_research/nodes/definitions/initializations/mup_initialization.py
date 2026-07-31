"""mup_initialization — InitializationDef-channel thin definition.

This node has no provider: ``POST /api/train`` applies μP re-initialization.
It has no fields or defaults.
"""
from __future__ import annotations

from comfy_research.nodes.registry import initialization_def
from comfy_research.nodes.schema import FrontendSpec, InitializationDef

DEF = initialization_def(
    InitializationDef(
        type="mup_initialization",
        label="MuP initialization",
        hint="μP-style weight init (Linear / Embedding / LayerNorm). Wire into a model’s initialization target before training.",
        frontend=FrontendSpec(component_key="MupInitializationNode", codegen_key="mup_initialization"),
    )
)
