"""rotary_embed_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="rotary_embed_layer",
        label='Rotary positional embedding',
        hint="RoPE on the last (even) feature dimension along positions on axis -2; shape-preserving in the chain.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="rotaryDim", label='Rotary Dim', default=64, min=2),
            IntField(key="thetaBase", label='Theta Base', default=10000),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="RotaryEmbedLayerNode", codegen_key="rotary_embed_layer"),
    )
)
