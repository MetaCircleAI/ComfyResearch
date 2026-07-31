"""local_mixing_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="local_mixing_layer",
        label='Local mixing layer (Canon-lite)',
        hint="Causal depthwise conv + residual along sequence (Canon-lite); same last dim as upstream [..., T, C].",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="modelDim", label='Model Dim', default=64, min=1),
            IntField(key="kernelSize", label='Kernel Size', default=5, min=1),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="LocalMixingLayerNode", codegen_key="local_mixing_layer"),
    )
)
