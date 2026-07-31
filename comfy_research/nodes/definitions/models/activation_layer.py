"""activation_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes. leakyP
has min=-1 (component clamp); bias-like integer fields remain integers.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="activation_layer",
        label='Activation layer',
        hint="torch nonlinearity module (ReLU, GELU, …) between layers — not the analysis Activation node.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            EnumField(key="activation", label='Activation', default='relu', options=('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'silu', 'identity')),
            FloatField(key="leakyP", label='Leaky P', default=0, min=-1),
        ),
        frontend=FrontendSpec(component_key="ActivationLayerNode", codegen_key="activation_layer"),
    )
)
