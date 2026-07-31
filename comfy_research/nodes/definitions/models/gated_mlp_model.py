"""gated_mlp_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。builder provider lazily references the corresponding
MODEL_BUILDERS function. Activation options only enter generatedNodeSpecs;
outputScale is included when present.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="gated_mlp_model",
        label="Gated MLP",
        hint="MLP with multiplicative gating per hidden block: act(Wg x) * (Wv x).",
        family=("mlp_family", "vector_model", "activation_model", "silu_default_activation_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="inputDim", label="Input Dim", default=10, min=1),
            IntField(key="outputDim", label="Output Dim", default=1, min=1),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="width", label="Width", default=64, min=1),
            EnumField(key="activation", label="Activation", default='silu', options=('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'silu', 'identity')),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="GatedMlpModelNode", codegen_key="gated_mlp_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_gated_mlp_model

    return _build_gated_mlp_model(data, context)
