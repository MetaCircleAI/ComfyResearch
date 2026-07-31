"""mlp_token_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。三种 token MLP 不在 MODEL_SWEEP_ALLOWLIST，因此不生成
sweep 轴；enum 字段没有 options。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="mlp_token_model",
        label="MLP_token model",
        family=("token_model", "mlp_token_family", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=1),
            IntField(key="embedDim", label="Embed Dim", default=64, min=1),
            IntField(key="tokensPerInput", label="Tokens Per Input", default=1, min=1),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="width", label="Width", default=64, min=1),
            IntField(key="numExperts", label="Num Experts", default=4, min=1),
            EnumField(key="activation", label="Activation", default='relu'),
            EnumField(key="tieWeights", label="Tie Weights", default='yes'),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="MlpTokenModelNode", codegen_key="mlp_token_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_mlp_token_model

    return _build_mlp_token_model(data, context)
