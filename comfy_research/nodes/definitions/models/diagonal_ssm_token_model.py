"""diagonal_ssm_token_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。所有字段均生成 sweep 轴；enum options 只进入
generatedNodeSpecs。Float 字段使用 floatChoices。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="diagonal_ssm_token_model",
        label="Diagonal SSM (tokens)",
        hint="Diagonal input-dependent SSM recurrence stacked with residuals; token LM.",
        family=("token_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            IntField(key="embedDim", label="Embed Dim", default=32, min=1),
            IntField(key="contextLength", label="Context Length", default=8, min=1),
            IntField(key="seed", label="Seed", default=0, min=0),
            IntField(key="localMixingKernel", label="Local Mixing Kernel", default=0, min=0),
            IntField(key="numLayers", label="Num Layers", default=2, min=1),
        ),
        frontend=FrontendSpec(component_key="AlternativeArchTokenLmNode", codegen_key="diagonal_ssm_token_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_diagonal_ssm_token_model

    return _build_diagonal_ssm_token_model(data, context)
