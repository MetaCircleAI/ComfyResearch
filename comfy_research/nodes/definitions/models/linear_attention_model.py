"""linear_attention_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。所有字段均生成 sweep 轴；enum options 只进入
generatedNodeSpecs。Float 字段使用 floatChoices。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="linear_attention_model",
        label="Linear attention (tokens)",
        hint="ELU+1 causal linear attention LM bundle (optional DW conv mixing); last-token CE.",
        family=("token_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            IntField(key="embedDim", label="Embed Dim", default=32, min=1),
            IntField(key="contextLength", label="Context Length", default=8, min=1),
            IntField(key="seed", label="Seed", default=0, min=0),
            IntField(key="localMixingKernel", label="Local Mixing Kernel", default=0, min=0),
            IntField(key="numHeads", label="Num Heads", default=4, min=1),
            EnumField(key="causalAttention", label="Causal Attention", default='yes', options=('yes', 'no')),
        ),
        frontend=FrontendSpec(component_key="AlternativeArchTokenLmNode", codegen_key="linear_attention_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_linear_attention_model

    return _build_linear_attention_model(data, context)
