"""hyena_like_conv_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。所有字段均生成 sweep 轴；enum options 只进入
generatedNodeSpecs。ffMult sweep 走 int 语义(sweep_kind, runtime/_scalar_int/UI/notebook
全 int——浮点轴会造成标签与实际训练不一致)。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="hyena_like_conv_model",
        label="Hyena-like conv (tokens)",
        hint="Causal depthwise conv + gated FFN blocks; convolutional sequence LM.",
        family=("token_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            IntField(key="embedDim", label="Embed Dim", default=32, min=1),
            IntField(key="contextLength", label="Context Length", default=8, min=1),
            IntField(key="seed", label="Seed", default=0, min=0),
            IntField(key="localMixingKernel", label="Local Mixing Kernel", default=0, min=0),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="convKernel", label="Conv Kernel", default=7, min=1),
            FloatField(key="ffMult", label="Ff Mult", default=2, sweep_kind="int"),
        ),
        frontend=FrontendSpec(component_key="AlternativeArchTokenLmNode", codegen_key="hyena_like_conv_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_hyena_like_conv_model

    return _build_hyena_like_conv_model(data, context)
