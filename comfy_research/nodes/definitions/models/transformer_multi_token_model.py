"""transformer_multi_token_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。所有字段均生成 sweep 轴，六个 enum 字段的 options
只进入 generatedNodeSpecs。prepare_graph enforces the multi-token
circular_motion_dataset compatibility constraint.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="transformer_multi_token_model",
        label="Transformer (multiple tokens)",
        hint="K token ids per timestep [batch, L, K], fused per position, TransformerEncoder → last-timestep logits [batch, K, V]. For circular motion use K = 2.",
        family=("token_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            IntField(key="contextLength", label="Context Length", default=4, min=1),
            IntField(key="tokensPerPosition", label="Tokens Per Position", default=2, min=1),
            IntField(key="modelDim", label="Model Dim", default=32, min=1),
            IntField(key="numHeads", label="Num Heads", default=1, min=1),
            IntField(key="numLayers", label="Num Layers", default=1, min=1),
            IntField(key="ffDim", label="Ff Dim", default=64, min=1),
            EnumField(key="encoderBackend", label="Encoder Backend", default='pytorch', options=('pytorch', 'stable')),
            FloatField(key="encoderDropout", label="Encoder Dropout", default=0),
            EnumField(key="spectralNormLinears", label="Spectral Norm Linears", default='no', options=('yes', 'no')),
            FloatField(key="lmLogitScale", label="Lm Logit Scale", default=1),
            EnumField(key="stableQkNorm", label="Stable Qk Norm", default='no', options=('yes', 'no')),
            FloatField(key="stableAttnTemperature", label="Stable Attn Temperature", default=1),
            FloatField(key="stableAttnLogitCap", label="Stable Attn Logit Cap", default=0),
            FloatField(key="stableAttnDropout", label="Stable Attn Dropout", default=0),
            EnumField(key="tieEmbeddingLmHead", label="Tie Embedding Lm Head", default='no', options=('yes', 'no')),
            EnumField(key="causalAttention", label="Causal Attention", default='yes', options=('yes', 'no')),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="TransformerMultiTokenModelNode", codegen_key="transformer_multi_token_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_transformer_multi_token_model

    return _build_transformer_multi_token_model(data, context)
