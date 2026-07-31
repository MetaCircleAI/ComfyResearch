"""numeric_transformer_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。numeric_hyena exposes no sweep axes; the other four
models derive axes from their fields. ``prepare_build_vector`` retains its
specialized ``ModelBuildContext`` construction.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="numeric_transformer_model",
        label="Transformer (numerics)",
        hint="Transformer encoder on numeric tensors [batch, T, D_in] → [batch, T, D_out]; default causal self-attention (optional full-context / bidirectional).",
        family=("vector_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="contextLength", label="Context Length", default=2, min=1),
            IntField(key="inputDim", label="Input Dim", default=1, min=1),
            IntField(key="outputDim", label="Output Dim", default=1, min=1),
            IntField(key="modelDim", label="Model Dim", default=32, min=1),
            IntField(key="numHeads", label="Num Heads", default=1, min=1),
            IntField(key="numLayers", label="Num Layers", default=1, min=1),
            IntField(key="ffDim", label="Ff Dim", default=64, min=1),
            EnumField(key="activation", label="Activation", default='gelu', options=('gelu', 'relu', 'silu')),
            EnumField(key="encoderBackend", label="Encoder Backend", default='pytorch', options=('pytorch', 'stable')),
            FloatField(key="encoderDropout", label="Encoder Dropout", default=0),
            EnumField(key="spectralNormLinears", label="Spectral Norm Linears", default='no', options=('yes', 'no')),
            EnumField(key="stableQkNorm", label="Stable Qk Norm", default='no', options=('yes', 'no')),
            FloatField(key="stableAttnTemperature", label="Stable Attn Temperature", default=1),
            FloatField(key="stableAttnLogitCap", label="Stable Attn Logit Cap", default=0),
            FloatField(key="stableAttnDropout", label="Stable Attn Dropout", default=0),
            EnumField(key="causalAttention", label="Causal Attention", default='yes', options=('yes', 'no')),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="NumericTransformerModelNode", codegen_key="numeric_transformer_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_numeric_transformer_model

    return _build_numeric_transformer_model(data, context)
