"""numeric_hyena_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。numeric_hyena exposes no sweep axes; the other four
models derive axes from their fields. ``prepare_build_vector`` retains its
specialized ``ModelBuildContext`` construction.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="numeric_hyena_model",
        label="Hyena-like LM (numerics)",
        family=("vector_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="contextLength", label="Context Length", default=8, min=1),
            IntField(key="inputDim", label="Input Dim", default=2, min=1),
            IntField(key="outputDim", label="Output Dim", default=2, min=1),
            IntField(key="modelDim", label="Model Dim", default=64, min=1),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="convKernel", label="Conv Kernel", default=7, min=1),
            FloatField(key="ffMult", label="Ff Mult", default=2, sweep_kind="int"),
            IntField(key="localMixingKernel", label="Local Mixing Kernel", default=0, min=0),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="NumericHyenaModelNode", codegen_key="numeric_hyena_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_numeric_hyena_model

    return _build_numeric_hyena_model(data, context)
