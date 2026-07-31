"""mpp_spatiotemporal_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。numeric_hyena exposes no sweep axes; the other four
models derive axes from their fields. ``prepare_build_vector`` retains its
specialized ``ModelBuildContext`` construction.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="mpp_spatiotemporal_model",
        label="MPP-style spatiotemporal ViT",
        hint="MPP-like ViT: patchify frames, Transformer over space-time tokens, unpatch; matches arXiv:2310.02994-style field normalization.",
        family=("vector_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="contextFrames", label="Context Frames", default=4, min=1),
            IntField(key="channels", label="Channels", default=1, min=1),
            IntField(key="gridSize", label="Grid Size", default=16, min=4),
            IntField(key="inputDim", label="Input Dim", default=1024, min=1),
            IntField(key="outputDim", label="Output Dim", default=1024, min=1),
            IntField(key="patchSize", label="Patch Size", default=4, min=1),
            IntField(key="embedDim", label="Embed Dim", default=128, min=1),
            IntField(key="depth", label="Depth", default=4, min=1),
            IntField(key="numHeads", label="Num Heads", default=4, min=1),
            FloatField(key="ffRatio", label="Ff Ratio", default=4),
            FloatField(key="dropout", label="Dropout", default=0),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="MppSpatiotemporalModelNode", codegen_key="mpp_spatiotemporal_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_mpp_spatiotemporal_model

    return _build_mpp_spatiotemporal_model(data, context)
