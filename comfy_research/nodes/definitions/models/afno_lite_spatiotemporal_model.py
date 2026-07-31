"""afno_lite_spatiotemporal_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。numeric_hyena exposes no sweep axes; the other four
models derive axes from their fields. ``prepare_build_vector`` retains its
specialized ``ModelBuildContext`` construction.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="afno_lite_spatiotemporal_model",
        label="AFNO-lite spatiotemporal model",
        hint="AFNO-lite composite model: patchify + low-frequency spectral mixer + feedforward blocks over spatiotemporal tokens, then decode back to flat fields.",
        family=("vector_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="contextFrames", label="Context Frames", default=4, min=1),
            IntField(key="channels", label="Channels", default=1, min=1),
            IntField(key="gridSize", label="Grid Size", default=16, min=4),
            IntField(key="inputDim", label="Input Dim", default=1024, min=1),
            IntField(key="outputDim", label="Output Dim", default=1024, min=1),
            IntField(key="patchSize", label="Patch Size", default=4, min=1),
            IntField(key="embedDim", label="Embed Dim", default=64, min=1),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="numHeads", label="Num Heads", default=4, min=1),
            FloatField(key="ffRatio", label="Ff Ratio", default=2),
            FloatField(key="dropout", label="Dropout", default=0),
            IntField(key="numSpectralBlocks", label="Num Spectral Blocks", default=1, min=1),
            IntField(key="maxFrequencyModes", label="Max Frequency Modes", default=4, min=1),
            FloatField(key="spectralShrinkFactor", label="Spectral Shrink Factor", default=1),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="AfnoLiteSpatiotemporalModelNode", codegen_key="afno_lite_spatiotemporal_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_afno_lite_spatiotemporal_model

    return _build_afno_lite_spatiotemporal_model(data, context)
