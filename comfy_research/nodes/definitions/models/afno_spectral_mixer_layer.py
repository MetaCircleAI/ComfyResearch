"""afno_spectral_mixer_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import FloatField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="afno_spectral_mixer_layer",
        label='AFNO spectral mixer layer',
        hint="AFNO atomic layer: low-frequency spectral mixing on flattened spatiotemporal fields.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="contextFrames", label='Context Frames', default=4, min=1),
            IntField(key="channels", label='Channels', default=1, min=1),
            IntField(key="gridSize", label='Grid Size', default=16, min=4),
            IntField(key="inputDim", label='Input Dim', default=1024, min=1),
            IntField(key="outputDim", label='Output Dim', default=1024, min=1),
            IntField(key="patchSize", label='Patch Size', default=4, min=1),
            IntField(key="embedDim", label='Embed Dim', default=64, min=8),
            IntField(key="numHeads", label='Num Heads', default=4, min=1),
            FloatField(key="ffRatio", label='Ff Ratio', default=2, min=0.25),
            FloatField(key="dropout", label='Dropout', default=0, min=0),
            IntField(key="numSpectralBlocks", label='Num Spectral Blocks', default=1, min=1),
            IntField(key="maxFrequencyModes", label='Max Frequency Modes', default=4, min=1),
            FloatField(key="spectralShrinkFactor", label='Spectral Shrink Factor', default=1, min=0.01),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="AfnoAtomicLayerNode", codegen_key="afno_spectral_mixer_layer"),
    )
)
