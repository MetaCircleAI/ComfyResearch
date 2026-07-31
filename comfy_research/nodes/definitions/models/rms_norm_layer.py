"""rms_norm_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import FloatField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="rms_norm_layer",
        label='RMSNorm layer',
        hint="LLaMA-style RMS normalization on the last tensor dimension (optional affine scale).",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="normalizedShape", label='Normalized Shape', default=64, min=1),
            FloatField(key="eps", label='Eps', default=1e-06),
            IntField(key="elementwiseAffine", label='Elementwise Affine', default=1, min=0),
        ),
        frontend=FrontendSpec(component_key="RmsNormLayerNode", codegen_key="rms_norm_layer"),
    )
)
