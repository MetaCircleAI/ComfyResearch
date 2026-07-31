"""layer_norm_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import FloatField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="layer_norm_layer",
        label='LayerNorm layer',
        hint="torch.nn.LayerNorm over the trailing normalized_shape dimensions.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="normalizedShape", label='Normalized Shape', default=64, min=1),
            FloatField(key="eps", label='Eps', default=1e-05),
            IntField(key="elementwiseAffine", label='Elementwise Affine', default=1, min=0),
        ),
        frontend=FrontendSpec(component_key="LayerNormLayerNode", codegen_key="layer_norm_layer"),
    )
)
