"""unembedding_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="unembedding_layer",
        label='Unembedding layer',
        hint="LM-head style torch.nn.Linear(d_model → vocab); pair after blocks that emit hidden activations.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="inFeatures", label='In Features', default=64, min=1),
            IntField(key="outFeatures", label='Out Features', default=4096, min=1),
            IntField(key="bias", label='Bias', default=1, min=0),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="UnembeddingLayerNode", codegen_key="unembedding_layer"),
    )
)
