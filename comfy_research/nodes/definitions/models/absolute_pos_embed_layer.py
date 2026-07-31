"""absolute_pos_embed_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="absolute_pos_embed_layer",
        label='Absolute positional embedding',
        hint="Learned [max_seq, dim] bias added along the sequence axis after token embeddings (same last dim).",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="maxSeqLen", label='Max Seq Len', default=512, min=1),
            IntField(key="embeddingDim", label='Embedding Dim', default=64, min=1),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="AbsolutePosEmbedLayerNode", codegen_key="absolute_pos_embed_layer"),
    )
)
