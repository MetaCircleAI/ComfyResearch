"""embedding_layer — ModelDef-channel thin definition.

程序化生成自 manifest。Atomic-chain nodes have no standalone builder provider;
``atomic_layer_chain`` dispatches them. This node exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="embedding_layer",
        label='Embedding layer',
        hint="torch.nn.Embedding: left tensor = index map (long at runtime; float batches are cast), right tensor = float vectors.",
        family=('atomic_layer_model', 'canvas_trainer_model_source'),
        fields=(
            IntField(key="numEmbeddings", label='Num Embeddings', default=4096, min=1),
            IntField(key="embeddingDim", label='Embedding Dim', default=64, min=1),
            IntField(key="numIndexColumns", label='Num Index Columns', default=1, min=1),
            IntField(key="paddingIdx", label='Padding Idx', default=-1, min=-1),
            IntField(key="scaleGradByFreq", label='Scale Grad By Freq', default=0, min=0),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="EmbeddingLayerNode", codegen_key="embedding_layer"),
    )
)
