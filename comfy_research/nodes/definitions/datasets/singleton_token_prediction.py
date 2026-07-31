"""token_prediction_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="token_prediction_dataset",
        label="Token Retrieval (position/content) dataset",
        hint="Random token sequences; choose retrieval by fixed position index or by nearest prior content to the last token.",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            EnumField(key="retrievalMode", label="Retrieval Mode", default='position'),
            IntField(key="vocabSize", label="Vocab Size", default=4, min=2),
            IntField(key="contextLength", label="Context Length", default=4, min=1),
            IntField(key="whichToken", label="Which Token", default=-1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="TokenPredictionDatasetNode", codegen_key="token_prediction_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
