"""circle_random_walk_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="circle_random_walk_dataset",
        label="Circle random walk dataset",
        hint="Bigram random walk on a circular vocabulary: from each token, move left/right with configurable bias.",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=10, min=2),
            IntField(key="contextLength", label="Context Length", default=1, min=1),
            FloatField(key="rightStepProb", label="Right Step Prob", default=0.5, min=0),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="CircleRandomWalkDatasetNode", codegen_key="circle_random_walk_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
