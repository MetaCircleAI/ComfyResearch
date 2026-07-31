"""in_context_associative_recall_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="in_context_associative_recall_dataset",
        label="In-context associative recall dataset",
        hint="Associative recall sequences: key-value pairs plus a query key; target is the mapped value token.",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=64, min=2),
            IntField(key="numPairs", label="Num Pairs", default=32, min=1),
            IntField(key="inContextRepeat", label="In Context Repeat", default=1, min=1),
            FloatField(key="crossSampleRepeatProb", label="Cross Sample Repeat Prob", default=0, min=0),
            IntField(key="repeatedTokenCount", label="Repeated Token Count", default=2, min=0),
            IntField(key="trainSize", label="Train Size", default=10000, min=1),
            IntField(key="testSize", label="Test Size", default=2000, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="InContextAssociativeRecallDatasetNode", codegen_key="in_context_associative_recall_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
