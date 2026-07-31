"""bigram_low_rank_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="bigram_low_rank_dataset",
        label="Bigram low-rank dataset",
        hint="Bigram dataset with low-rank transition logits; sample x from stationary distribution then y from P(y|x).",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            IntField(key="rank", label="Rank", default=20, min=1),
            FloatField(key="logitScale", label="Logit Scale", default=1),
            FloatField(key="corruptRatio", label="Corrupt Ratio", default=0, min=0),
            FloatField(key="corruptScale", label="Corrupt Scale", default=5, min=0),
            EnumField(key="decayType", label="Decay Type", default='power_law', options=('power_law', 'exponential')),
            FloatField(key="alpha", label="Alpha", default=0),
            IntField(key="trainSize", label="Train Size", default=1200, min=1),
            IntField(key="testSize", label="Test Size", default=300, min=0),
            IntField(key="seed", label="Seed", default=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="BigramLowRankDatasetNode", codegen_key="bigram_low_rank_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
