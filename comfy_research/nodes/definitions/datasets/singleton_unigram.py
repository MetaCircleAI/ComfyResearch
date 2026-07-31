"""unigram_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="unigram_dataset",
        label="Unigram dataset",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=100, min=2),
            EnumField(key="outputDistribution", label="Output Distribution", default='power_law_class_probs', options=('uniform_class_probs', 'power_law_class_probs', 'exponential_class_probs')),
            FloatField(key="alpha", label="Alpha", default=1, min=0.01),
            IntField(key="contextLength", label="Context Length", default=1, min=1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="UnigramDatasetNode", codegen_key="unigram_dataset"),
    )
)

