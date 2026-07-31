"""Explicit TinyShakespeare next-token dataset; download failure is an error, never a synthetic fallback."""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._toy_language_common import preview_toy_language
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import FrontendSpec, IntField, DatasetDef

DEF = dataset_def(
    DatasetDef(
        type="tinyshakespeare_lm_dataset",
        label="TinyShakespeare language-model dataset",
        hint="Real TinyShakespeare word-level next-token corpus. Download is required; this node never substitutes synthetic text.",
        family=("toy_language_token_dataset", "text_heavy_toy_language_dataset", "token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset"),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=256, min=8),
            IntField(key="contextLength", label="Context Length", default=32, min=1),
            IntField(key="trainSize", label="Train Size", default=4000, min=1),
            IntField(key="testSize", label="Test Size", default=0, min=0),
            IntField(key="seed", label="Seed", default=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0, sweepable=False),
            IntField(key="stride", label="Stride", default=1, min=1),
        ),
        frontend=FrontendSpec(
            component_key="GenericDatasetNode",
            codegen_key="tinyshakespeare_lm_dataset",
        ),
    )
)

dataset_preview_for(DEF)(preview_toy_language)
