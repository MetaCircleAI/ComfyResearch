"""modular_addition_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="modular_addition_dataset",
        label="Modular addition dataset",
        hint="Grokking-style modular arithmetic: input tokens [a,b], label is (a + b) mod p.",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="modulus", label="Modulus", default=59, min=2),
            FloatField(key="trainFraction", label="Train Fraction", default=0.3, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="ModularAdditionDatasetNode", codegen_key="modular_addition_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
