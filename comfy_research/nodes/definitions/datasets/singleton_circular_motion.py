"""circular_motion_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="circular_motion_dataset",
        label="Circular motion dataset",
        hint="Toy physics dataset for circular trajectories; emits interleaved x/y tokens with y shifted by +N.",
        family=("token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=128, min=2),
            IntField(key="contextLength", label="Context Length", default=20, min=1),
            FloatField(key="radiusMin", label="Radius Min", default=0.15),
            FloatField(key="radiusMax", label="Radius Max", default=0.35),
            FloatField(key="angularVelocity", label="Angular Velocity", default=0.5),
            IntField(key="trainSize", label="Train Size", default=4000, min=1),
            IntField(key="testSize", label="Test Size", default=1000, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="CircularMotionDatasetNode", codegen_key="circular_motion_dataset"),
    )
)
dataset_preview_for(DEF)(preview_engine_builder)
