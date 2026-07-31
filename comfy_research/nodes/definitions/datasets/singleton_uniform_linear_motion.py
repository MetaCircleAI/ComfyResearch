"""uniform_linear_motion_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    materialize_uniform_linear_motion,
    preview_engine_builder,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="uniform_linear_motion_dataset",
        label="Uniform linear motion dataset",
        hint="Physics toy dataset for x_{t+1}=2x_t-x_{t-1}: samples x0 and velocity, input is [x0, x1], target is x2.",
        family=("vector_regression_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="contextLength", label="Context Length", default=2, min=1),
            IntField(key="positionDim", label="Position Dim", default=1, min=1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            EnumField(key="positionDistribution", label="Position Distribution", default='standard_normal', options=('standard_normal', 'uniform_neg1_1', 'uniform_0_1')),
            EnumField(key="velocityDistribution", label="Velocity Distribution", default='standard_normal', options=('standard_normal', 'uniform_neg1_1', 'uniform_0_1')),
            FloatField(key="velocityScale", label="Velocity Scale", default=1),
            EnumField(key="outputDistribution", label="Output Distribution", default='deterministic', options=('additive_gaussian', 'deterministic')),
            FloatField(key="noiseLevel", label="Noise Level", default=0, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="UniformLinearMotionDatasetNode", codegen_key="uniform_linear_motion_dataset"),
    )
)
dataset_materializer_for(DEF)(materialize_uniform_linear_motion)
dataset_preview_for(DEF)(preview_engine_builder)
