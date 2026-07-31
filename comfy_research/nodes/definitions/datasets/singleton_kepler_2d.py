"""kepler_2d_dataset — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._singleton_common import (
    materialize_kepler_2d,
)
from comfy_research.nodes.registry import dataset_def, dataset_materializer_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="kepler_2d_dataset",
        label="Kepler 2D dataset",
        family=("vector_regression_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="contextLength", label="Context Length", default=8, min=1),
            IntField(key="trainSize", label="Train Size", default=1600, min=1),
            IntField(key="testSize", label="Test Size", default=400, min=0),
            FloatField(key="semiMajorAxisMin", label="Semi Major Axis Min", default=0.7),
            FloatField(key="semiMajorAxisMax", label="Semi Major Axis Max", default=1.3),
            IntField(key="eccentricityMin", label="Eccentricity Min", default=0),
            FloatField(key="eccentricityMax", label="Eccentricity Max", default=0.55),
            FloatField(key="meanMotion", label="Mean Motion", default=0.4),
            EnumField(key="outputDistribution", label="Output Distribution", default='deterministic'),
            FloatField(key="noiseLevel", label="Noise Level", default=0, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed'),
        ),
        frontend=FrontendSpec(component_key="Kepler2dDatasetNode", codegen_key="kepler_2d_dataset"),
    )
)
dataset_materializer_for(DEF)(materialize_kepler_2d)
