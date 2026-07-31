"""dataset_mixer — NodeDef-channel definition for the dataset channel.

程序化生成自 manifest。materialization keeps teacher dispatch ahead of dense
families and the mixer early return ahead of hooks. Mixer fields are not sweepable.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, InPort, PortAccept, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="dataset_mixer",
        label="Dataset mixer A",
        family=("dataset_mixer", "vector_regression_dataset", "diffusion_noise_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="trainTotalSamples", label="Train Total Samples", default=800, min=1, sweepable=False),
            IntField(key="testTotalSamples", label="Test Total Samples", default=0, min=0, sweepable=False),
            FloatField(key="proportionA", label="Proportion A", default=0.5, min=0, sweepable=False),
            IntField(key="initSeed", label="Init Seed", default=0, min=0, sweepable=False),
            EnumField(key="samplingMode", label="Sampling Mode", default='fixed', sweepable=False),
        ),
        # 声明式入口端口(原 cascade 分支:th∈{dataset_a,dataset_b} ←
        # isDatasetTensorListSource == canvas_dataset_source × 三 handle)。
        ports=(
            InPort(id="dataset_a", accepts=(PortAccept(handles=("dataset", "train_dataset", "test_dataset"), source_family="canvas_dataset_source"),)),
            InPort(id="dataset_b", accepts=(PortAccept(handles=("dataset", "train_dataset", "test_dataset"), source_family="canvas_dataset_source"),)),
        ),
        frontend=FrontendSpec(component_key="DatasetMixerNode", codegen_key="dataset_mixer"),
    )
)
