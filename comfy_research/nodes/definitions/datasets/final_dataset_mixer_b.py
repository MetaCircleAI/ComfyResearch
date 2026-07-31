"""dataset_mixer_b — NodeDef-channel definition for the dataset channel.

程序化生成自 manifest。materialization keeps teacher dispatch ahead of dense
families and the mixer early return ahead of hooks. Mixer fields are not sweepable.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, InPort, PortAccept, FloatField, FrontendSpec

DEF = dataset_def(
    DatasetDef(
        type="dataset_mixer_b",
        label="Dataset mixer B",
        family=("dataset_mixer", "vector_regression_dataset", "diffusion_noise_dataset", "canvas_dataset_source", "canvas_activation_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            FloatField(key="interpolationLambda", label="Interpolation Lambda", default=0.5, min=0, sweepable=False),
        ),
        # 声明式入口端口(原 cascade 分支:th∈{dataset_a,dataset_b} ←
        # isDatasetTensorListSource == canvas_dataset_source × 三 handle)。
        ports=(
            InPort(id="dataset_a", accepts=(PortAccept(handles=("dataset", "train_dataset", "test_dataset"), source_family="canvas_dataset_source"),)),
            InPort(id="dataset_b", accepts=(PortAccept(handles=("dataset", "train_dataset", "test_dataset"), source_family="canvas_dataset_source"),)),
        ),
        frontend=FrontendSpec(component_key="DatasetMixerBNode", codegen_key="dataset_mixer_b"),
    )
)
