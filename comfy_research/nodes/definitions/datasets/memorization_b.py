"""memorization_b_dataset — NodeDef-channel definition + preview.

Materialization shares the task-dependent dense/token path with mem_a.
inputDim/outputDim 是 vocabSize 的镜像写入目标 → sweepable=False(防组合爆炸);
inputDistribution 在 UI 中隐藏且不生成 sweep 轴；noiseLevel 可 sweep。
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._linear_common import (
    MEM_B_FAMILY,
    MEM_OUTPUT_DIST_OPTIONS,
    preview_direct_arrays,
)
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

MEMORIZATION_B = dataset_def(
    DatasetDef(
        type="memorization_b_dataset",
        label="Memorization B dataset",
        hint="Memory 1 dataset B: random input class + random output class (one-hot x); cross-entropy only, vs continuous-x dataset A.",
        family=MEM_B_FAMILY,
        fields=(
            IntField(key="inputDim", label="Input Dim", default=40, min=1, sweepable=False),
            IntField(key="outputDim", label="Output Dim", default=40, min=1, sweepable=False),
            IntField(key="vocabSize", label="Vocab Size", default=40, min=2),
            EnumField(key="inputDistribution", label="Input Distribution", default="standard_normal", sweepable=False),
            EnumField(key="outputDistribution", label="Output Distribution", default="uniform_class_probs", options=MEM_OUTPUT_DIST_OPTIONS),
            IntField(key="trainSize", label="Train Size", default=160, min=1),
            IntField(key="testSize", label="Test Size", default=0, min=0),
            FloatField(key="noiseLevel", label="Noise Level", default=0, min=0),
            FloatField(key="alpha", label="Alpha", default=1, min=0),
            IntField(key="seed", label="Seed", default=0),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="specCodeName", label="Spec Code Name", default="Memorization_B_Dataset"),
        ),
        frontend=FrontendSpec(component_key="LinearDatasetNode", codegen_key="memorization_b_dataset"),
    )
)

dataset_preview_for(MEMORIZATION_B)(preview_direct_arrays)
