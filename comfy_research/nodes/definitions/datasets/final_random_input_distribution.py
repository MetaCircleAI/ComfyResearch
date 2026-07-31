"""random_input_distribution — NodeDef-channel definition for the dataset channel.

程序化生成自 manifest。materialization keeps teacher dispatch ahead of dense
families and the mixer early return ahead of hooks.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="random_input_distribution",
        label="Random input distribution",
        hint="Random x generator settings (base law + optional Gaussian jitter) for a teacher dataset.",
        family=None,
        fields=(
            IntField(key="inputDim", label="Input Dim", default=10, min=1),
            EnumField(key="inputDistribution", label="Input Distribution", default='standard_normal'),
            EnumField(key="noiseDistribution", label="Noise Distribution", default='deterministic'),
            FloatField(key="noiseLevel", label="Noise Level", default=0, min=0),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="RandomInputDistributionNode", codegen_key="random_input_distribution"),
    )
)
