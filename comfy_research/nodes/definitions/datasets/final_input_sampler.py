"""input_sampler — NodeDef-channel definition for the dataset channel.

程序化生成自 manifest。materialization keeps teacher dispatch ahead of dense
families and the mixer early return ahead of hooks.
"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, FrontendSpec, InPort, IntField, PortAccept

DEF = dataset_def(
    DatasetDef(
        type="input_sampler",
        label="Input sampler",
        hint="Samples N inputs from a wired random-input distribution and emits a sample tensor.",
        family=None,
        fields=(
            IntField(key="numSamples", label="Num Samples", default=800, min=1),
        ),
        # 原 cascade 分支 return-style:distribution ← random_input_distribution.input_distribution。
        ports=(InPort(id="distribution", accepts=(PortAccept(handles=("input_distribution",), source_type="random_input_distribution"),)),),
        frontend=FrontendSpec(component_key="InputSamplerNode", codegen_key="input_sampler"),
    )
)
