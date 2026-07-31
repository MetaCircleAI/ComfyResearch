"""crl_env_config — NodeDef-channel definition。程序化生成自 manifest。"""
from __future__ import annotations

from comfy_research.nodes.registry import dataset_def
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="crl_env_config",
        label="CRL environment (U4 maze / point)",
        hint="CRL navigation env preset (point U4 maze or Ant U4–style sizing; server-side sim).",
        family=None,
        fields=(
            EnumField(key="preset", label="Preset", default='point_u4_maze'),
            IntField(key="numEnvs", label="Num Envs", default=8, min=1),
            IntField(key="episodeLength", label="Episode Length", default=200, min=16),
            IntField(key="mazeSizeScaling", label="Maze Size Scaling", default=4, min=1),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="CrlEnvConfigNode", codegen_key="crl_env_config"),
    )
)

