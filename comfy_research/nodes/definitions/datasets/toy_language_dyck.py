"""dyck_dataset — NodeDef-channel definition + preview.

程序化生成自 committed manifest(字段 kind/label/序字节保真);sweep 元数据见
_toy_language_common 定义。materialize 由 token capability 分支及共享的
ctx-override/text-heavy fallback 处理。
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._toy_language_common import preview_toy_language
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="dyck_dataset",
        label="Dyck language dataset",
        hint="Balanced bracket strings as token ids (Dyck-style); cross-entropy LM probe.",
        family=("toy_language_token_dataset", "token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=2, min=2, sweepable=False),
            IntField(key="contextLength", label="Context Length", default=16, min=1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0, sweepable=False),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="dataSource", label="Data Source", default="synthetic", options=('synthetic', 'download')),
            EnumField(key="cacheDir", label="Cache Dir", default=""),
            EnumField(key="inspectFormat", label="Inspect Format", default="id"),
            IntField(key="numBracketTypes", label="Num Bracket Types", default=1, min=1),
            IntField(key="maxNestingDepth", label="Max Nesting Depth", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="ToyLanguageLmDatasetNode", codegen_key="dyck_dataset"),
    )
)

dataset_preview_for(DEF)(preview_toy_language)
