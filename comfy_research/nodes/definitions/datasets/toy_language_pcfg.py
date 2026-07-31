"""pcfg_dataset — NodeDef-channel definition + preview.

程序化生成自 committed manifest(字段 kind/label/序字节保真);sweep 元数据见
_toy_language_common 定义。materialize 由 token capability 分支及共享的
ctx-override/text-heavy fallback 处理。
"""
from __future__ import annotations

from comfy_research.nodes.definitions.datasets._toy_language_common import preview_toy_language
from comfy_research.nodes.registry import dataset_def, dataset_preview_for
from comfy_research.nodes.schema import DatasetDef, EnumField, FloatField, FrontendSpec, IntField

DEF = dataset_def(
    DatasetDef(
        type="pcfg_dataset",
        label="PCFG toy LM dataset",
        hint="PCFG toy LM: binary-tree legacy mode or cfg_sentence (weighted rules + lexicon) → padded next-token windows.",
        family=("toy_language_token_dataset", "token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=32, min=2),
            IntField(key="contextLength", label="Context Length", default=16, min=1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0, sweepable=False),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="dataSource", label="Data Source", default="synthetic", options=('synthetic', 'download')),
            EnumField(key="cacheDir", label="Cache Dir", default=""),
            EnumField(key="inspectFormat", label="Inspect Format", default="id"),
            EnumField(key="pcfgGenMode", label="Pcfg Gen Mode", default="binary_tree", options=('binary_tree', 'cfg_sentence')),
            EnumField(key="pcfgGrammarId", label="Pcfg Grammar Id", default="world_model"),
            IntField(key="pcfgMaxDepth", label="Pcfg Max Depth", default=8, min=1),
            FloatField(key="pcfgTermProb", label="Pcfg Term Prob", default=0.35, min=0.05, max=0.95),
        ),
        frontend=FrontendSpec(component_key="ToyLanguageLmDatasetNode", codegen_key="pcfg_dataset"),
    )
)

dataset_preview_for(DEF)(preview_toy_language)
