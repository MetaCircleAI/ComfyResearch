"""phi1_style_dataset — NodeDef-channel definition + preview.

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
        type="phi1_style_dataset",
        label="phi-1 style corpus dataset",
        hint="phi-1–style mixed textbook/QA/code toy corpus LM (synthetic / optional URL).",
        family=("toy_language_token_dataset", "text_heavy_toy_language_dataset", "token_classification_dataset", "canvas_dataset_source", "canvas_trainer_autoconnect_dataset",),
        fields=(
            IntField(key="vocabSize", label="Vocab Size", default=256, min=2),
            IntField(key="contextLength", label="Context Length", default=96, min=1),
            IntField(key="trainSize", label="Train Size", default=800, min=1),
            IntField(key="testSize", label="Test Size", default=200, min=0),
            IntField(key="seed", label="Seed", default=0),
            IntField(key="initSeed", label="Init Seed", default=0, min=0, sweepable=False),
            EnumField(key="samplingMode", label="Sampling Mode", default="fixed"),
            EnumField(key="dataSource", label="Data Source", default="synthetic", options=('synthetic', 'download')),
            EnumField(key="cacheDir", label="Cache Dir", default=""),
            EnumField(key="inspectFormat", label="Inspect Format", default="id"),
            FloatField(key="vocabCap", label="Vocab Cap", default=256, sweepable=False),
            EnumField(key="tokenizerMode", label="Tokenizer Mode", default="char", options=('char', 'word')),
            IntField(key="seqLen", label="Seq Len", default=600),
            IntField(key="stride", label="Stride", default=96, min=1),
            EnumField(key="domainMix", label="Domain Mix", default="mixed"),
        ),
        frontend=FrontendSpec(component_key="ToyLanguageLmDatasetNode", codegen_key="phi1_style_dataset"),
    )
)

dataset_preview_for(DEF)(preview_toy_language)
