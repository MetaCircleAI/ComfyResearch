"""vgg11_cifar_model — ModelDef-channel definition + builder provider.

字段转写自 repro-reference manifest。canvas_full_model 随 canvas_trainer_model_source
携带(恰等 invariant;repro 的 fullModelCanvasTypes 字面表亦含本型)。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="vgg11_cifar_model",
        label="VGG-11 (CIFAR)",
        hint="VGG-11 style stack for CIFAR-10 (32×32 RGB).",
        family=("vision_model", "canvas_trainer_model_source", "canvas_full_model"),
        fields=(
            IntField(key="seed", label="Seed", default=0, min=0),
            EnumField(key="specCodeName", label="Spec Code Name", default="vgg11CifarModelSpec", options=("vgg11CifarModelSpec",)),
        ),
        frontend=FrontendSpec(component_key="Vgg11CifarModelNode"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_vgg11_cifar_model

    return _build_vgg11_cifar_model(data, context)
