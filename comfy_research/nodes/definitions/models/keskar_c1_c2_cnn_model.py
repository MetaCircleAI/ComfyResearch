"""keskar_c1_c2_cnn_model — ModelDef-channel definition + builder provider.

字段转写自 repro-reference manifest。canvas_full_model 随 canvas_trainer_model_source
携带(恰等 invariant:canvas_full_model == trainer_model_source ∖ atomic ∖ combined;
repro 的 fullModelCanvasTypes 字面表亦含本型)。builder 消费 context
(input_channels/num_classes),与 resnet 同款。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="keskar_c1_c2_cnn_model",
        label="Keskar C1/C2 CNN",
        hint="Keskar C1/C2 CNN for 32×32 CIFAR-10; batch-size sharpness repro.",
        family=("vision_model", "canvas_trainer_model_source", "canvas_full_model"),
        fields=(
            EnumField(key="architecture", label="Architecture", default="c1", options=("c1", "c2")),
            IntField(key="seed", label="Seed", default=0, min=0),
            EnumField(key="specCodeName", label="Spec Code Name", default="keskarCnnModelSpec", options=("keskarCnnModelSpec",)),
        ),
        frontend=FrontendSpec(component_key="KeskarCnnModelNode"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_keskar_c1_c2_cnn_model

    return _build_keskar_c1_c2_cnn_model(data, context)
