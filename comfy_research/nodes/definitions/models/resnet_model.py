"""resnet_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest(fields 为 defaults 推断,labels title-case)。vision 二型均
allowlist 外，因此没有 sweep 轴；builder 消费 context(input_channels/num_classes)。
specCode 使用 ``visionModelSpecCode``，组件提供 level-mode UI。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="resnet_model",
        label='ResNet (vision)',
        hint="Small ResNet-style CNN for 1×H×W inputs; use with vision datasets + cross-entropy.",
        family=('vision_model', 'canvas_trainer_model_source', 'canvas_full_model'),
        fields=(
            EnumField(key="variant", label='Variant', default='resnet18', options=('resnet18', 'resnet34', 'self_defined')),
            IntField(key="baseChannels", label='Base Channels', default=32, min=8),
            IntField(key="blocksStage1", label='Blocks Stage1', default=2, min=1),
            IntField(key="blocksStage2", label='Blocks Stage2', default=2, min=1),
            IntField(key="blocksStage3", label='Blocks Stage3', default=2, min=1),
            IntField(key="blocksStage4", label='Blocks Stage4', default=2, min=1),
            IntField(key="kernelSize", label='Kernel Size', default=3, min=3),
            IntField(key="seed", label='Seed', default=0, min=0),
            EnumField(key="specCodeName", label='Spec Code Name', default='resnetModelSpec', options=('resnetModelSpec',)),
        ),
        frontend=FrontendSpec(component_key="ResnetModelNode", codegen_key="resnet_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_resnet_model

    return _build_resnet_model(data, context)
