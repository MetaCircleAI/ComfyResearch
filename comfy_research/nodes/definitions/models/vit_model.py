"""vit_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest(fields 为 defaults 推断,labels title-case)。vision 二型均
allowlist 外，因此没有 sweep 轴；builder 消费 context(input_channels/num_classes/image_size)。
specCode 使用 ``visionModelSpecCode``，组件提供 level-mode UI。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="vit_model",
        label='ViT (vision)',
        hint="Tiny patch transformer for low-res grayscale; image size must divide patch size.",
        family=('vision_model', 'canvas_trainer_model_source', 'canvas_full_model'),
        fields=(
            EnumField(key="variant", label='Variant', default='tiny', options=('tiny', 'small')),
            IntField(key="patchSize", label='Patch Size', default=4, min=2),
            IntField(key="hiddenDim", label='Hidden Dim', default=128, min=32),
            IntField(key="depth", label='Depth', default=3, min=1),
            IntField(key="numHeads", label='Num Heads', default=4, min=1),
            IntField(key="seed", label='Seed', default=0, min=0),
            EnumField(key="specCodeName", label='Spec Code Name', default='vitModelSpec', options=('vitModelSpec',)),
        ),
        frontend=FrontendSpec(component_key="VitModelNode", codegen_key="vit_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_vit_model

    return _build_vit_model(data, context)
