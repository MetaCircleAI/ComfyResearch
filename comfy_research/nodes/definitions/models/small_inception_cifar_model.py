"""Small Inception CIFAR model definition."""

from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="small_inception_cifar_model",
        label="Small Inception (CIFAR)",
        hint=(
            "Small Inception used in Zhang et al.'s CIFAR experiments. It accepts the default "
            "32×32 CIFAR inputs; use the CIFAR paper whitening preprocessing to match the paper's "
            "28×28 protocol. 1,649,402 trainable parameters for 10 classes."
        ),
        family=("vision_model", "canvas_trainer_model_source", "canvas_full_model"),
        fields=(
            IntField(key="seed", label="Seed", default=0, min=0),
            EnumField(key="specCodeName", label="Spec Code Name", default="smallInceptionCifarModelSpec", options=("smallInceptionCifarModelSpec",)),
        ),
        frontend=FrontendSpec(component_key="Vgg11CifarModelNode"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_small_inception_cifar_model

    return _build_small_inception_cifar_model(data, context)
