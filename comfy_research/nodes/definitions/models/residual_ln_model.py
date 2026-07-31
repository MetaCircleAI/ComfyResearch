"""residual_ln_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。activation_model 族；lnMode 支持 pre_ln/post_ln。
该类型不生成 sweep 轴。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FloatField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="residual_ln_model",
        label='Residual LN model',
        hint="Composable residual model with Pre-LN/Post-LN mode and alpha-scaled FC2 blocks.",
        family=('activation_model', 'canvas_trainer_model_source', 'canvas_full_model'),
        fields=(
            IntField(key="dim", label='Dim', default=256, min=1),
            IntField(key="depth", label='Depth', default=100, min=1),
            FloatField(key="alpha", label='Alpha', default=1),
            EnumField(key="lnMode", label='Ln Mode', default='pre_ln', options=('pre_ln', 'post_ln')),
            EnumField(key="activation", label='Activation', default='relu', options=('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'silu', 'identity')),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="ResidualLnModelNode", codegen_key="residual_ln_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_residual_ln_model

    return _build_residual_ln_model(data, context)
