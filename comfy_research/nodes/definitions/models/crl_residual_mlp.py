"""crl_residual_mlp — ModelDef-channel definition + builder provider.

程序化生成自 manifest。activation 不生成 sweep 轴。family=activation_model；
该节点通过 crl_trainer 的 model handle 连接。
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="crl_residual_mlp",
        label='CRL residual MLP (actor + critic)',
        hint="Deep residual MLP for CRL actor and φ/ψ encoders (four Dense+LN+Swish stages per block).",
        family=('activation_model',),
        fields=(
            IntField(key="stateDim", label='State Dim', default=4, min=1),
            IntField(key="actionDim", label='Action Dim', default=2, min=1),
            IntField(key="goalDim", label='Goal Dim', default=2, min=1),
            IntField(key="actorWidth", label='Actor Width', default=128, min=8),
            IntField(key="criticWidth", label='Critic Width', default=128, min=8),
            IntField(key="actorDepth", label='Actor Depth', default=4, min=4),
            IntField(key="criticDepth", label='Critic Depth', default=4, min=4),
            IntField(key="embedDim", label='Embed Dim', default=64, min=8),
            EnumField(key="activation", label='Activation', default='silu', options=('relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'silu', 'identity'), sweepable=False),
            IntField(key="seed", label='Seed', default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="CrlResidualMlpNode", codegen_key="crl_residual_mlp"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_crl_residual_mlp

    return _build_crl_residual_mlp(data, context)
