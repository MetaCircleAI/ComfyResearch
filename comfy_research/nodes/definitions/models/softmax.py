"""softmax — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="softmax",
        label='Softmax',
        hint="Softmax along a configurable dimension in the tensor chain.",
        family=("canvas_layer_strip_chain",),
        fields=(
            IntField(key="dimension", label='Dimension', default=-1),
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        frontend=FrontendSpec(component_key="SoftmaxNode", codegen_key="softmax"),
    )
)
