"""einsum — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="einsum",
        label='Einsum',
        hint="Einstein summation / einsum equation on the tensor chain.",
        family=None,
        fields=(
            EnumField(key="equation", label='Equation', default='b h t d, b h s d -> b h t s', options=('b h t d, b h s d -> b h t s',)),
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        frontend=FrontendSpec(component_key="EinsumNode", codegen_key="einsum"),
    )
)
