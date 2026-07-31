"""reshape — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="reshape",
        label='Reshape',
        hint="Einops-style reshape rule string on the tensor chain (documentation / subgraph layout).",
        family=("canvas_layer_strip_chain",),
        fields=(
            EnumField(key="reshapeRule", label='Reshape Rule', default='b t d -> b t d', options=('b t d -> b t d',)),
            EnumField(key="shapeHint", label='Shape Hint', default='split heads', options=('split heads',)),
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        frontend=FrontendSpec(component_key="ReshapeNode", codegen_key="reshape"),
    )
)
