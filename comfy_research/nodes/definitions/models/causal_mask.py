"""causal_mask — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="causal_mask",
        label='Causal mask',
        hint="Apply a causal (triangular) mask before softmax in attention-style subgraphs.",
        family=("canvas_layer_strip_chain",),
        fields=(
            IntField(key="diagonalOffset", label='Diagonal Offset', default=1),
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        frontend=FrontendSpec(component_key="CausalMaskNode", codegen_key="causal_mask"),
    )
)
