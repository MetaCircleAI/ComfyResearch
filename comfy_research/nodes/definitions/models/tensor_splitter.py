"""tensor_splitter — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="tensor_splitter",
        label='Tensor splitter',
        family=("canvas_layer_strip_chain",),
        fields=(
            IntField(key="splitDimension", label='Split Dimension', default=-1),
            IntField(key="numParts", label='Num Parts', default=3),
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        frontend=FrontendSpec(component_key="TensorSplitterNode"),
    )
)
