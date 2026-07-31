"""tensor_linspace — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="tensor_linspace",
        label='Tensor linspace',
        hint="Creates a 1D tensor with evenly spaced points between start and end, either linearly or in log10 space.",
        family=('canvas_tensor_source',),
        fields=(
            IntField(key="start", label='Start', default=0),
            IntField(key="end", label='End', default=1),
            IntField(key="numPoints", label='Num Points', default=8),
            EnumField(key="space", label='Space', default='linear', options=('linear', 'log10')),
        ),
        spawn_defaults=(('start', 0), ('end', 1), ('numPoints', 8), ('space', 'linear'), ('outputTensor', None), ('lastError', None),),
        frontend=FrontendSpec(component_key="TensorLinspaceNode"),
    )
)
