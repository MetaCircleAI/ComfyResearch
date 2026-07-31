"""flatten — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="flatten",
        label='Flatten',
        hint="Collapse tensor axes: optional dimension index to keep separate; empty/null flattens to a single axis (shape check).",
        family=("canvas_layer_strip_chain",),
        fields=(
            EnumField(key="ioMode", label='Io Mode', default='input-output', options=('model', 'input-output')),
            EnumField(key="levelMode", label='Level Mode', default='high', options=('high', 'low')),
        ),
        spawn_defaults=(('exceptDim', None), ('ioMode', 'input-output'), ('levelMode', 'high'),),
        frontend=FrontendSpec(component_key="FlattenNode", codegen_key="flatten"),
    )
)
