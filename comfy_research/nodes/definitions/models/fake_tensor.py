"""fake_tensor — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="fake_tensor",
        label='Fake tensor (shape check)',
        hint="Declare a shape/dtype and run Check to annotate tensor sizes on wires or mark nodes where static shape rules disagree.",
        family=None,
        fields=(
            EnumField(key="dtype", label='Dtype', default='float', options=('long', 'float')),
        ),
        spawn_defaults=(('shape', [2, 3, 4]), ('dtype', 'float'), ('lastError', None),),
        frontend=FrontendSpec(component_key="FakeTensorNode"),
    )
)
