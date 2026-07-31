"""energy_readout_layer — ModelDef-channel thin definition.

程序化生成自 manifest。This ai4science atomic layer shares LinearLayerNode
and does not define a codegen key.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="energy_readout_layer",
        label='Energy readout layer',
        family=None,
        fields=(
            IntField(key="inFeatures", label='In Features', default=48, min=1),
            IntField(key="outFeatures", label='Out Features', default=4, min=1),
            IntField(key="bias", label='Bias', default=1, min=0),
            IntField(key="seed", label='Seed', default=7, min=0),
        ),
        frontend=FrontendSpec(component_key="LinearLayerNode"),
    )
)
