"""tensor_constant — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="tensor_constant",
        label='Tensor constant',
        hint="Configurable dense tensor (zeros, uniform [-1,1], or Gaussian); init seed for random fills; view values in parameters.",
        family=('canvas_tensor_source',),
        fields=(
            EnumField(key="init", label='Init', default='zero', options=('zero', 'uniform_m11', 'gaussian')),
            IntField(key="initSeed", label='Init Seed', default=0),
        ),
        spawn_defaults=(('shape', [2, 3]), ('init', 'zero'), ('initSeed', 0), ('outputTensor', None), ('lastError', None),),
        frontend=FrontendSpec(component_key="TensorConstantNode"),
    )
)
