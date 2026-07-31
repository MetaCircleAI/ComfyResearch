"""tensor_concat — ModelDef-channel thin definition.

程序化生成自 manifest。This tensor/canvas helper has no provider or sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import IntField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="tensor_concat",
        label='Tensor concat',
        hint="Concatenate multiple tensors into one output tensor (configurable number of inputs).",
        family=("canvas_tensor_multi_input",),
        fields=(
            IntField(key="inputCount", label='Input Count', default=2),
            IntField(key="concatDimension", label='Concat Dimension', default=0),
        ),
        frontend=FrontendSpec(component_key="TensorConcatNode"),
    )
)
