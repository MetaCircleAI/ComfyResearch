"""tensor_add — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="tensor_add",
        label='Tensor add',
        hint="Two tensors: element-wise sum with NumPy-style broadcasting.",
        family=("canvas_tensor_multi_input",),
        fields=(),
        spawn_defaults=(('outputTensor', None), ('lastError', None),),
        frontend=FrontendSpec(component_key="TensorAddNode"),
    )
)
