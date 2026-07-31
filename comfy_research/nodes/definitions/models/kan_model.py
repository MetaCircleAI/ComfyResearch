"""kan_model — ModelDef-channel definition + builder provider.

程序化生成自 manifest。numeric_hyena exposes no sweep axes; the other four
models derive axes from their fields. ``prepare_build_vector`` retains its
specialized ``ModelBuildContext`` construction.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_builder_for, model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ModelDef

DEF = model_def(
    ModelDef(
        type="kan_model",
        label="KAN (pykan)",
        hint="Kolmogorov–Arnold network (pykan KAN) for regression with the MSE trainer.",
        family=("vector_model", "activation_model", "canvas_trainer_model_source", "canvas_full_model",),
        fields=(
            IntField(key="inputDim", label="Input Dim", default=10, min=1),
            IntField(key="outputDim", label="Output Dim", default=1, min=1),
            IntField(key="depth", label="Depth", default=2, min=1),
            IntField(key="width", label="Width", default=5, min=1),
            IntField(key="grid", label="Grid", default=3, min=1),
            IntField(key="k", label="K", default=3, min=1),
            EnumField(key="baseFun", label="Base Fun", default='silu', options=('silu', 'identity', 'zero')),
            IntField(key="seed", label="Seed", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="KanModelNode", codegen_key="kan_model"),
    )
)


@model_builder_for(DEF)
def build(data, context):
    from comfy_research.engine.models.model_builders import _build_kan_model

    return _build_kan_model(data, context)
