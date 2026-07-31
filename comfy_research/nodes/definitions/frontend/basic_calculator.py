"""basic_calculator — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="basic_calculator",
        label="Basic calculator",
        category="analysis",
        hint="Combine N scalar tensors with a LaTeX formula (x_1, x_2, …) into one scalar output tensor.",
        family=("canvas_tensor_multi_input",),
        fields=(
            IntField(key="inputCount", label="Input Count", default=2),
            EnumField(key="equationLatex", label="Equation Latex", default="x_1 + x_2"),
        ),
        defaults=(
            ("inputCount", 2),
            ("equationLatex", "x_1 + x_2"),
            ("outputTensor", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="BasicCalculatorNode"),
    )
)
