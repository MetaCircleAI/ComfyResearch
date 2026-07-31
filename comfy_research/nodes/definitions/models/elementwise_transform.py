"""elementwise_transform — ModelDef-channel thin definition.

程序化生成自 manifest。spawn_defaults preserves runtime null values and shape
lists that fields cannot express.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import EnumField, FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="elementwise_transform",
        label='Elementwise transform',
        hint="Apply a scalar rule f(x) to each tensor element (LaTeX-style rule, default x^2).",
        family=("canvas_comment_source",),
        fields=(
            EnumField(key="ruleLatex", label='Rule Latex', default='x^2', options=('x^2',)),
        ),
        spawn_defaults=(('ruleLatex', 'x^2'), ('outputTensor', None), ('lastError', None),),
        frontend=FrontendSpec(component_key="ElementwiseTransformNode"),
    )
)
