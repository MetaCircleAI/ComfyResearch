"""combined_model — ModelDef-channel thin definition.

程序化生成自 manifest。This node has no provider or defaults because its canvas
add path supplies template options. Code generation uses ``buildCombinedModelTorch``.
"""
from __future__ import annotations

from comfy_research.nodes.registry import model_def
from comfy_research.nodes.schema import FrontendSpec, ModelDef

DEF = model_def(
    ModelDef(
        type="combined_model",
        label='Combined model',
        hint="Saved combined subgraph from the library (same as Model list): expandable shell with inner layers.",
        family=('canvas_trainer_model_source',),
        fields=(),
        frontend=FrontendSpec(component_key="CombinedModelNode", codegen_key="combined_model"),
    )
)
