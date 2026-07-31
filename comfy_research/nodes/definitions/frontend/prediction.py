"""prediction — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="prediction",
        label="Prediction",
        category="analysis",
        fields=(
            EnumField(key="split", label="Split", default="both"),
        ),
        defaults=(
            ("split", "both"),
            ("trainPrediction", None),
            ("testPrediction", None),
            ("trainerTask", None),
            ("lastError", None),
        ),
        frontend=FrontendSpec(component_key="PredictionNode"),
    )
)
