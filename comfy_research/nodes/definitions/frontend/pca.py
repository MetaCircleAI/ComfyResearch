"""pca — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="pca",
        label="PCA",
        category="analysis",
        hint="Principal component analysis on tensor batches.",
        family=("observable_user_tensor_transform",),
        fields=(
            EnumField(key="representationId", label="Representation Id", default=""),
            IntField(key="nComponents", label="N Components", default=2),
        ),
        defaults=(
            ("representationId", ""),
            ("nComponents", 2),
            ("transformedTensor", None),
            ("principalComponents", None),
            ("explainedVarianceRatio", None),
            ("pcaSummary", None),
        ),
        frontend=FrontendSpec(component_key="PcaNode"),
    )
)
