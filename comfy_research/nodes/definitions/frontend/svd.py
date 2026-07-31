"""svd — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="svd",
        label="SVD",
        category="analysis",
        hint="Full singular value decomposition (U, S, Vh); optional column mean removal.",
        fields=(
            EnumField(key="representationId", label="Representation Id", default=""),
            BoolField(key="removeMean", label="Remove Mean", default=False),
        ),
        defaults=(
            ("representationId", ""),
            ("removeMean", False),
            ("uTensor", None),
            ("sTensor", None),
            ("vTensor", None),
            ("svdSummary", None),
        ),
        frontend=FrontendSpec(component_key="SvdNode"),
    )
)
