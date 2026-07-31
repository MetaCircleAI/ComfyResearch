"""protein_structure_comparison_viz — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="protein_structure_comparison_viz",
        label="Protein structure comparison",
        category="visualization",
        fields=(
            EnumField(key="predCoordsFlat", label="Pred Coords Flat", default=""),
            EnumField(key="trueCoordsFlat", label="True Coords Flat", default=""),
            IntField(key="sampleIndex", label="Sample Index", default=0),
        ),
        defaults=(
            ("predCoordsFlat", ""),
            ("trueCoordsFlat", ""),
            ("sampleIndex", 0),
        ),
        frontend=FrontendSpec(component_key="ProteinStructureComparisonVizNode"),
    )
)
