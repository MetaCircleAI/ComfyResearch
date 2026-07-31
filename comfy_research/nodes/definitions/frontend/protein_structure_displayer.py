"""protein_structure_displayer — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="protein_structure_displayer",
        label="Protein structure displayer",
        category="visualization",
        fields=(
            EnumField(key="coordsFlat", label="Coords Flat", default=""),
            EnumField(key="resolvedCoordsFlat", label="Resolved Coords Flat", default=""),
            BoolField(key="showPolyline", label="Show Polyline", default=True),
            IntField(key="sampleIndex", label="Sample Index", default=0),
        ),
        defaults=(
            ("coordsFlat", ""),
            ("resolvedCoordsFlat", ""),
            ("showPolyline", True),
            ("sampleIndex", 0),
        ),
        frontend=FrontendSpec(component_key="ProteinStructureDisplayerNode"),
    )
)
