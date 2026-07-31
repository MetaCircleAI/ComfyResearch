"""curve_annotator — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="curve_annotator",
        label="Curve annotator",
        category="visualization",
        hint="Attach to Training or Observable viz; mirror plot and label x-ranges (plateau, spike, …).",
        defaults=(
            ("regions", []),
        ),
        # cascade return-style 分支逐字转写(三 viz 源 × annotator)。
        ports=(
            InPort(id="from_viz", accepts=(
                PortAccept(handles=("annotator",), source_type="training_visualization"),
                PortAccept(handles=("annotator",), source_type="observable_viz"),
                PortAccept(handles=("annotator",), source_type="observable_accuracy"),
            )),
        ),
        frontend=FrontendSpec(component_key="CurveAnnotatorNode"),
    )
)
