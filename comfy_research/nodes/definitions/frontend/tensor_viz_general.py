"""tensor_viz_general — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, IntField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="tensor_viz_general",
        label="General tensor viz",
        category="visualization",
        hint="General-purpose tensor visualization.",
        family=("observable_user_tensor_viz_display", "observable_user_tensor_viz_anchor"),
        fields=(
            EnumField(key="plot1dStyle", label="Plot1d Style", default="line"),
            EnumField(key="plot1dLineSort", label="Plot1d Line Sort", default="original"),
            IntField(key="histBins", label="Hist Bins", default=20),
            EnumField(key="plot2dStyle", label="Plot2d Style", default="scatter"),
            IntField(key="plot2dScatterAxis", label="Plot2d Scatter Axis", default=1),
            IntField(key="plot2dScatterI1", label="Plot2d Scatter I1", default=0),
            IntField(key="plot2dScatterI2", label="Plot2d Scatter I2", default=1),
        ),
        defaults=(
            ("plot1dStyle", "line"),
            ("plot1dLineSort", "original"),
            ("histBins", 20),
            ("plot2dStyle", "scatter"),
            ("plot2dScatterAxis", 1),
            ("plot2dScatterI1", 0),
            ("plot2dScatterI2", 1),
        ),
        frontend=FrontendSpec(component_key="TensorVizGeneralNode"),
    )
)
