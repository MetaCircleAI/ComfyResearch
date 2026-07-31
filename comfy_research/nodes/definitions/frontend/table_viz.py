"""table_viz — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, BoolField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="table_viz",
        label="Table viz",
        category="analysis",
        hint="Line plot from a sweep data table (x-axis and legend here).",
        fields=(
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
        ),
        defaults=(
            ("plotXParamKey", None),
            ("logScaleX", False),
            ("logScaleY", False),
        ),
        # 试点:cascade return-style 分支逐字转写。
        ports=(
            InPort(id="table", accepts=(
                PortAccept(handles=("table",), source_type="sweep_data_table"),
            )),
        ),
        frontend=FrontendSpec(component_key="TableVizNode"),
    )
)
