"""metric_compare — frontend-only comparison of two resolved curve visualizations."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec, InPort, PortAccept


DEF = frontend_node_def(
    FrontendNodeDef(
        type="metric_compare",
        label="Metric compare",
        category="visualization",
        fields=(
            EnumField(key="layout", label="Layout", default="horizontal", options=("horizontal", "vertical", "overlay")),
        ),
        defaults=(
            ("layout", "horizontal"),
            ("logScaleX", False),
            ("logScaleY", False),
        ),
        ports=(
            InPort(id="left", accepts=(
                PortAccept(handles=("compare",), source_type="curve_series_viz"),
                PortAccept(handles=("compare",), source_type="observable_viz"),
                PortAccept(handles=("compare",), source_type="tensor_viz_1d"),
            )),
            InPort(id="right", accepts=(
                PortAccept(handles=("compare",), source_type="curve_series_viz"),
                PortAccept(handles=("compare",), source_type="observable_viz"),
                PortAccept(handles=("compare",), source_type="tensor_viz_1d"),
            )),
        ),
        frontend=FrontendSpec(component_key="MetricCompareNode"),
    )
)
