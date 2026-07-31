"""curve_series_viz — FrontendNodeDef-channel definition.

曲线叠加 viz(SB/LB overlay,双轴 loss/accuracy)。defaults 键序 =
defaultCurveSeriesVizData 字面序;fields = scalar 子集(dualAxis/plotXMode/
plotXKey——repro manifest 实况;logScale 二键为 UI 运行态,不入 fields)。
ports:curves = curve_series_table 的 series(cascade 转写)。
"""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, FrontendNodeDef, FrontendSpec, InPort, PortAccept

DEF = frontend_node_def(
    FrontendNodeDef(
        type="curve_series_viz",
        label="Curve series viz",
        category="visualization",
        fields=(
            BoolField(key="dualAxis", label="Dual axis", default=True),
            BoolField(key="meanByRun", label="Mean by trainer run", default=False),
            EnumField(
                key="plotXMode",
                label="Plot X mode",
                default="progress",
                options=("progress", "step", "epoch", "param"),
            ),
            EnumField(key="plotXKey", label="Plot X key", default="step", options=("step", "epoch", "param")),
        ),
        defaults=(
            ("logScaleX", False),
            ("logScaleY", False),
            ("dualAxis", True),
            ("meanByRun", False),
            ("plotXMode", "progress"),
            ("plotXKey", "step"),
        ),
        ports=(
            InPort(id="curves", accepts=(
                PortAccept(handles=("series",), source_type="curve_series_table"),
            )),
        ),
        frontend=FrontendSpec(component_key="CurveSeriesVizNode"),
    )
)
