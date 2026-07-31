"""curve_series_table — FrontendNodeDef-channel definition.

时间序列曲线累积表（train-complete/parametric sampler 推流）。
defaults 全 list/null 型 → 零 scalar fields(manifest 无 fields 键,repro 实况)。
ports 转写自 repro cascade:stream = training_visualization(stream|out_tensor_list)
∨ observable_viz(out_tensor) ∨ parametric_path_sampler(stream)。
"""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec, InPort, PortAccept

DEF = frontend_node_def(
    FrontendNodeDef(
        type="curve_series_table",
        label="Curve series table",
        category="analysis",
        family=("curve_series_sink",),
        defaults=(
            ("rows", []),
            ("selectedSeriesIds", None),
            ("captureMetrics", ["train_acc", "test_acc"]),
            ("paramKeyOrder", None),
        ),
        ports=(
            InPort(id="stream", accepts=(
                PortAccept(handles=("stream", "out_tensor_list"), source_type="training_visualization"),
                PortAccept(handles=("out_tensor",), source_type="observable_viz"),
                PortAccept(handles=("stream",), source_type="parametric_path_sampler"),
            )),
        ),
        frontend=FrontendSpec(component_key="CurveSeriesTableNode"),
    )
)
