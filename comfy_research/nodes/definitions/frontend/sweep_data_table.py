"""sweep_data_table — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="sweep_data_table",
        label="Sweep data table",
        category="analysis",
        hint="Accumulate sweep runs in a table from a 0D tensor viz stream.",
        family=("canvas_comment_source",),
        defaults=(
            ("rows", []),
            ("selectedRowIds", None),
            ("paramKeyOrder", None),
        ),
        # 试点:cascade return-style 分支逐字转写。
        ports=(
            InPort(id="stream", accepts=(
                PortAccept(handles=("out_tensor",), source_type="tensor_viz_0d"),
            )),
        ),
        frontend=FrontendSpec(component_key="SweepDataTableNode"),
    )
)
