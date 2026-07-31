"""dimension_permutator — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="dimension_permutator",
        label="Dimension permutator",
        category="analysis",
        hint="Reorder tensor dimensions by wiring in↔out dots (row-major); left and right sockets are both labeled tensor.",
        # capability correction:canvas_comment_source 摘除——
        # 组件只渲染 tensor_in/tensor_out,无 comment source handle;旧 cascade 的
        # source 块也实际拦死该路径(family 名不副实,ports 接管暴露)。
        defaults=(
            ("axes", []),
        ),
        frontend=FrontendSpec(component_key="DimensionPermutatorNode"),
    )
)
