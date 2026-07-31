"""comment — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import InPort, PortAccept, EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="comment",
        label="Comment",
        category="language",
        hint="Sticky note on the canvas.",
        family=("agent_text_context",),
        fields=(
            EnumField(key="text", label="Text", default=""),
            EnumField(key="url", label="Url", default=""),
        ),
        defaults=(("text", ""), ("url", "")),
        # cascade return-style 分支逐字转写。VIZ_COMMENT_SOURCE_TYPES 是
        # **两 family 并集**(display ∪ comment_source)——初版只声明其一被金标
        # 当场抓丢 3 true(comment 桶 14→11),accepts 的 some 语义天然表达并集。
        # mirror 块注释保留于 cascade。
        ports=(
            InPort(id="comment", accepts=(
                PortAccept(handles=("comment",), source_family="observable_user_tensor_viz_display"),
                PortAccept(handles=("comment",), source_family="canvas_comment_source"),
            )),
        ),
        frontend=FrontendSpec(component_key="CommentNode"),
    )
)
