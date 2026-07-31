"""hypothesis — FrontendNodeDef-channel definition。
defaults = { ...defaultCommentData(), text: "Hypothesis: " } 的字面展开(键序保序)。"""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="hypothesis",
        label="Hypothesis",
        category="language",
        family=("agent_text_context",),
        fields=(
            EnumField(key="text", label="Text", default="Hypothesis: "),
            EnumField(key="url", label="Url", default=""),
        ),
        defaults=(("text", "Hypothesis: "), ("url", "")),
        # 共享组件复用:hypothesis 一直渲染为 CommentNode(HAND registry 实况)。
        frontend=FrontendSpec(component_key="CommentNode"),
    )
)
