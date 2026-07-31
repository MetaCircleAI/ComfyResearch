"""graph_assist_failure_overlay — FrontendNodeDef-channel definition。
manifest 实况:resizable=false、hasDefaults=false(defaults=None 三态)、internal 类;
不在 add-node 面板(panel catalog 独立),无 spawn 分支。"""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="graph_assist_failure_overlay",
        label="Graph assist failure overlay",
        category="internal",
        defaults=None,
        resizable=False,
        frontend=FrontendSpec(component_key="GraphAssistFailureOverlayNode"),
    )
)
