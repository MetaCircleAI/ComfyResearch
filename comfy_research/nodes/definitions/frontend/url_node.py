"""url_node — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import EnumField, FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="url_node",
        label="URL",
        category="language",
        hint="Store a link; click the resolved URL below to open it in a new tab (http/https only).",
        fields=(EnumField(key="url", label="Url", default=""),),
        defaults=(("url", ""),),
        frontend=FrontendSpec(component_key="UrlNode"),
    )
)
