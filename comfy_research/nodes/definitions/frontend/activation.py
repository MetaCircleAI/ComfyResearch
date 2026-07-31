"""activation — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="activation",
        label="Activation",
        category="analysis",
        hint="Capture layer activations from the MLP for tensor analysis.",
        defaults=(
            ("representationOptions", []),
            ("selectedRepresentationIds", []),
            ("activationWirePicks", []),
            ("scanMessage", None),
            ("collectedActivations", None),
            ("collectSummary", None),
            ("activationRunId", None),
            ("activationManifest", None),
        ),
        frontend=FrontendSpec(component_key="ActivationNode", codegen_key="activation"),
    )
)
