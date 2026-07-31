"""observable_viz — FrontendNodeDef-channel definition."""
from __future__ import annotations

from comfy_research.nodes.registry import frontend_node_def
from comfy_research.nodes.schema import BoolField, EnumField, FrontendNodeDef, FrontendObservableBlock, FrontendSpec

DEF = frontend_node_def(
    FrontendNodeDef(
        type="observable_viz",
        label="Observable viz",
        category="visualization",
        hint="Mirror for an observable wired to the trainer (auto-spawns when you connect observable → trainer).",
        family=("canvas_comment_source",),
        fields=(
            EnumField(key="observableName", label="Observable Name", default="User observable"),
            BoolField(key="logScaleX", label="Log Scale X", default=False),
            BoolField(key="logScaleY", label="Log Scale Y", default=False),
            BoolField(key="showSeries", label="Show Series", default=True),
            BoolField(key="showTrainCurve", label="Show Train Curve", default=True),
            BoolField(key="showTestCurve", label="Show Test Curve", default=True),
            EnumField(key="vizVariant", label="Viz Variant", default="user"),
        ),
        # pairedObservableId/pairedTrainerId default to undefined and therefore
        # cannot be represented in JSON; consumers treat missing keys identically.
        defaults=(
            ("observableName", "User observable"),
            ("logScaleX", False),
            ("logScaleY", False),
            ("showSeries", True),
            ("showTrainCurve", True),
            ("showTestCurve", True),
            ("vizVariant", "user"),
        ),
        observable=FrontendObservableBlock(
            info_markdown="**Observable viz** — mirror node that plots streamed histories from a paired trainer observable. Wire Trainer → observable source, pair this viz to that source, then Train.",
        ),
        frontend=FrontendSpec(component_key="ObservableVizNode"),
    )
)
