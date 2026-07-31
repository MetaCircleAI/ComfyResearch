"""cross_entropy_loss — LossDef-channel definition + criterion provider.

lossMaskCustom is a free-text mask draft, so it is not sweepable.
"""
from __future__ import annotations

from comfy_research.nodes.registry import loss_criterion_for, loss_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, LossDef

DEF = loss_def(
    LossDef(
        type="cross_entropy_loss",
        label="Cross-entropy",
        hint="Cross-entropy for classification (e.g. token prediction).",
        family=("trainer_primary_loss", "trainer_loss_viz_spawn"),
        fields=(
            FloatField(key="lossScale", label="Loss Scale", default=1),
            FloatField(key="labelSmoothing", label="Label Smoothing", default=0),
            IntField(key="lossMaskContextLength", label="Loss Mask Context Length", default=1),
            EnumField(key="lossMaskMode", label="Loss Mask Mode", default="all", options=("all", "last_context", "custom")),
            EnumField(key="lossMaskCustom", label="Loss Mask Custom", default="", options=("",), sweepable=False),
        ),
        frontend=FrontendSpec(component_key="CrossEntropyLossNode", codegen_key="cross_entropy_loss"),
    )
)


@loss_criterion_for(DEF)
def build(loss_d, context):
    from comfy_research.engine.losses.loss_builders import _build_cross_entropy_loss_criterion

    return _build_cross_entropy_loss_criterion(loss_d, context)
