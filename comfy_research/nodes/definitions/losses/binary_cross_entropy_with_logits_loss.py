"""Binary classification loss for a single model logit."""

from __future__ import annotations

from comfy_research.nodes.registry import loss_criterion_for, loss_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, LossDef


DEF = loss_def(
    LossDef(
        type="binary_cross_entropy_with_logits_loss",
        label="Binary cross-entropy (logits)",
        hint=(
            "Binary cross-entropy for a single-logit classifier. Targets must be "
            "integer class labels 0/1; the runtime converts them to floating point."
        ),
        family=("trainer_primary_loss", "trainer_loss_viz_spawn"),
        fields=(FloatField(key="lossScale", label="Loss Scale", default=1),),
        frontend=FrontendSpec(component_key="BinaryCrossEntropyWithLogitsLossNode"),
    )
)


@loss_criterion_for(DEF)
def build(loss_d, context):
    from comfy_research.engine.losses.loss_builders import (
        _build_binary_cross_entropy_with_logits_loss_criterion,
    )

    return _build_binary_cross_entropy_with_logits_loss_criterion(loss_d, context)
