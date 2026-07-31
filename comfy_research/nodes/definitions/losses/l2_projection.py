"""l2_projection — LossDef-channel definition without a provider.

targetNorm is not sweepable because this auxiliary loss exposes no sweep axes.
"""
from __future__ import annotations

from comfy_research.nodes.registry import loss_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, LossDef

DEF = loss_def(
    LossDef(
        type="l2_projection",
        label="L2 Projection",
        hint="After each optimizer step, project trainable weights onto an L2 norm shell; wire to Trainer loss socket (does not add to loss).",
        family=("trainer_loss_socket_aux",),
        fields=(FloatField(key="targetNorm", label="Target Norm", default=1, sweepable=False),),
        frontend=FrontendSpec(component_key="L2ProjectionNode"),
    )
)
