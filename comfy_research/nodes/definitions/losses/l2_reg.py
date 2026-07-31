"""l2_reg — LossDef-channel thin definition。零 provider。"""
from __future__ import annotations

from comfy_research.nodes.registry import loss_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, LossDef

DEF = loss_def(
    LossDef(
        type="l2_reg",
        label="L2 Reg",
        hint="L2 weight regularization (loss scale × Σw²) added to the primary loss; wire to Trainer loss together with MSE or cross-entropy.",
        family=("trainer_weight_regularizer_loss", "trainer_loss_socket_aux"),
        fields=(FloatField(key="lossScale", label="Loss Scale", default=1),),
        frontend=FrontendSpec(component_key="L2RegNode"),
    )
)
