"""l1_reg — LossDef-channel definition without a provider or code generator.

Weight regularization is evaluated by ``loss_terms``.
"""
from __future__ import annotations

from comfy_research.nodes.registry import loss_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, LossDef

DEF = loss_def(
    LossDef(
        type="l1_reg",
        label="L1 Reg",
        hint="L1 weight regularization (loss scale × Σ|w|) added to the primary loss; wire to Trainer loss together with MSE or cross-entropy.",
        family=("trainer_weight_regularizer_loss", "trainer_loss_socket_aux"),
        fields=(FloatField(key="lossScale", label="Loss Scale", default=1),),
        frontend=FrontendSpec(component_key="L1RegNode"),
    )
)
