"""diffusion_mse_loss — LossDef-channel definition + criterion provider."""
from __future__ import annotations

from comfy_research.nodes.registry import loss_criterion_for, loss_def
from comfy_research.nodes.schema import FloatField, FrontendSpec, LossDef

DEF = loss_def(
    LossDef(
        type="diffusion_mse_loss",
        label="Diffusion MSE loss",
        hint="Loss socket for diffusion_score_model training (noise MSE inside trainer loop).",
        family=("trainer_primary_loss",),
        fields=(FloatField(key="lossScale", label="Loss Scale", default=1),),
        frontend=FrontendSpec(component_key="DiffusionMseLossNode", codegen_key="diffusion_mse_loss"),
    )
)


@loss_criterion_for(DEF)
def build(loss_d, context):
    from comfy_research.engine.losses.loss_builders import _build_diffusion_mse_loss_criterion

    return _build_diffusion_mse_loss_criterion(loss_d, context)
