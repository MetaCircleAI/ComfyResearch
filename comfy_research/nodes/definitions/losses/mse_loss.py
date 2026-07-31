"""mse_loss — LossDef-channel definition + criterion provider."""
from __future__ import annotations

from comfy_research.nodes.registry import loss_criterion_for, loss_def
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, LossDef

DEF = loss_def(
    LossDef(
        type="mse_loss",
        label="MSE Loss",
        hint="Mean squared error loss for regression.",
        family=("trainer_primary_loss", "trainer_loss_viz_spawn"),
        fields=(
            FloatField(key="lossScale", label="Loss scale", default=1),
            IntField(key="lossMaskContextLength", label="Loss mask context length", default=1),
            EnumField(key="lossMaskMode", label="Loss mask mode", default="all", options=("all", "last_context", "custom"), manifest_options=True),
        ),
        frontend=FrontendSpec(component_key="MseLossNode", codegen_key="mse_loss"),
    )
)


@loss_criterion_for(DEF)
def build(loss_d, context):
    from comfy_research.engine.losses.loss_builders import _build_mse_loss_criterion

    return _build_mse_loss_criterion(loss_d, context)
