"""kan_reg — NodeDef-channel definition + recorder.

category="loss" 与 manifest 保持一致；family 恒为 ["observable"]。
loss 侧求和
(_kan_reg_loss_term 在 loss_terms)与 observables handle 的 sweep 轴
(6 轴=字段集)；loss handle 上零轴(LOSS_SWEEP_ALLOWLIST 排除)。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

KAN_REG = observable_def(
    ObservableDef(
        type="kan_reg",
        label="KAN reg",
        hint="pykan get_reg() regularization (λ and inner lamb_* coeffs) added to MSE for kan_model only; connect to Trainer observables.",
        category="loss",
        viz=VizSpec(
            variant="kan_reg",
            title="KAN reg",
            info_markdown=(
                "**KAN reg** — adds pykan-style regularization terms (L1 on activations, entropy, "
                "coefficient sparsity, etc.) scaled by the λ fields. Logged as a scalar when used "
                "as a loss-side observable."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="scalar_series"),
        ),
        fields=(
            EnumField(key="regMetric", label="Reg Metric", default="edge_forward_spline_n",
                      options=("edge_forward_spline_n", "edge_forward_spline_u", "edge_forward_sum",
                               "edge_backward", "node_backward")),
            FloatField(key="lamb", label="Lamb", default=0.01),
            FloatField(key="lambL1", label="Lamb L1", default=1),
            FloatField(key="lambEntropy", label="Lamb Entropy", default=2),
            FloatField(key="lambCoef", label="Lamb Coef", default=0),
            FloatField(key="lambCoefDiff", label="Lamb Coef Diff", default=0),
        ),
        frontend=FrontendSpec(component_key="KanRegNode", codegen_key="kan_reg"),
    )
)


@recorder_for(KAN_REG)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_kan_reg."""
    import torch

    from comfy_research.engine.trainer.loss_terms import _kan_reg_loss_term
    from comfy_research.engine.trainer.model_helpers import (
        _prepare_x_for_atomic_sequential,
        _regression_batch_for_model,
    )

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    xr = rec._xr
    get_reg = getattr(model, "get_reg", None)
    if get_reg is None:
        observable_metric_histories[on.id].append(float("nan"))
        return
    was_training = model.training
    model.train()
    try:
        with torch.no_grad():
            model(_prepare_x_for_atomic_sequential(model, _regression_batch_for_model(model, xr)))
            observable_metric_histories[on.id].append(float(_kan_reg_loss_term(model, on).item()))
    finally:
        model.train(was_training)
