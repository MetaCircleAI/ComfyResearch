"""observable_last_layer_weight_norm — NodeDef-channel definition + recorder.

Slingshot Effect repro(Thilak et al., TMLR 2024, Fig. 1):分类层(最后一个
``nn.Linear``)权重 L2 范数——弹弓循环的标志信号(范数阶梯爬升 → loss 尖峰
→ 平台）。严格 ``isinstance(m, nn.Linear)`` 判层，禁止按“含 weight”
泛判,防 Norm 层混入);纯读权重,零额外前向。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

LAST_LAYER_WEIGHT_NORM = observable_def(
    ObservableDef(
        type="observable_last_layer_weight_norm",
        label="Last layer weight norm",
        hint="L2 norm of the last nn.Linear weight (classification layer); the Slingshot Effect signature signal.",
        viz=VizSpec(
            variant="user",
            title="Last layer weight norm",
            info_markdown=(
                "**Last layer weight norm** — ‖W‖₂ of the final `nn.Linear` (classification layer). "
                "During the Terminal Phase of Training with Adam-family optimizers this norm shows "
                "cyclic rapid-growth → curtailment phases; each transition coincides with a training "
                "loss spike (the Slingshot Effect, Thilak et al. 2024). NaN if the model has no Linear layer."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="‖W‖₂"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(LAST_LAYER_WEIGHT_NORM)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """L2 norm of the last strict ``nn.Linear`` weight; NaN when no Linear exists."""
    import torch
    from torch import nn

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    last_linear: nn.Linear | None = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        observable_metric_histories[on.id].append(float("nan"))
        return
    with torch.no_grad():
        observable_metric_histories[on.id].append(
            float(torch.linalg.vector_norm(last_linear.weight.detach()).item())
        )
