"""observable_hessian_eigenvalues — NodeDef-channel definition + recorder.

Custom component adapter with rich order labels. ``order_from_field`` passes
the source node's order to the spawned visualization.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

HESSIAN_EIGENVALUES = observable_def(
    ObservableDef(
        type="observable_hessian_eigenvalues",
        label="Hessian eigenvalues",
        hint="Logs ranked Hessian eigenvalues of the training loss (second derivatives wrt parameters).",
        viz=VizSpec(
            variant="hessian_eigenvalues",
            title="Hessian eigenvalues",
            info_markdown=(
                "**Hessian eigenvalues** — top-k eigenvalues of the loss Hessian (or surrogate) "
                "when the trainer can form them; limited to modest models."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="hessian_topk", top_k_from_field="topK", order_from_field="order"),
        ),
        fields=(
            IntField(key="topK", label="Top K", default=5),
            EnumField(key="order", label="Order", default="descending"),
        ),
        frontend=FrontendSpec(component_key="HessianEigenvaluesObservableNode"),
    )
)


@recorder_for(HESSIAN_EIGENVALUES)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_hessian_eigenvalues."""
    from typing import Literal

    from comfy_research.engine.trainer.observable_metrics import (
        HESSIAN_FORCE_MAX_PARAMS,
        HESSIAN_PARAM_LIMIT,
        _hessian_loss_eigenvalues,
    )
    from comfy_research.engine.trainer.scalar import _scalar_int, _scalar_str

    criterion = rec.criterion
    hessian_oversized_mode = rec.hessian_oversized_mode
    kan_regs = rec.kan_regs
    loss_scale = rec.loss_scale
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    trainer_task = rec.trainer_task
    weight_reg_loss_nodes = rec.weight_reg_loss_nodes
    xr = rec._xr
    yr = rec._yr
    od_h: dict[str, Any] = on.data or {}
    top_k = max(1, _scalar_int(od_h.get("topK"), 5))
    order = _scalar_str(od_h.get("order"), "descending").strip().lower()
    sort_order: Literal["descending", "ascending"] = (
        "ascending" if order == "ascending" else "descending"
    )
    if hessian_oversized_mode == "skip":
        eigvals = [float("nan")] * top_k
    else:
        h_max = HESSIAN_FORCE_MAX_PARAMS if hessian_oversized_mode == "force" else HESSIAN_PARAM_LIMIT
        try:
            eigvals = _hessian_loss_eigenvalues(
                model,
                criterion,
                xr,
                yr,
                loss_scale,
                trainer_task=trainer_task,
                top_k=top_k,
                order=sort_order,
                kan_regs=kan_regs,
                weight_reg_loss_nodes=weight_reg_loss_nodes,
                max_params=h_max,
            )
        except Exception:
            eigvals = [float("nan")] * top_k
    observable_metric_histories[on.id].append(eigvals[0] if eigvals else float("nan"))
    for i, ev in enumerate(eigvals):
        rank_key = f"{on.id}::{i}"
        observable_metric_histories.setdefault(rank_key, [])
        observable_metric_histories[rank_key].append(float(ev))
