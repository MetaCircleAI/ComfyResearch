"""observable_weight_l1 — NodeDef-channel definition + recorder.

Generic component with a scalar_series spawn. The dedicated weight_l1
visualization defaults to ``{showSeries}`` rather than hessian_topk(1). The
manifest intentionally has no hint.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

WEIGHT_L1 = observable_def(
    ObservableDef(
        type="observable_weight_l1",
        label="Weight L1",
        hint="Log L1 norm of weights during training.",
        viz=VizSpec(
            variant="weight_l1",
            title="Weight L1",
            info_markdown=(
                "**Weight L1** — sum of absolute values of weights each log step. Complements L2 "
                "for sparsity / scale diagnostics."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="scalar_series"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(WEIGHT_L1)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_weight_l1."""
    from comfy_research.engine.trainer.tensor_norms import _weight_l1_norm

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    observable_metric_histories[on.id].append(_weight_l1_norm(model))
