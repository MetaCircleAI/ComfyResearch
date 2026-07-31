"""observable_relu_nonlinear_count — NodeDef-channel definition + recorder.

Generic component rendering one IntField and hint from the generated spec;
scalar_series spawn.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

RELU_NONLINEAR_COUNT = observable_def(
    ObservableDef(
        type="observable_relu_nonlinear_count",
        label="ReLU nonlinear count",
        hint="Count post-ReLU units that are on for some batch samples and off for others (per hidden layer).",
        viz=VizSpec(
            variant="relu_nonlinear",
            title="ReLU nonlinear count",
            info_markdown=(
                "**ReLU nonlinear count** — counts hidden units that are on for some samples and "
                "off for others on the batch (a coarse “nonlinearity” signal)."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="scalar_series"),
        ),
        fields=(IntField(key="hiddenLayerIndex", label="Hidden Layer Index", default=1),),
        frontend=FrontendSpec(),
    )
)


@recorder_for(RELU_NONLINEAR_COUNT)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_relu_nonlinear_count."""
    import torch.nn as nn

    from comfy_research.engine.trainer.model_helpers import _count_nonlinear_relu_neurons
    from comfy_research.engine.trainer.scalar import _scalar_int

    depth = rec.depth
    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    xr = rec._xr
    od_r: dict[str, Any] = on.data or {}
    hidx = _scalar_int(od_r.get("hiddenLayerIndex"), 1)
    assert isinstance(model, nn.Sequential)
    v_r = float(
        _count_nonlinear_relu_neurons(
            model,
            xr,
            hidden_layer_index=hidx,
            depth=depth,
        )
    )
    observable_metric_histories[on.id].append(v_r)
