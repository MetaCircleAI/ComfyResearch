"""Information-plane observable with bounded CPU-side measurement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import (
    BoolField,
    EnumField,
    FrontendSpec,
    IntField,
    ObservableDef,
    SpawnSpec,
    VizSpec,
)

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


INFORMATION_PLANE = observable_def(
    ObservableDef(
        type="observable_information_plane",
        label="Information plane",
        hint=(
            "Binned I(X;T) / I(T;Y) trajectories. maxSamples bounds deterministic evaluation sampling and "
            "CPU transfer per log point; Trainer log frequency controls how often this work runs."
        ),
        viz=VizSpec(
            variant="information_plane",
            title="Information plane",
            info_markdown=(
                "**Information plane** bins bounded activations and plots each layer at `I(X;T)` versus "
                "`I(T;Y)`. It samples at most `maxSamples` evaluation rows and synchronizes those activations "
                "to CPU at each Trainer log point; increase the Trainer log interval for lower overhead."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="neuron_trajectory"),
        ),
        fields=(
            IntField(key="bins", label="Bins", default=30, min=2, max=256),
            IntField(key="maxSamples", label="Max samples", default=512, min=16, max=4096),
            BoolField(key="includeOutput", label="Include output", default=True),
            EnumField(
                key="binning",
                label="Binning",
                default="uniform_intervals",
                options=(
                    "uniform_intervals",
                    "idnns_equal_points",
                    "adaptive_minmax",
                    "saxe_fixed_width_0_07",
                ),
            ),
            EnumField(
                key="outputMapping",
                label="Output mapping",
                default="tanh",
                options=("tanh", "probability", "signed_probability"),
            ),
        ),
        frontend=FrontendSpec(component_key="InformationPlaneObservableNode"),
    )
)


@recorder_for(INFORMATION_PLANE)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Append an empty frame when measurement is unavailable; never stop training."""
    import torch

    from comfy_research.engine.trainer.information_plane import information_plane_for_model
    from comfy_research.engine.trainer.scalar import _scalar_bool, _scalar_int

    history = rec.observable_embedding_histories[on.id]
    x_eval = rec.x_test_t if rec.x_test_t is not None else rec._xr
    y_eval = rec.y_test_t if rec.y_test_t is not None else rec._yr
    if not isinstance(x_eval, torch.Tensor) or not isinstance(y_eval, torch.Tensor) or x_eval.ndim < 1 or y_eval.ndim < 1:
        history.append([])
        return
    data = on.data or {}
    try:
        count = min(int(x_eval.shape[0]), max(16, _scalar_int(data.get("maxSamples"), 512)))
        if count <= 0 or int(y_eval.shape[0]) != int(x_eval.shape[0]):
            history.append([])
            return
        if count < int(x_eval.shape[0]):
            indices = torch.linspace(0, int(x_eval.shape[0]) - 1, count, device=x_eval.device).long()
            x_eval, y_eval = x_eval.index_select(0, indices), y_eval.index_select(0, indices)
        history.append(information_plane_for_model(
            rec.model, x_eval, y_eval,
            bins=max(2, _scalar_int(data.get("bins"), 30)),
            include_output=_scalar_bool(data.get("includeOutput"), True),
            binning=str(data.get("binning") or "uniform_intervals"),
            output_mapping=str(data.get("outputMapping") or "tanh"),
        ))
    except Exception:
        history.append([])
