"""Fourier-component observable definition and recorder."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import EnumField, FloatField, FrontendSpec, IntField, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


FOURIER_COMPONENT = observable_def(
    ObservableDef(
        type="observable_fourier_component",
        label="Fourier component",
        hint="Projects a regression target and prediction onto a sin/cos basis at one input frequency.",
        viz=VizSpec(
            variant="user",
            title="Fourier component",
            info_markdown=(
                "**Fourier component** projects the logging-batch target and prediction onto a sin/cos "
                "basis. `relative_projection_mse` is normalized component error; `amplitude_ratio` "
                "is prediction amplitude divided by target amplitude."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar"),
        ),
        fields=(
            FloatField(key="frequency", label="Frequency", default=1.0, min=0.0),
            EnumField(key="metric", label="Metric", default="relative_projection_mse", options=("relative_projection_mse", "amplitude_ratio")),
            IntField(key="inputAxis", label="Input axis", default=0, min=0),
            IntField(key="outputIndex", label="Output index", default=0, min=0),
        ),
        frontend=FrontendSpec(component_key="FourierComponentObservableNode"),
    )
)


@recorder_for(FOURIER_COMPONENT)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Record non-fatally: an invalid component must never interrupt training."""
    import torch

    from comfy_research.engine.trainer.model_helpers import _forward_reg
    from comfy_research.engine.trainer.observable_metrics import _fourier_component_observable_value, _observable_fourier_metric
    from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int

    histories = rec.observable_metric_histories
    if rec.trainer_task != "mse_regression":
        histories[on.id].append(float("nan"))
        return
    data: dict[str, Any] = on.data or {}
    try:
        with torch.no_grad():
            prediction = _forward_reg(rec.model, rec._xr)
            value = _fourier_component_observable_value(
                rec._xr, rec._yr, prediction,
                frequency=_scalar_float(data.get("frequency"), 1.0),
                metric=_observable_fourier_metric(data),
                input_axis=_scalar_int(data.get("inputAxis"), 0),
                output_index=_scalar_int(data.get("outputIndex"), 0),
            )
    except Exception:
        value = float("nan")
    histories[on.id].append(float(value))
