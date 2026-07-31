"""Power-law spectrum exponent of a selected representation covariance."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from comfy_research.nodes.definitions.observables.representation_rankme import _rankme_and_eigenvalues
from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import BoolField, EnumField, FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node


REPRESENTATION_ALPHA_REQ = observable_def(
    ObservableDef(
        type="observable_representation_alpha_req",
        label="Representation alphaReQ",
        hint="Power-law decay exponent of a selected representation covariance spectrum.",
        viz=VizSpec(
            variant="user",
            title="Representation alphaReQ",
            info_markdown=(
                "**Representation alphaReQ** - fit `log(lambda_i) = c - alpha log(i)` to the "
                "positive centered-covariance eigenvalues. Larger alpha means more concentrated variance."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="exponent"),
        ),
        fields=(
            EnumField(key="representationId", label="Representation ID", default="0::output"),
            BoolField(key="tokenPositionsAsSamples", label="Token Positions As Samples", default=False),
        ),
        frontend=FrontendSpec(codegen_key="observable_representation_alpha_req"),
    )
)


@recorder_for(REPRESENTATION_ALPHA_REQ)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    from comfy_research.engine.trainer.scalar import _scalar_bool, _scalar_str

    data: dict[str, Any] = on.data or {}
    representation_id = _scalar_str(data.get("representationId"), "0::output").strip()
    _, eigenvalues = _rankme_and_eigenvalues(
        rec._representation_tensors_for_log().get(representation_id),
        token_positions_as_samples=_scalar_bool(data.get("tokenPositionsAsSamples"), False),
    )
    positive = [float(x) for x in eigenvalues if math.isfinite(x) and x > 1e-12]
    if len(positive) < 3:
        rec.observable_metric_histories[on.id].append(float("nan"))
        return
    log_index = [math.log(index) for index in range(1, len(positive) + 1)]
    log_values = [math.log(value) for value in positive]
    mean_x = sum(log_index) / len(log_index)
    mean_y = sum(log_values) / len(log_values)
    denom = sum((x - mean_x) ** 2 for x in log_index)
    if denom <= 0:
        rec.observable_metric_histories[on.id].append(float("nan"))
        return
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_index, log_values)) / denom
    rec.observable_metric_histories[on.id].append(float(-slope))
