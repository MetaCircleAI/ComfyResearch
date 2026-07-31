"""observable_weight_displacement — NodeDef-channel definition + recorder.

Generic component; user-variant scalar path (user_whitelisted derives the
observable_viz user whitelist entry). Init snapshot state stays on the
ObservableRecorder (rec.weight_displacement_init).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

WEIGHT_DISPLACEMENT = observable_def(
    ObservableDef(
        type="observable_weight_displacement",
        label="Weight displacement (‖Δw‖/‖w₀‖)",
        hint="Fractional parameter displacement ‖w(t)−w(0)‖_F / ‖w(0)‖_F from the first log snapshot.",
        viz=VizSpec(
            variant="user",
            title="Weight displacement",
            info_markdown=(
                "**Weight displacement** — ‖w(t)−w(0)‖_F / ‖w(0)‖_F over all parameters from the first "
                "log snapshot. Small in the lazy/NTK regime, large under feature learning."
            ),
            spawns=True,
            user_whitelisted=True,
            spawn=SpawnSpec(kind="user_scalar", unit="‖Δw‖/‖w₀‖"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(WEIGHT_DISPLACEMENT)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Return fractional displacement ‖w(t)−w(0)‖_F / ‖w(0)‖_F."""
    import math

    model = rec.model
    observable_metric_histories = rec.observable_metric_histories
    weight_displacement_init = rec.weight_displacement_init
    if not weight_displacement_init:
        weight_displacement_init.update(
            {n: p.detach().clone() for n, p in model.named_parameters()}
        )
    total_sq = 0.0
    init_sq = 0.0
    for pname, param in model.named_parameters():
        if pname in weight_displacement_init:
            diff = param.detach() - weight_displacement_init[pname]
            total_sq += float(diff.pow(2).sum().item())
            init_sq += float(weight_displacement_init[pname].pow(2).sum().item())
    disp = math.sqrt(total_sq) / (math.sqrt(init_sq) + 1e-12)
    observable_metric_histories[on.id].append(disp)
