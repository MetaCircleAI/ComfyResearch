"""observable_neuron_trajectory_2d — NodeDef-channel definition + recorder.

Generic component for the observable node itself; the dedicated viz node type
(``observable_viz_neuron_trajectory_2d``) is registered separately.
Records into the embedding-histories pathway (rec.observable_embedding_histories).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

NEURON_TRAJECTORY_2D = observable_def(
    ObservableDef(
        type="observable_neuron_trajectory_2d",
        label="Neuron trajectory 2D",
        hint="Records first hidden-layer weight rows wᵢ∈ℝ² at each log step (2D neuron positions for rich/lazy regime).",
        viz=VizSpec(
            variant="neuron_trajectory_2d",
            title="Neuron trajectory 2D",
            info_markdown=(
                "**Neuron trajectory 2D** — records first hidden-layer weight positions wᵢ∈ℝ² at each "
                "log step; visualize as scatter+trails to see feature learning (rich regime) vs frozen "
                "features (lazy regime), Chizat et al. 2019."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="neuron_trajectory"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(NEURON_TRAJECTORY_2D)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Return the first ``nn.Linear`` weight rows as neuron positions."""
    import torch.nn as nn

    model = rec.model
    observable_embedding_histories = rec.observable_embedding_histories
    first_linear: nn.Linear | None = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is not None:
        w_np = first_linear.weight.detach().cpu().numpy()
        observable_embedding_histories[on.id].append(w_np.tolist())
    else:
        observable_embedding_histories[on.id].append([])
