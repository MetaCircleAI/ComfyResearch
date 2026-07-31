"""observable_capacity — NodeDef-channel definition + recorder.

Generic component using the dedicated ``capacity`` visualization stream.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from comfy_research.nodes.registry import observable_def, recorder_for
from comfy_research.nodes.schema import FrontendSpec, ObservableDef, SpawnSpec, VizSpec

if TYPE_CHECKING:  # pragma: no cover
    from comfy_research.engine.trainer.recorder import ObservableRecorder
    from comfy_research.schemas.graph import Node

CAPACITY = observable_def(
    ObservableDef(
        type="observable_capacity",
        label="Capacity",
        hint="Memorization capacity in bits: (log(V) - loss) * N / log(2).",
        viz=VizSpec(
            variant="capacity",
            title="Capacity",
            info_markdown=(
                "**Capacity** — summary statistic related to effective model capacity on the "
                "logging batch (implementation-defined in the trainer)."
            ),
            spawns=True,
            spawn=SpawnSpec(kind="user_scalar", unit="value"),
        ),
        frontend=FrontendSpec(),
    )
)


@recorder_for(CAPACITY)
def record(rec: "ObservableRecorder", on: "Node") -> None:
    """Body moved verbatim from ObservableRecorder._record_capacity."""
    import math

    import torch

    observable_metric_histories = rec.observable_metric_histories
    xr = rec._xr
    yr = rec._yr
    vocab_size = 0
    if yr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        with torch.no_grad():
            vmax = int(torch.max(yr).item()) if yr.numel() > 0 else -1
        vocab_size = vmax + 1
    if vocab_size <= 1:
        observable_metric_histories[on.id].append(float("nan"))
    else:
        c_bits = (math.log(vocab_size) - float(rec._primary_train_val)) * float(xr.shape[0]) / math.log(2.0)
        observable_metric_histories[on.id].append(float(c_bits))
