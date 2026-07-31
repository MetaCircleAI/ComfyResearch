"""Cyclic schedule node-data helpers (CBS/CLR)."""

from typing import Any

from comfy_research.engine.optimizers.cyclic_schedules import epochs_to_cycle_steps
from comfy_research.engine.trainer.scalar import _scalar_int, _scalar_str


def _cyclic_cycle_steps_from_node_data(
    data: dict[str, Any],
    *,
    train_size: int,
    default_ref_batch: int,
) -> int:
    explicit = _scalar_int(data.get("cycleLengthSteps"), 0)
    if explicit > 0:
        return explicit
    epochs = max(1, _scalar_int(data.get("cycleLengthEpochs"), 10))
    ref_batch = max(1, _scalar_int(data.get("refBatchSize"), default_ref_batch))
    return epochs_to_cycle_steps(epochs, train_size, ref_batch)


def _cyclic_schedule_mode_from_data(data: dict[str, Any]) -> str:
    raw = _scalar_str(data.get("scheduleMode"), "discrete_epoch").strip().lower().replace("-", "_")
    if raw in ("square_epoch", "square"):
        return "square_epoch"
    if raw in ("triangular_step", "triangular", "step"):
        return "triangular_step"
    return "discrete_epoch"


def _cyclic_cycle_epochs_from_node_data(data: dict[str, Any]) -> int:
    return max(1, _scalar_int(data.get("cycleLengthEpochs"), 10))
