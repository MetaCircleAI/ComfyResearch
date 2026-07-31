"""Cyclic LR / batch schedules (Jastrzębski Fig 1 left: discrete per-epoch; optional triangular per-step)."""

from __future__ import annotations

import math
from typing import Literal

CyclicScheduleMode = Literal["discrete_epoch", "square_epoch", "triangular_step"]


def triangular_cycle_fraction(pos: int, cycle_length: int) -> float:
    """Return 0→1→0 over one triangular cycle (integer positions)."""
    cl = max(1, int(cycle_length))
    p = int(pos) % cl
    half = cl / 2.0
    if p < half:
        return p / half if half > 0 else 1.0
    return 2.0 - (p / half) if half > 0 else 0.0


def square_cycle_fraction(pos: int, cycle_length: int) -> float:
    """Return 0 for the first half-cycle and 1 for the second."""
    cl = max(1, int(cycle_length))
    return 0.0 if int(pos) % cl < math.ceil(cl / 2) else 1.0


def steps_per_epoch(train_size: int, ref_batch: int) -> int:
    return max(1, math.ceil(int(train_size) / max(1, int(ref_batch))))


def epoch_index_for_step(step: int, *, steps_per_epoch: int) -> int:
    spe = max(1, int(steps_per_epoch))
    return max(0, int(step)) // spe


def cyclic_fraction_for_step(
    step: int,
    *,
    mode: CyclicScheduleMode,
    cycle_length_epochs: int,
    cycle_length_steps: int,
    steps_per_epoch: int,
) -> float:
    if mode in ("discrete_epoch", "square_epoch"):
        epoch = epoch_index_for_step(step, steps_per_epoch=steps_per_epoch)
        if mode == "square_epoch":
            return square_cycle_fraction(epoch, max(1, cycle_length_epochs))
        return triangular_cycle_fraction(epoch, max(1, cycle_length_epochs))
    return triangular_cycle_fraction(step, max(1, cycle_length_steps))


def cyclic_lr_for_step(
    step: int,
    *,
    lr_min: float,
    lr_max: float,
    mode: CyclicScheduleMode = "discrete_epoch",
    cycle_length_epochs: int = 10,
    cycle_length_steps: int = 3910,
    steps_per_epoch: int = 391,
) -> float:
    """CLR triangle is inverted vs CBS so η/S stays in phase (Fig 1 left).

    CBS: η fixed, B = lo + (hi-lo)*frac → η/S high at frac=0.
    CLR: B fixed, η = hi - (hi-lo)*frac → η/S high at frac=0.
    """
    lo = float(min(lr_min, lr_max))
    hi = float(max(lr_min, lr_max))
    frac = cyclic_fraction_for_step(
        step,
        mode=mode,
        cycle_length_epochs=cycle_length_epochs,
        cycle_length_steps=cycle_length_steps,
        steps_per_epoch=steps_per_epoch,
    )
    return hi - (hi - lo) * frac


def cyclic_batch_for_step(
    step: int,
    *,
    batch_min: int,
    batch_max: int,
    mode: CyclicScheduleMode = "discrete_epoch",
    cycle_length_epochs: int = 10,
    cycle_length_steps: int = 3910,
    steps_per_epoch: int = 391,
) -> int:
    lo = int(min(batch_min, batch_max))
    hi = int(max(batch_min, batch_max))
    frac = cyclic_fraction_for_step(
        step,
        mode=mode,
        cycle_length_epochs=cycle_length_epochs,
        cycle_length_steps=cycle_length_steps,
        steps_per_epoch=steps_per_epoch,
    )
    return max(1, int(round(lo + (hi - lo) * frac)))


def cyclic_batch_for_data_epoch(
    data_epoch: int,
    *,
    batch_min: int,
    batch_max: int,
    cycle_length_epochs: int = 10,
    mode: CyclicScheduleMode = "discrete_epoch",
) -> int:
    """Discrete cyclic batch at data-epoch boundaries (Jastrzębski CBS)."""
    lo = int(min(batch_min, batch_max))
    hi = int(max(batch_min, batch_max))
    frac = (
        square_cycle_fraction(int(data_epoch), max(1, int(cycle_length_epochs)))
        if mode == "square_epoch"
        else triangular_cycle_fraction(int(data_epoch), max(1, int(cycle_length_epochs)))
    )
    return max(1, int(round(lo + (hi - lo) * frac)))


def cyclic_lr_for_data_epoch(
    data_epoch: int,
    *,
    lr_min: float,
    lr_max: float,
    cycle_length_epochs: int = 10,
    mode: CyclicScheduleMode = "discrete_epoch",
) -> float:
    """Discrete cyclic LR at data-epoch boundaries (Jastrzębski CLR).

    Inverted vs CBS batch triangle so η/S matches (see cyclic_lr_for_step).
    """
    lo = float(min(lr_min, lr_max))
    hi = float(max(lr_min, lr_max))
    frac = (
        square_cycle_fraction(int(data_epoch), max(1, int(cycle_length_epochs)))
        if mode == "square_epoch"
        else triangular_cycle_fraction(int(data_epoch), max(1, int(cycle_length_epochs)))
    )
    return hi - (hi - lo) * frac


def data_epoch_state_for_cyclic_batch(
    global_step: int,
    *,
    train_size: int,
    batch_min: int,
    batch_max: int,
    cycle_length_epochs: int,
    mode: CyclicScheduleMode = "discrete_epoch",
) -> tuple[int, int, int]:
    """Simulate CBS data-epoch walk; return (data_epoch_index, steps_in_epoch, batch_size)."""
    if global_step <= 0:
        b0 = cyclic_batch_for_data_epoch(
            0, batch_min=batch_min, batch_max=batch_max, cycle_length_epochs=cycle_length_epochs, mode=mode
        )
        return 0, 0, b0
    ep = 0
    steps_in = 0
    remaining = int(global_step)
    batch = cyclic_batch_for_data_epoch(
        ep, batch_min=batch_min, batch_max=batch_max, cycle_length_epochs=cycle_length_epochs, mode=mode
    )
    while remaining > 0:
        epoch_len = steps_per_epoch(train_size, batch)
        if remaining >= epoch_len:
            remaining -= epoch_len
            ep += 1
            batch = cyclic_batch_for_data_epoch(
                ep, batch_min=batch_min, batch_max=batch_max, cycle_length_epochs=cycle_length_epochs, mode=mode
            )
            steps_in = 0
        else:
            steps_in = remaining
            remaining = 0
    return ep, steps_in, batch


def epochs_to_cycle_steps(epochs: int, train_size: int, ref_batch: int) -> int:
    return max(1, int(epochs) * steps_per_epoch(train_size, ref_batch))


def cbs_epochs_to_training_steps(
    epochs: int,
    *,
    train_size: int,
    batch_min: int,
    batch_max: int,
    cycle_length_epochs: int = 10,
    mode: CyclicScheduleMode = "discrete_epoch",
) -> int:
    """Total optimizer steps for N CBS data epochs (batch changes each epoch)."""
    total = 0
    for ep in range(max(0, int(epochs))):
        batch = cyclic_batch_for_data_epoch(
            ep,
            batch_min=batch_min,
            batch_max=batch_max,
            cycle_length_epochs=cycle_length_epochs,
            mode=mode,
        )
        total += steps_per_epoch(train_size, batch)
    return max(1, total)
