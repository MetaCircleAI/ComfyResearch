"""Single source of truth for the streaming per-step seed.

Imported by both trainer_run (minibatch permutation) and trainer_dataset_streaming
(data rematerialize) so the two seed paths can never silently diverge.
"""

from __future__ import annotations


def streaming_train_step_seed(base: int, step: int) -> int:
    """Mix ``base`` and ``step`` into a 64-bit seed (unsigned wrap). Pure ``int`` avoids NumPy uint64 multiply warnings."""
    m = 0xFFFFFFFFFFFFFFFF
    b = int(base) & m
    s = int(step) & m
    c = 0x9E3779B97F4A7C15
    return int((b + s * c) & m)
