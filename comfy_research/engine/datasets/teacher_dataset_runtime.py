"""Runtime helpers for ``teacher_dataset`` nodes: ``y = f_teacher(x)`` with ``x`` from a wired ``random_input_distribution``.

The graph passes the distribution node via the ``input_distribution`` handle; the trainer resolves
that node's JSON and builds a NumPy ``Generator`` (see ``random_input_distribution_runtime``).
This module implements draw-then-label in one place for training and activation collection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from comfy_research.engine.datasets.random_input_distribution_runtime import (
    rng_from_random_input_distribution_data,
    sample_x_from_random_input_dict,
)


def teacher_labels_numpy(teacher: nn.Module, x_np: np.ndarray) -> np.ndarray:
    """Run the teacher MLP on ``x_np`` (float32) and return ``y`` as float32 numpy."""
    with torch.no_grad():
        try:
            dev = next(teacher.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        x_t = torch.from_numpy(x_np).to(dev)
        return teacher(x_t).detach().cpu().numpy().astype(np.float32)


def generate_teacher_supervised_xy_numpy(
    rid_train_data: dict[str, Any],
    teacher: nn.Module,
    train_size: int,
    test_size: int,
    rid_test_data: dict[str, Any],
    *,
    test_rid_same_node_as_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Draw ``x`` from random-input specs, ``y = teacher(x)``.

    When train and test use the same physical RID node, the same ``Generator`` is reused for both
    splits so the stream matches prior ``trainer_run`` behavior.
    """
    rng_tr = rng_from_random_input_distribution_data(rid_train_data)
    x_np = sample_x_from_random_input_dict(rid_train_data, train_size, rng_tr)
    y_np = teacher_labels_numpy(teacher, x_np)
    if test_size <= 0:
        return x_np, y_np, None, None
    if test_rid_same_node_as_train:
        rng_te = rng_tr
    else:
        rng_te = rng_from_random_input_distribution_data(rid_test_data)
    x_test_np = sample_x_from_random_input_dict(rid_test_data, test_size, rng_te)
    y_test_np = teacher_labels_numpy(teacher, x_test_np)
    return x_np, y_np, x_test_np, y_test_np
