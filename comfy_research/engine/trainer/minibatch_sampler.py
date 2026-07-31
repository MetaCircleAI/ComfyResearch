"""Stateful minibatch policies used by the generic Trainer."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from comfy_research.engine.datasets.sampling_orders import affine_epoch_positions
from comfy_research.engine.trainer.dataset_helpers import (
    _take_epoch_shuffled_minibatch,
    _take_train_minibatch,
)

MinibatchSampling = Literal["independent_step", "epoch_shuffle", "affine_epoch"]


class TrainerMinibatchSampler:
    """Select batches while keeping paper-specific order logic out of the loop."""

    def __init__(
        self,
        *,
        mode: MinibatchSampling,
        train_size: int,
        steps_per_epoch: int,
        seed: int,
    ) -> None:
        self.mode = mode
        self.train_size = int(train_size)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.seed = int(seed)
        self._affine_rng = (
            np.random.default_rng(self.seed + 1_000_003)
            if mode == "affine_epoch"
            else None
        )
        self._affine_epoch = -1
        self._affine_positions: np.ndarray | None = None

    def _positions_for_affine_epoch(self, epoch: int) -> np.ndarray:
        if self._affine_rng is None:
            raise RuntimeError("affine sampler is not initialized")
        if epoch < self._affine_epoch:
            raise RuntimeError("affine epoch sampler moved backwards")
        while self._affine_epoch < epoch:
            self._affine_positions = affine_epoch_positions(
                self._affine_rng,
                count=self.train_size,
            )[0]
            self._affine_epoch += 1
        if self._affine_positions is None:
            raise RuntimeError("affine epoch sampler has no positions")
        return self._affine_positions

    def take(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        step: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 0 or batch_size >= int(x.shape[0]):
            return x, y
        if int(x.shape[0]) != self.train_size:
            raise RuntimeError(
                "fixed-dataset minibatch sampler received a different train size"
            )
        if self.mode == "independent_step":
            return _take_train_minibatch(
                x,
                y,
                batch_size,
                step=step,
                perm_seed=self.seed,
            )

        epoch = int(step) // self.steps_per_epoch
        step_in_epoch = int(step) % self.steps_per_epoch
        if self.mode == "epoch_shuffle":
            return _take_epoch_shuffled_minibatch(
                x,
                y,
                batch_size,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                run_seed=self.seed,
            )

        positions = self._positions_for_affine_epoch(epoch)
        start = step_in_epoch * int(batch_size)
        stop = min(start + int(batch_size), self.train_size)
        indices = torch.as_tensor(
            positions[start:stop],
            dtype=torch.long,
            device=x.device,
        )
        return x.index_select(0, indices), y.index_select(0, indices)
