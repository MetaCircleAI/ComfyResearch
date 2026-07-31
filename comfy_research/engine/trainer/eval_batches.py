"""Batched eval helpers shared by trainer loop and path tools."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn

from comfy_research.engine.models.diffusion_score_model import (
    DiffusionScoreMLP,
    diffusion_noise_mse_eval_mean,
)
from comfy_research.engine.trainer.loss_terms import _trainer_primary_loss_tensor
from comfy_research.engine.trainer.model_helpers import _forward_reg

_MAX_EVAL_BATCH_SIZE = 256


def _bounded_eval_batch_size(train_batch_size: int) -> int:
    if train_batch_size < 1:
        return _MAX_EVAL_BATCH_SIZE
    return min(train_batch_size, _MAX_EVAL_BATCH_SIZE)


@contextmanager
def _evaluation_mode(
    model: nn.Module,
    *,
    batch_norm_batch_stats: bool,
) -> Iterator[None]:
    training_states = [(module, module.training) for module in model.modules()]
    batch_norm_states: list[tuple[nn.modules.batchnorm._BatchNorm, bool]] = []
    model.eval()
    if batch_norm_batch_stats:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                batch_norm_states.append((module, module.track_running_stats))
                module.train()
                module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in batch_norm_states:
            module.track_running_stats = track_running_stats
        for module, training in training_states:
            module.training = training


def _batched_primary_loss_mean(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    trainer_task: str,
    criterion: nn.Module,
    loss_scale: float,
    batch_norm_batch_stats: bool = False,
) -> float:
    """Eval-mode mean primary loss over ``x``/``y`` in mini-batches (avoids GPU OOM on full CIFAR)."""
    bs = max(1, int(batch_size))
    n = int(x.shape[0])
    if n <= 0:
        return float("nan")
    total = 0.0
    with _evaluation_mode(
        model, batch_norm_batch_stats=batch_norm_batch_stats
    ), torch.no_grad():
        for i in range(0, n, bs):
            xb = x[i : i + bs]
            yb = y[i : i + bs]
            if trainer_task == "diffusion_noise":
                dm = cast(DiffusionScoreMLP, model)
                chunk = float(
                    diffusion_noise_mse_eval_mean(
                        dm, xb, timesteps=int(dm.max_timesteps), num_noise_draws=2
                    )
                    * loss_scale
                )
            else:
                pred = _forward_reg(model, xb)
                chunk = float(
                    _trainer_primary_loss_tensor(
                        pred,
                        yb,
                        trainer_task=trainer_task,
                        criterion=criterion,
                        loss_scale=loss_scale,
                    ).item()
                )
            total += chunk * int(xb.shape[0])
    return total / n


def _batched_classification_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    trainer_task: str,
    batch_norm_batch_stats: bool = False,
) -> float:
    bs = max(1, int(batch_size))
    n = int(x.shape[0])
    if n <= 0 or trainer_task == "diffusion_noise":
        return float("nan")
    correct = 0
    total = 0
    with _evaluation_mode(
        model, batch_norm_batch_stats=batch_norm_batch_stats
    ), torch.no_grad():
        for i in range(0, n, bs):
            xb = x[i : i + bs]
            yb = y[i : i + bs]
            pred = _forward_reg(model, xb)
            if (
                trainer_task == "token_classification"
                and pred.dim() == 3
                and yb.dim() == 2
            ):
                top1 = torch.argmax(pred, dim=-1)
                correct += int((top1 == yb.long()).sum().item())
                total += int(yb.numel())
            elif (
                trainer_task == "token_classification"
                and pred.dim() == 3
                and yb.dim() == 1
            ):
                top1 = torch.argmax(pred[:, -1, :], dim=-1)
                correct += int((top1 == yb.long()).sum().item())
                total += int(yb.shape[0])
            elif pred.dim() >= 2 and int(pred.shape[-1]) == 1 and yb.dim() == 1:
                top1 = (pred.reshape(-1) >= 0).long()
                correct += int((top1 == yb.long()).sum().item())
                total += int(yb.shape[0])
            elif pred.dim() >= 2 and yb.dim() == 1:
                top1 = torch.argmax(pred, dim=-1)
                correct += int((top1 == yb.long()).sum().item())
                total += int(yb.shape[0])
            else:
                return float("nan")
    return correct / total if total else float("nan")
