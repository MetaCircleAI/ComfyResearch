"""Reusable Zhang et al. Figure 1(a) reproduction engine.

The public paper specifies the data transformations, Small Inception model,
SGD hyperparameters, and 0.95-per-epoch learning-rate decay.  It does not
specify initialization, mini-batch size, or the precise smoothing used for the
plotted ``average_loss`` curves.  Those implementation choices are explicit in
``RandomLabelsProtocol`` and in every persisted run protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from comfy_research.engine.datasets.sampling_orders import (
    seedsequence_epoch_permutation,
)
from comfy_research.engine.datasets.vision_datasets_runtime import (
    _apply_cifar10_randomization,
)
from comfy_research.engine.models.cifar_models import build_small_inception_cifar


CONDITION_ORDER = (
    "true_labels",
    "random_labels",
    "shuffled_pixels",
    "random_pixels",
    "gaussian",
)

CONDITION_LABELS = {
    "true_labels": "true labels",
    "random_labels": "random labels",
    "shuffled_pixels": "shuffled pixels",
    "random_pixels": "random pixels",
    "gaussian": "gaussian",
}

_CONDITION_TRANSFORMS = {
    "true_labels": ("none", 0.0),
    "random_labels": ("none", 1.0),
    "shuffled_pixels": ("shuffled_pixels", 0.0),
    "random_pixels": ("random_pixels", 0.0),
    "gaussian": ("gaussian", 0.0),
}


@dataclass(frozen=True)
class RandomLabelsProtocol:
    condition: str
    train_size: int = 50_000
    test_size: int = 10_000
    steps: int = 25_000
    batch_size: int = 128
    log_every: int = 100
    full_eval_every: int = 1_000
    eval_batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_decay_factor: float = 0.95
    lr_decay_epochs: int = 1
    dataset_seed: int = 1_703
    transform_seed: int = 1_704
    model_seed: int = 1_706
    minibatch_seed: int = 1_707
    preprocessing: str = "center_crop_28_per_image_whiten"
    precision: str = "float32"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if self.condition not in CONDITION_ORDER:
            raise ValueError(f"unknown random-label condition: {self.condition}")
        for name in (
            "train_size",
            "steps",
            "batch_size",
            "log_every",
            "full_eval_every",
            "eval_batch_size",
            "lr_decay_epochs",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.test_size < 0:
            raise ValueError("test_size must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        input_transform, label_corruption = _CONDITION_TRANSFORMS[self.condition]
        payload.update(
            {
                "input_transform": input_transform,
                "label_corruption": label_corruption,
                "condition_label": CONDITION_LABELS[self.condition],
                "model": "Small Inception (Appendix Figure 3; 1,649,402 trainable parameters)",
                "curve_statistic": (
                    "sample-weighted mean of online mini-batch cross-entropy over each log interval"
                ),
                "full_eval_statistic": "exact full-dataset mean cross-entropy and accuracy",
            }
        )
        return payload


@dataclass
class RandomLabelsResult:
    step_ticks: np.ndarray
    interval_loss: np.ndarray
    interval_accuracy: np.ndarray
    learning_rate: np.ndarray
    full_eval_steps: np.ndarray
    full_train_loss: np.ndarray
    full_train_accuracy: np.ndarray
    final_test_loss: float
    final_test_accuracy: float
    elapsed_seconds: float
    peak_cuda_memory_bytes: int

    def aggregate(self) -> dict[str, Any]:
        return {
            "final_interval_loss": float(self.interval_loss[-1]),
            "final_interval_accuracy": float(self.interval_accuracy[-1]),
            "minimum_interval_loss": float(np.min(self.interval_loss)),
            "final_full_train_loss": float(self.full_train_loss[-1]),
            "final_full_train_accuracy": float(self.full_train_accuracy[-1]),
            "final_test_loss": float(self.final_test_loss),
            "final_test_accuracy": float(self.final_test_accuracy),
            "elapsed_seconds": float(self.elapsed_seconds),
            "peak_cuda_memory_bytes": int(self.peak_cuda_memory_bytes),
        }


def condition_transform(condition: str) -> tuple[str, float]:
    try:
        return _CONDITION_TRANSFORMS[condition]
    except KeyError as exc:
        raise ValueError(f"unknown random-label condition: {condition}") from exc


def prepare_condition_arrays(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    protocol: RandomLabelsProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply the selected corruption once so it remains fixed across epochs."""
    input_transform, label_corruption = condition_transform(protocol.condition)
    return _apply_cifar10_randomization(
        *arrays,
        input_transform=input_transform,
        label_corruption=label_corruption,
        rng=np.random.default_rng(protocol.transform_seed),
        preprocessing=protocol.preprocessing,
    )


def numpy_sha256(array: np.ndarray) -> str:
    """Content fingerprint that also commits to dtype and shape."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def steps_per_epoch(train_size: int, batch_size: int) -> int:
    return max(1, math.ceil(int(train_size) / int(batch_size)))


def learning_rate_for_step(protocol: RandomLabelsProtocol, step: int) -> float:
    epoch = max(0, int(step)) // steps_per_epoch(protocol.train_size, protocol.batch_size)
    exponent = epoch // max(1, protocol.lr_decay_epochs)
    return float(protocol.learning_rate * protocol.lr_decay_factor**exponent)


def epoch_permutation(train_size: int, *, seed: int, epoch: int) -> np.ndarray:
    """Deterministic epoch shuffle used by a fresh run."""
    return seedsequence_epoch_permutation(train_size, seed=seed, epoch=epoch)


def _evaluate(
    model: nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    if int(x_cpu.shape[0]) == 0:
        return float("nan"), float("nan")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    try:
        with torch.inference_mode():
            for start in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[start : start + batch_size].to(device, non_blocking=True)
                yb = y_cpu[start : start + batch_size].to(device, non_blocking=True)
                logits = model(xb)
                total_loss += float(nn.functional.cross_entropy(logits, yb, reduction="sum").item())
                total_correct += int((logits.argmax(dim=1) == yb).sum().item())
                total_count += int(yb.shape[0])
    finally:
        model.train(was_training)
    return total_loss / total_count, total_correct / total_count


ProgressCallback = Callable[[int, int, dict[str, float]], None]
def train_random_label_condition(
    protocol: RandomLabelsProtocol,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None],
    *,
    device: torch.device,
    progress: ProgressCallback | None = None,
) -> tuple[RandomLabelsResult, nn.Module]:
    """Train one condition; callers serialize each condition before starting another."""
    x_train, y_train, x_test, y_test = arrays
    if x_train.shape[0] != protocol.train_size or y_train.shape[0] != protocol.train_size:
        raise ValueError("prepared train arrays do not match protocol.train_size")
    if protocol.test_size > 0:
        if x_test is None or y_test is None:
            raise ValueError("protocol requests a test set but prepared test arrays are missing")
        if x_test.shape[0] != protocol.test_size or y_test.shape[0] != protocol.test_size:
            raise ValueError("prepared test arrays do not match protocol.test_size")

    torch.manual_seed(protocol.model_seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(protocol.deterministic_algorithms)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(protocol.model_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = protocol.deterministic_algorithms
        torch.cuda.reset_peak_memory_stats(device)

    model = build_small_inception_cifar(in_channels=3, num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=protocol.learning_rate,
        momentum=protocol.momentum,
        weight_decay=protocol.weight_decay,
    )
    x_train_cpu = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32))
    y_train_cpu = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64))
    x_test_cpu = None if x_test is None else torch.from_numpy(np.ascontiguousarray(x_test, dtype=np.float32))
    y_test_cpu = None if y_test is None else torch.from_numpy(np.ascontiguousarray(y_test, dtype=np.int64))

    tick_history: list[int] = []
    loss_history: list[float] = []
    accuracy_history: list[float] = []
    lr_history: list[float] = []
    full_steps: list[int] = []
    full_losses: list[float] = []
    full_accuracies: list[float] = []

    initial_loss, initial_accuracy = _evaluate(
        model,
        x_train_cpu,
        y_train_cpu,
        device=device,
        batch_size=protocol.eval_batch_size,
    )
    full_steps.append(0)
    full_losses.append(initial_loss)
    full_accuracies.append(initial_accuracy)

    interval_loss_sum = 0.0
    interval_correct = 0
    interval_count = 0
    current_epoch = -1
    current_permutation = np.empty(0, dtype=np.int64)
    epoch_length = steps_per_epoch(protocol.train_size, protocol.batch_size)
    started = time.perf_counter()
    model.train()

    for step in range(protocol.steps):
        epoch = step // epoch_length
        if epoch != current_epoch:
            current_epoch = epoch
            current_permutation = epoch_permutation(
                protocol.train_size, seed=protocol.minibatch_seed, epoch=epoch
            )
        within_epoch = step % epoch_length
        start = within_epoch * protocol.batch_size
        stop = min(start + protocol.batch_size, protocol.train_size)
        indices = torch.from_numpy(current_permutation[start:stop])
        xb = x_train_cpu.index_select(0, indices).to(device, non_blocking=True)
        yb = y_train_cpu.index_select(0, indices).to(device, non_blocking=True)

        lr = learning_rate_for_step(protocol, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()

        count = int(yb.shape[0])
        interval_loss_sum += float(loss.detach().item()) * count
        interval_correct += int((logits.detach().argmax(dim=1) == yb).sum().item())
        interval_count += count
        done = step + 1

        if done % protocol.log_every == 0 or done == protocol.steps:
            mean_loss = interval_loss_sum / interval_count
            mean_accuracy = interval_correct / interval_count
            tick_history.append(done)
            loss_history.append(mean_loss)
            accuracy_history.append(mean_accuracy)
            lr_history.append(lr)
            interval_loss_sum = 0.0
            interval_correct = 0
            interval_count = 0
            if progress is not None:
                progress(done, protocol.steps, {"loss": mean_loss, "accuracy": mean_accuracy, "lr": lr})

        if done % protocol.full_eval_every == 0 or done == protocol.steps:
            train_loss, train_accuracy = _evaluate(
                model,
                x_train_cpu,
                y_train_cpu,
                device=device,
                batch_size=protocol.eval_batch_size,
            )
            full_steps.append(done)
            full_losses.append(train_loss)
            full_accuracies.append(train_accuracy)

    elapsed = time.perf_counter() - started
    if x_test_cpu is not None and y_test_cpu is not None:
        final_test_loss, final_test_accuracy = _evaluate(
            model,
            x_test_cpu,
            y_test_cpu,
            device=device,
            batch_size=protocol.eval_batch_size,
        )
    else:
        final_test_loss, final_test_accuracy = float("nan"), float("nan")
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    result = RandomLabelsResult(
        step_ticks=np.asarray(tick_history, dtype=np.int64),
        interval_loss=np.asarray(loss_history, dtype=np.float64),
        interval_accuracy=np.asarray(accuracy_history, dtype=np.float64),
        learning_rate=np.asarray(lr_history, dtype=np.float64),
        full_eval_steps=np.asarray(full_steps, dtype=np.int64),
        full_train_loss=np.asarray(full_losses, dtype=np.float64),
        full_train_accuracy=np.asarray(full_accuracies, dtype=np.float64),
        final_test_loss=float(final_test_loss),
        final_test_accuracy=float(final_test_accuracy),
        elapsed_seconds=float(elapsed),
        peak_cuda_memory_bytes=peak_memory,
    )
    torch.use_deterministic_algorithms(previous_deterministic)
    return result, model
