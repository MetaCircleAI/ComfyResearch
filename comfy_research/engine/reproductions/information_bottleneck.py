"""Batched-seed Information Bottleneck reproduction engine.

The original network is tiny, so launching fifty independent GPU jobs wastes
most of the device.  This module stores the parameters of all repeats along a
leading ensemble dimension.  Every repeat still owns independent weights and
an independent training subset, while a single optimizer step executes them as
one batched workload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.engine.analysis.snapshot_schedules import idnns_epoch_snapshots
from comfy_research.engine.datasets.information_bottleneck_dataset import VAR_U_PATH, load_var_u
from comfy_research.engine.datasets.sampling_orders import affine_epoch_positions
from comfy_research.engine.trainer.information_plane import binned_information_pair


ActivationName = Literal["tanh", "relu", "softsign", "softplus"]
OutputMode = Literal["binary_sigmoid", "two_softmax"]
OptimizerName = Literal["adam", "sgd"]
InitializerName = Literal["idnns_fan_in", "saxe_fan_out"]
ShuffleMode = Literal["fixed", "affine"]
SnapshotSchedule = Literal["idnns_logspace", "saxe_callback"]


@dataclass(frozen=True)
class InformationBottleneckProtocol:
    """Fully specified training/measurement protocol for one Figure 3 panel."""

    train_percent: int
    repeats: int = 50
    epochs: int = 10_000
    batch_size: int = 256
    learning_rate: float = 4e-4
    optimizer: OptimizerName = "adam"
    hidden_dims: tuple[int, ...] = (10, 8, 6, 4, 2)
    output_mode: OutputMode = "binary_sigmoid"
    activation: ActivationName = "tanh"
    initializer: InitializerName = "idnns_fan_in"
    shuffle_mode: ShuffleMode = "affine"
    bins: int = 30
    binning: str = "idnns_equal_points"
    snapshot_schedule: SnapshotSchedule = "idnns_logspace"
    snapshot_samples: int = 1800
    seed: int = 1703
    data_path: str = str(VAR_U_PATH)

    @property
    def train_size(self) -> int:
        return int(np.rint(4096 * (self.train_percent / 100.0)))

    @property
    def layer_count(self) -> int:
        return len(self.hidden_dims) + 1

    def validate(self) -> None:
        if self.train_percent < 1 or self.train_percent > 100:
            raise ValueError("train_percent must be in [1, 100]")
        if self.repeats < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("repeats, epochs, and batch_size must be positive")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if self.snapshot_samples < 1:
            raise ValueError("snapshot_samples must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["train_size"] = self.train_size
        payload["layer_count"] = self.layer_count
        return payload


@dataclass
class InformationBottleneckResult:
    epochs: np.ndarray
    train_loss: np.ndarray
    train_accuracy: np.ndarray
    information_x: np.ndarray
    information_y: np.ndarray

    def aggregate(self) -> dict[str, object]:
        def stats(values: np.ndarray) -> dict[str, object]:
            return {
                "mean": np.mean(values, axis=1).tolist(),
                "std": np.std(values, axis=1).tolist(),
                "sem": (np.std(values, axis=1) / math.sqrt(values.shape[1])).tolist(),
            }

        return {
            "epochs": self.epochs.astype(int).tolist(),
            "train_loss": stats(self.train_loss),
            "train_accuracy": stats(self.train_accuracy),
            "information_x": stats(self.information_x),
            "information_y": stats(self.information_y),
        }


def saxe_epoch_snapshots(epochs: int = 10_000) -> np.ndarray:
    """Epoch callback schedule published in IBnet_SaveActivations.ipynb."""
    values = [
        epoch
        for epoch in range(epochs)
        if epoch < 20
        or (epoch < 100 and epoch % 5 == 0)
        or (epoch < 2000 and epoch % 20 == 0)
        or epoch % 100 == 0
    ]
    return np.asarray(values, dtype=np.int64)


class BatchedSeedMlp(nn.Module):
    """Independent MLP repeats evaluated through a leading batch dimension."""

    def __init__(
        self,
        *,
        repeats: int,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_mode: OutputMode,
        activation: ActivationName,
        initializer: InitializerName,
        seed: int,
    ) -> None:
        super().__init__()
        self.repeats = int(repeats)
        self.output_mode = output_mode
        self.activation_name = activation
        output_dim = 1 if output_mode == "binary_sigmoid" else 2
        dims = (int(input_dim), *map(int, hidden_dims), output_dim)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        torch.manual_seed(int(seed))
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            weight = nn.Parameter(torch.empty(self.repeats, in_dim, out_dim))
            bias = nn.Parameter(torch.zeros(self.repeats, 1, out_dim))
            fan = in_dim if initializer == "idnns_fan_in" else out_dim
            std = 1.0 / math.sqrt(float(fan))
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            self.weights.append(weight)
            self.biases.append(bias)

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "tanh":
            return torch.tanh(value)
        if self.activation_name == "relu":
            return F.relu(value)
        if self.activation_name == "softsign":
            return F.softsign(value)
        if self.activation_name == "softplus":
            return F.softplus(value)
        raise ValueError(f"unknown activation: {self.activation_name}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(0).expand(self.repeats, -1, -1)
        if x.ndim != 3 or int(x.shape[0]) != self.repeats:
            raise ValueError("BatchedSeedMlp expects [N,D] or [repeats,N,D]")
        hidden: list[torch.Tensor] = []
        value = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = torch.bmm(value, weight) + bias
            if index < len(self.weights) - 1:
                value = self._activation(value)
                hidden.append(value)
        return value, hidden

    def output_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_mode == "binary_sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)


def _make_optimizer(
    model: nn.Module,
    protocol: InformationBottleneckProtocol,
) -> torch.optim.Optimizer:
    if protocol.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=protocol.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    if protocol.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=protocol.learning_rate)
    raise ValueError(f"unknown optimizer: {protocol.optimizer}")


def _loss_per_repeat(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_mode: OutputMode,
) -> torch.Tensor:
    if output_mode == "binary_sigmoid":
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            labels.to(dtype=logits.dtype),
            reduction="none",
        ).mean(dim=1)
    per_item = F.cross_entropy(
        logits.reshape(-1, 2),
        labels.reshape(-1).long(),
        reduction="none",
    ).reshape(logits.shape[0], logits.shape[1])
    return per_item.mean(dim=1)


def _accuracy_per_repeat(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_mode: OutputMode,
) -> torch.Tensor:
    if output_mode == "binary_sigmoid":
        predictions = (logits.squeeze(-1) >= 0).long()
    else:
        predictions = torch.argmax(logits, dim=-1)
    return (predictions == labels.long()).float().mean(dim=1)


def _sample_training_subsets(
    rng: np.random.Generator,
    *,
    repeats: int,
    total: int,
    train_size: int,
) -> np.ndarray:
    return np.stack(
        [rng.choice(total, size=train_size, replace=False) for _ in range(repeats)],
        axis=0,
    ).astype(np.int64)


def _affine_epoch_order(
    base_indices: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    repeats, count = map(int, base_indices.shape)
    positions = torch.as_tensor(
        affine_epoch_positions(rng, count=count, repeats=repeats),
        dtype=torch.long,
        device=base_indices.device,
    )
    return base_indices.gather(1, positions)


def _measure(
    model: BatchedSeedMlp,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    train_indices: torch.Tensor,
    protocol: InformationBottleneckProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        train_x = x_all[train_indices]
        train_y = y_all[train_indices]
        train_logits, _ = model(train_x)
        losses = _loss_per_repeat(train_logits, train_y, protocol.output_mode)
        accuracy = _accuracy_per_repeat(train_logits, train_y, protocol.output_mode)
        logits, hidden = model(x_all)
        layers = [*hidden, model.output_probabilities(logits)]

    labels_np = y_all.detach().cpu().numpy()
    ix = np.empty((protocol.repeats, len(layers)), dtype=np.float64)
    iy = np.empty_like(ix)
    for layer_index, layer in enumerate(layers):
        values = layer.detach().cpu().numpy()
        for repeat in range(protocol.repeats):
            ix[repeat, layer_index], iy[repeat, layer_index] = binned_information_pair(
                values[repeat],
                labels_np,
                bins=protocol.bins,
                strategy=protocol.binning,
            )
    return (
        losses.detach().cpu().numpy().astype(np.float64),
        accuracy.detach().cpu().numpy().astype(np.float64),
        ix,
        iy,
    )


ProgressCallback = Callable[[int, int, bool], None]
def train_information_bottleneck_ensemble(
    protocol: InformationBottleneckProtocol,
    *,
    device: str | torch.device = "cpu",
    progress: ProgressCallback | None = None,
) -> tuple[InformationBottleneckResult, BatchedSeedMlp, np.ndarray]:
    """Train one sample-percentage panel and return all per-seed MI trajectories."""
    protocol.validate()
    x_np, y_np = load_var_u(Path(protocol.data_path))
    rng = np.random.default_rng(protocol.seed)
    shuffle_rng = np.random.default_rng(protocol.seed + 1_000_003)
    train_indices_np = _sample_training_subsets(
        rng,
        repeats=protocol.repeats,
        total=x_np.shape[0],
        train_size=protocol.train_size,
    )

    target_device = torch.device(device)
    x_all = torch.from_numpy(x_np).to(device=target_device, dtype=torch.float32)
    y_all = torch.from_numpy(y_np).to(device=target_device, dtype=torch.long)
    train_indices = torch.from_numpy(train_indices_np).to(device=target_device)
    model = BatchedSeedMlp(
        repeats=protocol.repeats,
        input_dim=12,
        hidden_dims=protocol.hidden_dims,
        output_mode=protocol.output_mode,
        activation=protocol.activation,
        initializer=protocol.initializer,
        seed=protocol.seed,
    ).to(target_device)
    optimizer = _make_optimizer(model, protocol)
    if protocol.snapshot_schedule == "idnns_logspace":
        snapshots = idnns_epoch_snapshots(
            protocol.epochs,
            samples=protocol.snapshot_samples,
        )
    elif protocol.snapshot_schedule == "saxe_callback":
        snapshots = saxe_epoch_snapshots(protocol.epochs)
    else:
        raise ValueError(f"unknown snapshot schedule: {protocol.snapshot_schedule}")
    snapshot_set = set(map(int, snapshots.tolist()))
    recorded_epochs: list[int] = []
    losses: list[np.ndarray] = []
    accuracies: list[np.ndarray] = []
    information_x: list[np.ndarray] = []
    information_y: list[np.ndarray] = []

    for epoch in range(protocol.epochs):
        is_snapshot = epoch in snapshot_set
        if is_snapshot:
            loss, accuracy, ix, iy = _measure(
                model,
                x_all,
                y_all,
                train_indices,
                protocol,
            )
            recorded_epochs.append(epoch)
            losses.append(loss)
            accuracies.append(accuracy)
            information_x.append(ix)
            information_y.append(iy)

        order = train_indices
        if protocol.shuffle_mode == "affine" and protocol.train_size > 1:
            order = _affine_epoch_order(train_indices, shuffle_rng)
        for start in range(0, protocol.train_size, protocol.batch_size):
            batch_indices = order[:, start : start + protocol.batch_size]
            batch_x = x_all[batch_indices]
            batch_y = y_all[batch_indices]
            logits, _ = model(batch_x)
            per_repeat = _loss_per_repeat(logits, batch_y, protocol.output_mode)
            optimizer.zero_grad(set_to_none=True)
            # Parameter slices are disjoint across repeats; summing preserves the
            # same gradient each network would receive in an independent job.
            per_repeat.sum().backward()
            optimizer.step()

        if progress is not None:
            progress(epoch + 1, protocol.epochs, is_snapshot)

    result = InformationBottleneckResult(
        epochs=np.asarray(recorded_epochs, dtype=np.int64),
        train_loss=np.asarray(losses, dtype=np.float64),
        train_accuracy=np.asarray(accuracies, dtype=np.float64),
        information_x=np.asarray(information_x, dtype=np.float64),
        information_y=np.asarray(information_y, dtype=np.float64),
    )
    return result, model, train_indices_np
