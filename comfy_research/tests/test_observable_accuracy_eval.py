from __future__ import annotations

from types import SimpleNamespace

import torch

from comfy_research.nodes.definitions.observables.accuracy import record
from comfy_research.schemas.graph import Node, NodeKind


class _TrainingFlagClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.training_flags: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.training_flags.append(self.training)
        return torch.stack([1.0 - x[:, 0], x[:, 0]], dim=1)


def _rec(model: torch.nn.Module, *, override: float | None = None) -> SimpleNamespace:
    histories = {"acc": [], "acc::test": []}
    return SimpleNamespace(
        model=model,
        observable_metric_histories=histories,
        trainer_task="cross_entropy_dense",
        eval_batch_size=256,
        _x_test_log=torch.tensor([[0.0], [1.0]]),
        _y_test_log=torch.tensor([0, 1]),
        _xr=torch.tensor([[0.0], [1.0]]),
        _yr=torch.tensor([0, 1]),
        _metric_overrides={} if override is None else {"train_accuracy": override},
    )


def test_accuracy_observable_evaluates_in_eval_mode_and_restores_training() -> None:
    model = _TrainingFlagClassifier()
    model.train()
    rec = _rec(model)

    record(rec, Node(id="acc", type=NodeKind.observable_accuracy, data={}))

    assert model.training_flags == [False, False]
    assert model.training is True
    assert rec.observable_metric_histories["acc"] == [1.0]
    assert rec.observable_metric_histories["acc::test"] == [1.0]


def test_accuracy_observable_accepts_epoch_train_accuracy_override() -> None:
    model = _TrainingFlagClassifier()
    model.train()
    rec = _rec(model, override=0.75)

    record(rec, Node(id="acc", type=NodeKind.observable_accuracy, data={}))

    assert rec.observable_metric_histories["acc"] == [0.75]
    assert rec.observable_metric_histories["acc::test"] == [1.0]
    assert model.training_flags == [False]
    assert model.training is True
