from __future__ import annotations

import torch

from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.engine.trainer.eval_batches import (
    _batched_classification_accuracy,
    _batched_primary_loss_mean,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


class _PerfectTokenClassifier(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x.long(), num_classes=2).float()


class _BatchNormDropoutProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(p=0.9)
        self.mode_pairs: list[tuple[bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mode_pairs.append((self.batch_norm.training, self.dropout.training))
        return self.dropout(self.batch_norm(x))


def test_batched_token_accuracy_counts_tokens() -> None:
    target = torch.tensor([[0, 1, 0], [1, 1, 0]])

    accuracy = _batched_classification_accuracy(
        _PerfectTokenClassifier(),
        target.float(),
        target,
        batch_size=1,
        trainer_task="token_classification",
    )

    assert accuracy == 1.0


def test_batched_eval_can_use_batch_norm_batch_stats_without_dropout() -> None:
    model = _BatchNormDropoutProbe()
    model.train()
    with torch.no_grad():
        model.batch_norm.running_mean.fill_(100.0)
    running_mean = model.batch_norm.running_mean.clone()
    x = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])

    loss = _batched_primary_loss_mean(
        model,
        x,
        y,
        batch_size=4,
        trainer_task="vision_classification",
        criterion=torch.nn.CrossEntropyLoss(),
        loss_scale=1.0,
        batch_norm_batch_stats=True,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert model.mode_pairs == [(True, False)]
    torch.testing.assert_close(model.batch_norm.running_mean, running_mean)
    assert model.training is True


def test_trainer_bounds_loss_and_accuracy_eval_forward_batches(monkeypatch) -> None:
    max_eval_batch = 256
    seen_batch_sizes: list[int] = []
    original_forward = torch.nn.Sequential.forward

    def guarded_forward(model: torch.nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        if not model.training:
            batch_size = int(x.shape[0])
            seen_batch_sizes.append(batch_size)
            assert batch_size <= max_eval_batch, f"unbounded eval forward: {batch_size}"
        return original_forward(model, x)

    monkeypatch.setattr(torch.nn.Sequential, "forward", guarded_forward)

    nodes = [
        Node(
            id="dataset",
            type=NodeKind.gaussian_blob_dataset,
            data={
                "trainSize": 600,
                "testSize": 300,
                "numClasses": 2,
                "imageSize": 8,
                "flattenOutput": True,
                "samplingMode": "fixed",
                "seed": 0,
            },
        ),
        Node(
            id="model",
            type=NodeKind.mlp_model,
            data={"inputDim": 64, "outputDim": 2, "depth": 1, "width": 8, "seed": 0},
        ),
        Node(
            id="optimizer", type=NodeKind.adam_optimizer, data={"learningRate": 0.001}
        ),
        Node(id="loss", type=NodeKind.cross_entropy_loss, data={}),
        Node(id="accuracy", type=NodeKind.observable_accuracy, data={}),
        Node(
            id="trainer",
            type=NodeKind.trainer,
            data={
                "trainingSteps": 1,
                "logFrequency": 1,
                "batchSize": 512,
                "computeDevice": "cpu",
            },
        ),
    ]
    edges = [
        Edge(
            id="dataset-trainer",
            source="dataset",
            target="trainer",
            sourceHandle="dataset",
            targetHandle="dataset",
        ),
        Edge(
            id="model-trainer",
            source="model",
            target="trainer",
            sourceHandle="model",
            targetHandle="model",
        ),
        Edge(
            id="optimizer-trainer",
            source="optimizer",
            target="trainer",
            sourceHandle="optimizer",
            targetHandle="optimizer",
        ),
        Edge(
            id="loss-trainer",
            source="loss",
            target="trainer",
            sourceHandle="loss",
            targetHandle="loss",
        ),
        Edge(
            id="accuracy-trainer",
            source="accuracy",
            target="trainer",
            sourceHandle="observables",
            targetHandle="observables",
        ),
    ]

    events = list(iter_trainer_events(nodes, edges, "trainer"))

    assert events[-1]["type"] == "complete"
    assert seen_batch_sizes
