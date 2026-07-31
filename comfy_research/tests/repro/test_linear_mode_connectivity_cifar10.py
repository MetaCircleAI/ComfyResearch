"""
Template:
    repro: Linear Mode Connectivity on CIFAR-10

Purpose:
    Same-initialization / different-training-seed CIFAR-10 classifier workflow
    with linear and quadratic Bezier connectivity observables.
"""

from __future__ import annotations

import base64
import io
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from comfy_research.engine.analysis.parametric_path_sampler import run_parametric_path_sampler
from comfy_research.engine.runs.trainer_run import iter_trainer_events, prepare_trainer_run
from comfy_research.tests.repro.template_test_helpers import (
    has_edge,
    load_template,
    node_by_type,
    nodes_by_type,
)


pytestmark = pytest.mark.repro

_VISION_BUILD = "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"


def _synthetic_cifar(
    _kind: object,
    _data: dict,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    x_train = rng.normal(size=(train_size, 3, 32, 32)).astype(np.float32)
    y_train = np.arange(train_size, dtype=np.int64) % 10
    x_test = rng.normal(size=(test_size, 3, 32, 32)).astype(np.float32)
    y_test = np.arange(test_size, dtype=np.int64) % 10
    return x_train, y_train, x_test, y_test


def _checkpoint_b64(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({"model": state_dict}, buffer)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _reduced_graph() -> tuple[list, list]:
    _entry, nodes, edges = load_template("repro-linear-mode-connectivity-cifar10")
    nodes = [node.model_copy(deep=True) for node in nodes]
    edges = [edge.model_copy(deep=True) for edge in edges]
    for node in nodes:
        data = node.data or {}
        if node.type == "cifar10_dataset":
            data.update(trainSize=8, testSize=4)
        elif node.type == "resnet_model":
            data.update(variant="self_defined", baseChannels=8, blocksStage1=1, blocksStage2=1, blocksStage3=1, blocksStage4=1)
        elif node.type == "trainer":
            data.update(trainingLengthMode="steps", trainingSteps=1, logFrequency=1, batchSize=4, computeDevice="cpu", remoteGpu=False)
        elif node.type == "observable_linear_interpolation_barrier":
            data.update(alphaSteps=3, recomputeBnStats=False, evalBatchSize=4)
        elif node.type == "observable_bezier_mode_connectivity":
            data.update(alphaSteps=3, curveOptimizationSteps=1, curveSamplesPerStep=1, curveBatchSize=4, recomputeBnStats=False, evalBatchSize=4)
    return nodes, edges


def test_template_baseline() -> None:
    entry, nodes, edges = load_template("repro-linear-mode-connectivity-cifar10")
    assert entry.tier == "small"

    dataset = node_by_type(nodes, "cifar10_dataset")
    model = node_by_type(nodes, "resnet_model")
    loss = node_by_type(nodes, "cross_entropy_loss")
    barrier = node_by_type(nodes, "observable_linear_interpolation_barrier")
    bezier = node_by_type(nodes, "observable_bezier_mode_connectivity")

    assert dataset.data["trainSize"] == 50_000
    assert dataset.data["testSize"] == 10_000
    assert dataset.data["subsetSeed"] == 0
    assert model.data["seed"] == 0
    assert model.data["baseChannels"] == 16

    trainers = sorted(nodes_by_type(nodes, "trainer"), key=lambda node: node.id)
    assert [trainer.data["trainSeed"] for trainer in trainers] == [0, 1]
    assert all(trainer.data["trainingSteps"] == 50_000 for trainer in trainers)
    assert all(trainer.data["batchSize"] == 128 for trainer in trainers)

    checkpoints = sorted(nodes_by_type(nodes, "model_checkpoint"), key=lambda node: node.id)
    assert [checkpoint.data["checkpointSource"] for checkpoint in checkpoints] == ["memory", "memory"]
    assert barrier.data["alphaSteps"] == 21
    assert barrier.data["bnCalibrationBatches"] == 100
    assert bezier.data["alphaSteps"] == 21
    assert bezier.data["curveOptimizationSteps"] == 500
    assert bezier.data["bnCalibrationBatches"] == 100

    for trainer in trainers:
        assert has_edge(edges, dataset.id, trainer.id, "dataset", "dataset")
        assert has_edge(edges, model.id, trainer.id, "model", "model")
    for observable in (barrier, bezier):
        assert has_edge(edges, dataset.id, observable.id, "dataset", "dataset")
        assert has_edge(edges, model.id, observable.id, "model", "model")
        assert has_edge(edges, loss.id, observable.id, "loss", "loss")
        assert has_edge(edges, "lmc-checkpoint-a", observable.id, "model", "checkpoint_a")
        assert has_edge(edges, "lmc-checkpoint-b", observable.id, "model", "checkpoint_b")


def test_template_smoke_run() -> None:
    """Reduced template trains both endpoints and runs linear/Bezier observables on CPU."""
    nodes, edges = _reduced_graph()

    with patch(_VISION_BUILD, side_effect=_synthetic_cifar):
        for trainer in nodes_by_type(nodes, "trainer"):
            events = list(iter_trainer_events(nodes, edges, trainer.id))
            assert events[-1]["type"] == "complete"
            assert all(math.isfinite(float(value)) for value in events[-1]["loss_history"])

        context = prepare_trainer_run(nodes, edges, "lmc-trainer-a")

    state_a = {key: value.detach().cpu().clone() for key, value in context.model.state_dict().items()}
    state_b = {key: value.detach().cpu().clone() for key, value in context.model.state_dict().items()}
    for key, value in state_b.items():
        if torch.is_floating_point(value):
            state_b[key] = value + 0.001
    for node in nodes:
        if node.id == "lmc-checkpoint-a":
            node.data.update(checkpointSource="memory", memoryCheckpoint_b64=_checkpoint_b64(state_a))
        elif node.id == "lmc-checkpoint-b":
            node.data.update(checkpointSource="memory", memoryCheckpoint_b64=_checkpoint_b64(state_b))

    with patch(_VISION_BUILD, side_effect=_synthetic_cifar):
        linear = run_parametric_path_sampler(nodes, edges, "lmc-barrier")
        bezier = run_parametric_path_sampler(nodes, edges, "lmc-bezier")

    assert linear["alpha"] == [0.0, 0.5, 1.0]
    assert set(linear) >= {"train_loss", "test_loss", "train_acc", "test_acc", "lossBarrier", "accuracyDrop"}
    assert all(math.isfinite(float(value)) for value in linear["test_loss"])
    assert math.isfinite(float(linear["lossBarrier"]))
    assert bezier["alphaSeries"] == [0.0, 0.5, 1.0]
    assert set(bezier) >= {"bezierTrainLoss", "bezierTestLoss", "bezierTrainAcc", "bezierTestAcc", "linearLossBarrier", "bezierLossBarrier"}
    assert all(math.isfinite(float(value)) for value in bezier["bezierTestLoss"])
    assert math.isfinite(float(bezier["bezierLossBarrier"]))
