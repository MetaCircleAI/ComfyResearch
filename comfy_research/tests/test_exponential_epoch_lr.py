from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import (
    iter_trainer_events_from_context,
    prepare_trainer_run,
)
from comfy_research.engine.trainer.training_loop import _trainer_lr_mult_for_step
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _mult(step: int, *, warmup: int = 0, every: int = 1) -> float:
    return _trainer_lr_mult_for_step(
        step,
        training_steps=10_000,
        warmup_steps=warmup,
        schedule="exponential_epoch",
        cosine_min_fraction=0.0,
        steps_per_epoch=391,
        exponential_decay_factor=0.95,
        exponential_decay_epochs=every,
    )


def test_exponential_epoch_lr_decays_at_complete_epoch_boundaries() -> None:
    assert _mult(0) == 1.0
    assert _mult(390) == 1.0
    assert _mult(391) == pytest.approx(0.95)
    assert _mult(782) == pytest.approx(0.95**2)


def test_exponential_epoch_lr_supports_multi_epoch_period_and_warmup() -> None:
    assert _mult(0, warmup=2) == pytest.approx(0.5)
    assert _mult(1, warmup=2) == pytest.approx(1.0)
    assert _mult(2 + 2 * 391 - 1, warmup=2, every=2) == 1.0
    assert _mult(2 + 2 * 391, warmup=2, every=2) == pytest.approx(0.95)


def test_graph_schedule_reaches_context_and_optimizer_step_lrs() -> None:
    """A graph-node schedule must affect the real optimizer, not only the helper."""
    nodes = [
        Node(id="dataset", type=NodeKind.linear_dataset, data={
            "inputDim": 2, "outputDim": 1, "trainSize": 4, "testSize": 0,
            "noiseLevel": 0, "seed": 0, "samplingMode": "fixed",
        }),
        Node(id="model", type=NodeKind.mlp_model, data={
            "inputDim": 2, "outputDim": 1, "depth": 1, "width": 4,
            "activation": "relu", "seed": 0,
        }),
        Node(id="optimizer", type=NodeKind.sgd_optimizer, data={"learningRate": 0.1, "momentum": 0}),
        Node(id="loss", type=NodeKind.mse_loss, data={}),
        Node(id="schedule", type=NodeKind.lr_schedule, data={
            "lrWarmupSteps": 0, "lrSchedule": "exponential_epoch",
            "exponentialDecayFactor": 0.5, "exponentialDecayEpochs": 1,
        }),
        Node(id="trainer", type=NodeKind.trainer, data={
            "computeDevice": "cpu", "batchSize": 2, "trainingSteps": 5, "logFrequency": 5,
        }),
    ]
    edges = [
        Edge(id="dataset-edge", source="dataset", target="trainer", sourceHandle="dataset", targetHandle="dataset"),
        Edge(id="model-edge", source="model", target="trainer", sourceHandle="model", targetHandle="model"),
        Edge(id="optimizer-edge", source="optimizer", target="trainer", sourceHandle="optimizer", targetHandle="optimizer"),
        Edge(id="loss-edge", source="loss", target="trainer", sourceHandle="loss", targetHandle="loss"),
        Edge(id="schedule-edge", source="schedule", target="optimizer", sourceHandle="lr_schedule", targetHandle="lr_schedule"),
    ]

    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.lr_schedule == "exponential_epoch"
    assert ctx.exponential_lr_decay_factor == 0.5
    assert ctx.exponential_lr_decay_epochs == 1
    assert ctx.cyclic_steps_per_epoch == 2

    observed_lrs: list[float] = []
    original_step = ctx.optimizer.step

    def capture_step(*args, **kwargs):
        observed_lrs.append(float(ctx.optimizer.param_groups[0]["lr"]))
        return original_step(*args, **kwargs)

    ctx.optimizer.step = capture_step
    assert [event["type"] for event in iter_trainer_events_from_context(ctx)][-1] == "complete"
    assert observed_lrs == pytest.approx([0.1, 0.1, 0.05, 0.05, 0.025])
