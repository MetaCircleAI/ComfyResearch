from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _node(node_id: str, node_type: str, data: dict[str, object]) -> Node:
    return Node(id=node_id, type=NodeKind(node_type), data=data)


def _edge(
    edge_id: str,
    source: str,
    target: str,
    source_handle: str,
    target_handle: str,
) -> Edge:
    return Edge(
        id=edge_id,
        source=source,
        target=target,
        sourceHandle=source_handle,
        targetHandle=target_handle,
    )


def test_metrics_events_snapshot_each_log_interval() -> None:
    nodes = [
        _node(
            "dataset",
            "linear_dataset",
            {
                "inputDim": 2,
                "outputDim": 1,
                "trainSize": 8,
                "testSize": 4,
                "noiseLevel": 0,
                "seed": 0,
                "samplingMode": "fixed",
            },
        ),
        _node(
            "model",
            "mlp_model",
            {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0},
        ),
        _node("optimizer", "adam_optimizer", {"learningRate": 0.01}),
        _node("loss", "mse_loss", {}),
        _node(
            "trainer",
            "trainer",
            {
                "trainingSteps": 6,
                "logFrequency": 2,
                "batchSize": -1,
                "computeDevice": "cpu",
            },
        ),
    ]
    edges = [
        _edge("dataset-trainer", "dataset", "trainer", "dataset", "dataset"),
        _edge("model-trainer", "model", "trainer", "model", "model"),
        _edge("optimizer-trainer", "optimizer", "trainer", "optimizer", "optimizer"),
        _edge("loss-trainer", "loss", "trainer", "loss", "loss"),
    ]

    events = list(iter_trainer_events(nodes, edges, "trainer"))
    metrics = [event for event in events if event["type"] == "metrics"]
    complete = next(event for event in events if event["type"] == "complete")

    assert [int(event["step"]) for event in metrics] == [2, 4, 6]
    assert [int(step) for step in complete["step_ticks"]] == [0, 2, 4, 6]
    assert len(metrics) == len(complete["step_ticks"]) - 1

    for event in metrics:
        step_ticks = [int(step) for step in event["step_ticks"]]
        assert step_ticks[-1] == int(event["step"])
        assert len(event["loss_history"]) == len(step_ticks)
        assert len(event["test_loss_history"]) == len(step_ticks)
        assert len(event["reg_loss_history"]) == len(step_ticks)
        for series in (
            event["loss_history"],
            event["test_loss_history"],
            event["reg_loss_history"],
        ):
            assert all(math.isfinite(float(value)) for value in series)

    assert metrics[-1]["loss_history"] == complete["loss_history"]
    assert metrics[-1]["test_loss_history"] == complete["test_loss_history"]
    assert metrics[-1]["reg_loss_history"] == complete["reg_loss_history"]