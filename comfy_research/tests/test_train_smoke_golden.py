from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "train_smoke_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_TRAIN_SMOKE_GOLDEN"


def _node(node_id: str, node_type: str, data: dict[str, Any] | None = None) -> Node:
    return Node(id=node_id, type=NodeKind(node_type), data=dict(data or {}))


def _edge(edge_id: str, source: str, target: str, source_handle: str, target_handle: str) -> Edge:
    return Edge(
        id=edge_id,
        source=source,
        target=target,
        sourceHandle=source_handle,
        targetHandle=target_handle,
    )


def _fixtures() -> dict[str, dict[str, Any]]:
    return {
        "vector_mlp_mse": {
            "trainer_node_id": "trainer",
            "nodes": [
                {
                    "id": "dataset",
                    "type": "linear_dataset",
                    "data": {
                        "inputDim": 2,
                        "outputDim": 1,
                        "trainSize": 8,
                        "testSize": 4,
                        "noiseLevel": 0,
                        "seed": 0,
                        "samplingMode": "fixed",
                    },
                },
                {
                    "id": "model",
                    "type": "mlp_model",
                    "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0},
                },
                {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
                {"id": "loss", "type": "mse_loss", "data": {}},
                {
                    "id": "trainer",
                    "type": "trainer",
                    "data": {"trainingSteps": 3, "logFrequency": 1, "batchSize": -1, "computeDevice": "cpu"},
                },
            ],
            "edges": [
                ["dataset-trainer", "dataset", "trainer", "dataset", "dataset"],
                ["model-trainer", "model", "trainer", "model", "model"],
                ["optimizer-trainer", "optimizer", "trainer", "optimizer", "optimizer"],
                ["loss-trainer", "loss", "trainer", "loss", "loss"],
            ],
            "expected": {"steps": [0, 1, 2, 3], "loss_points": 4, "test_loss_points": 4, "reg_loss_points": 4},
        },
        "token_mlp_cross_entropy": {
            "trainer_node_id": "trainer",
            "nodes": [
                {
                    "id": "dataset",
                    "type": "token_prediction_dataset",
                    "data": {
                        "vocabSize": 5,
                        "contextLength": 3,
                        "whichToken": -1,
                        "trainSize": 8,
                        "testSize": 4,
                        "seed": 0,
                        "samplingMode": "fixed",
                    },
                },
                {
                    "id": "model",
                    "type": "mlp_token_model",
                    "data": {
                        "vocabSize": 5,
                        "embedDim": 4,
                        "tokensPerInput": 3,
                        "depth": 1,
                        "width": 6,
                        "activation": "relu",
                        "tieWeights": "no",
                        "seed": 0,
                    },
                },
                {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
                {"id": "loss", "type": "cross_entropy_loss", "data": {}},
                {
                    "id": "trainer",
                    "type": "trainer",
                    "data": {"trainingSteps": 3, "logFrequency": 1, "batchSize": -1, "computeDevice": "cpu"},
                },
            ],
            "edges": [
                ["dataset-trainer", "dataset", "trainer", "dataset", "dataset"],
                ["model-trainer", "model", "trainer", "model", "model"],
                ["optimizer-trainer", "optimizer", "trainer", "optimizer", "optimizer"],
                ["loss-trainer", "loss", "trainer", "loss", "loss"],
            ],
            "expected": {"steps": [0, 1, 2, 3], "loss_points": 4, "test_loss_points": 4, "reg_loss_points": 4},
        },
    }


def _round_series(values: list[float]) -> list[float]:
    return [round(float(value), 8) for value in values]


def _run_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    nodes = [_node(str(raw["id"]), str(raw["type"]), raw.get("data")) for raw in fixture["nodes"]]
    edges = [_edge(*raw) for raw in fixture["edges"]]
    progress: list[list[int]] = []
    complete: dict[str, Any] | None = None

    for event in iter_trainer_events(nodes, edges, str(fixture["trainer_node_id"])):
        if event["type"] == "progress":
            progress.append([int(event["step"]), int(event["total"])])
        elif event["type"] == "complete":
            complete = event

    assert complete is not None
    loss_history = [float(x) for x in complete["loss_history"]]
    test_loss_history = [float(x) for x in complete["test_loss_history"]]
    reg_loss_history = [float(x) for x in complete["reg_loss_history"]]
    step_ticks = [int(x) for x in complete["step_ticks"]]

    for series in (loss_history, test_loss_history, reg_loss_history):
        assert all(math.isfinite(value) for value in series)

    return {
        "progress": progress,
        "steps": step_ticks,
        "loss_history_rounded": _round_series(loss_history),
        "test_loss_history_rounded": _round_series(test_loss_history),
        "reg_loss_history_rounded": _round_series(reg_loss_history),
        "observable_metric_shapes": {str(key): len(value) for key, value in complete["observable_metric_histories"].items()},
    }


def _build_current_snapshot() -> dict[str, Any]:
    fixtures = _fixtures()
    return {
        "version": 1,
        "fixtures": {
            name: {
                "graph": {"nodes": fixture["nodes"], "edges": fixture["edges"], "trainer_node_id": fixture["trainer_node_id"]},
                "expected": fixture["expected"],
                "actual": _run_fixture(fixture),
            }
            for name, fixture in fixtures.items()
        },
    }


def test_train_smoke_golden() -> None:
    current = _build_current_snapshot()

    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if not expected.get("fixtures"):
        pytest.skip(f"bootstrap train smoke snapshot with {UPDATE_ENV}=1 in an environment with torch installed")

    # Structure (graph, hand-authored expected shape, progress/steps/observable
    # shapes) must match exactly. Loss series are compared with tolerance: torch
    # is not bit-reproducible across platforms (the snapshot is captured on one
    # OS/arch, CI runs on another), so exact float equality is flaky. A real
    # trainer regression changes losses by orders of magnitude more than 1e-4.
    _FLOAT_SERIES = ("loss_history_rounded", "test_loss_history_rounded", "reg_loss_history_rounded")
    assert set(current["fixtures"]) == set(expected["fixtures"])
    for name, cur_fx in current["fixtures"].items():
        exp_fx = expected["fixtures"][name]
        assert cur_fx["graph"] == exp_fx["graph"], f"{name}: graph"
        assert cur_fx["expected"] == exp_fx["expected"], f"{name}: expected shape"
        cur_a, exp_a = cur_fx["actual"], exp_fx["actual"]
        for key in ("progress", "steps", "observable_metric_shapes"):
            assert cur_a[key] == exp_a[key], f"{name}.{key}"
        for key in _FLOAT_SERIES:
            cur_series, exp_series = cur_a[key], exp_a[key]
            assert len(cur_series) == len(exp_series), f"{name}.{key}: length"
            for c, e in zip(cur_series, exp_series):
                assert math.isclose(c, e, rel_tol=1e-4, abs_tol=1e-4), f"{name}.{key}: {c} vs {e}"
