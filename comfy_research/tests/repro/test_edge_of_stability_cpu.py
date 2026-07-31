"""
Template:
    edge-of-stability-cpu

Purpose:
    A small full-batch CPU demonstration of the edge of stability: the leading
    Hessian eigenvalue reaches 2 / η while the second remains stable and loss
    is locally non-monotonic but decreases overall.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.schemas.graph import Edge, GraphDocument, Node
from comfy_research.schemas.saved_graph_library import SavedGraphEntry


pytestmark = pytest.mark.repro

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATE_PATH = _REPO_ROOT / "data/graph_library/templates/edge-of-stability-cpu.json"


def _load() -> SavedGraphEntry:
    return SavedGraphEntry.model_validate(json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8")))


def _type(node: Node) -> str:
    return node.type.value if hasattr(node.type, "value") else str(node.type)


def _one(document: GraphDocument, node_type: str) -> Node:
    matches = [node for node in document.nodes if _type(node) == node_type]
    assert len(matches) == 1
    return matches[0]


def _edge_signature(document: GraphDocument, edge: Edge) -> tuple[str, str | None, str, str | None]:
    node_types = {node.id: _type(node) for node in document.nodes}
    return (
        node_types[edge.source],
        edge.sourceHandle,
        node_types[edge.target],
        edge.targetHandle,
    )


def _complete(document: GraphDocument, trainer_id: str) -> dict:
    events = list(iter_trainer_events(list(document.nodes), list(document.edges), trainer_id))
    complete = [event for event in events if event.get("type") == "complete"]
    assert len(complete) == 1
    return complete[0]


def test_template_baseline() -> None:
    entry = _load()
    document = entry.document
    assert entry.id == "edge-of-stability-cpu"
    assert entry.name == "Edge of Stability (CPU)"

    expected = {
        "linear_dataset",
        "mlp_model",
        "mse_loss",
        "sgd_optimizer",
        "trainer",
        "training_visualization",
        "observable_hessian_eigenvalues",
        "observable_viz",
    }
    assert expected <= {_type(node) for node in document.nodes}

    dataset = _one(document, "linear_dataset").data
    model = _one(document, "mlp_model").data
    optimizer = _one(document, "sgd_optimizer").data
    trainer = _one(document, "trainer").data
    observable = _one(document, "observable_hessian_eigenvalues").data
    visualization = _one(document, "observable_viz").data
    assert dataset["inputDim"] == 10
    assert dataset["outputDim"] == 1
    assert dataset["trainSize"] == 80
    assert dataset["testSize"] == 0
    assert dataset["noiseLevel"] == pytest.approx(0.1)
    assert dataset["seed"] == 0
    assert model["depth"] == 2
    assert model["width"] == 24
    assert model["activation"] == "relu"
    assert model["seed"] == 0
    assert optimizer["learningRate"] == pytest.approx(0.2)
    assert optimizer["momentum"] == pytest.approx(0.0)
    assert optimizer["weightDecay"] == pytest.approx(0.0)
    assert trainer["trainingSteps"] == 80
    assert trainer["logFrequency"] == 1
    assert trainer["computeDevice"] == "cpu"
    assert trainer["batchSize"] == -1
    assert observable["topK"] == 2
    assert observable["order"] == "descending"
    assert visualization["topK"] == 2
    assert visualization["sharpnessThreshold"] == pytest.approx(10.0)

    signatures = {_edge_signature(document, edge) for edge in document.edges}
    assert {
        ("linear_dataset", "dataset", "trainer", "dataset"),
        ("mlp_model", "model", "trainer", "model"),
        ("mse_loss", "loss", "trainer", "loss"),
        ("sgd_optimizer", "optimizer", "trainer", "optimizer"),
        ("observable_hessian_eigenvalues", "observables", "trainer", "observables"),
        ("trainer", "loss_results", "training_visualization", "tensor_list"),
        ("trainer", "observable_results", "observable_viz", "tensor"),
    } <= signatures


def test_template_smoke_run() -> None:
    original = _TEMPLATE_PATH.read_text(encoding="utf-8")
    document = copy.deepcopy(_load().document)
    dataset = _one(document, "linear_dataset").data
    trainer = _one(document, "trainer")
    assert trainer.data is not None
    dataset["trainSize"] = 16
    trainer.data["trainingSteps"] = 4
    trainer.data["logFrequency"] = 1

    payload = _complete(document, trainer.id)
    assert payload["step_ticks"] == [0, 1, 2, 3, 4]
    assert all(math.isfinite(float(value)) for value in payload["loss_history"])
    assert _TEMPLATE_PATH.read_text(encoding="utf-8") == original


@pytest.mark.slow
def test_edge_of_stability_behavior() -> None:
    document = _load().document
    trainer = _one(document, "trainer")
    observable = _one(document, "observable_hessian_eigenvalues")
    payload = _complete(document, trainer.id)
    lambda_1 = payload["observable_metric_histories"][f"{observable.id}::0"]
    lambda_2 = payload["observable_metric_histories"][f"{observable.id}::1"]
    loss = payload["loss_history"]

    cutoff = 2.0 / _one(document, "sgd_optimizer").data["learningRate"]
    crossing = next(index for index, value in enumerate(lambda_1) if math.isfinite(value) and value >= cutoff)
    assert max(value for value in lambda_2 if math.isfinite(value)) < cutoff
    assert any(after > before for before, after in zip(loss[crossing:], loss[crossing + 1 :]))
    assert loss[-1] < loss[crossing]
