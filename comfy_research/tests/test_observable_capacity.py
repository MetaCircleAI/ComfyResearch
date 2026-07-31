"""Regression: observable_capacity must record finite bits for integer-target runs.

The capacity formula ``(log(vocab) - train_loss) * batch / log(2)`` referenced
an undefined ``train_val`` in the legacy implementation, so wiring
observable_capacity to any integer-target dataset (vocab_size > 1) crashed the
training stream with a NameError. Float-target datasets take the NaN branch
and never hit it, which is how the bug survived.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


def _graph() -> tuple[list[Node], list[Edge], str]:
    nodes_raw: list[dict[str, Any]] = [
        {"id": "dataset", "type": "token_prediction_dataset",
         "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8,
                  "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "mlp_token_model",
         "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6,
                  "activation": "relu", "tieWeights": "no", "seed": 0}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "cross_entropy_loss", "data": {}},
        {"id": "cap", "type": "observable_capacity", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 2, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-ds", "dataset", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-o", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-c", "cap", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def test_observable_capacity_records_finite_bits_for_token_targets() -> None:
    nodes, edges, tid = _graph()
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    assert complete, f"no complete event; got {[e.get('type') for e in events]}"
    hist = complete[0]["observable_metric_histories"]["cap"]
    assert len(hist) >= 2
    assert all(math.isfinite(v) for v in hist), hist
    # capacity = (ln(vocab) - loss) * batch / ln 2 with vocab=5, batch=8; the
    # step-0 loss of an untrained model is ~ln(5), so bits start near 0 and
    # must stay well below the ceiling log2(5)*8.
    assert all(v < math.log2(5) * 8 + 1e-6 for v in hist), hist
