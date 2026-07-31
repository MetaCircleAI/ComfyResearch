"""In-context associative recall template: phase-transition regression test.

Mirrors the runnable core of template f67f1c6e: associative-recall data,
a causal token transformer, accuracy, and its two attention-relation scores.
The full template is intentionally not loaded because it contains saved model
state and visualization-only nodes.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


def _graph(*, steps: int = 2000) -> tuple[list[Node], list[Edge], str]:
    nodes_raw: list[dict[str, Any]] = [
        {"id": "dataset", "type": "in_context_associative_recall_dataset",
         "data": {"vocabSize": 8, "numPairs": 4, "inContextRepeat": 1,
                  "crossSampleRepeatProb": 0, "repeatedTokenCount": 2,
                  "trainSize": 10000, "testSize": 2000, "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "transformer_token_model",
         "data": {"vocabSize": 8, "contextLength": 9, "modelDim": 6, "numHeads": 2,
                  "numLayers": 2, "ffDim": 24, "activation": "gelu", "encoderBackend": "pytorch",
                  "encoderDropout": 0, "spectralNormLinears": "no", "lmLogitScale": 1,
                  "stableQkNorm": "no", "stableAttnTemperature": 1, "stableAttnLogitCap": 0,
                  "stableAttnDropout": 0, "tieEmbeddingLmHead": "yes", "causalAttention": "yes",
                  "localMixingKernel": 0, "seed": 0}},
        {"id": "opt", "type": "adam_optimizer",
         "data": {"learningRate": 0.001, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
                  "weightDecay": 0}},
        {"id": "loss", "type": "cross_entropy_loss", "data": {"lossScale": 1, "labelSmoothing": 0,
                                                                        "lossMaskContextLength": 1, "lossMaskMode": "all"}},
        {"id": "obs-accuracy", "type": "observable_accuracy", "data": {}},
        {"id": "obs-layer1", "type": "observable_attention_relation_score",
         "data": {"keyRelation": "tok(k-1) == tok(q) and pos(k) % 2 == 1", "queryFilter": "pos(q) == -1",
                  "keyReduction": "mean", "layerIndex": 1, "headIndex": 1}},
        {"id": "obs-layer0", "type": "observable_attention_relation_score",
         "data": {"keyRelation": "pos(k) == pos(q) - 1", "queryFilter": "pos(q) % 2 == 1",
                  "keyReduction": "mean", "layerIndex": 0, "headIndex": 1}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": 256, "trainingSteps": steps, "logFrequency": steps}},
    ]
    edges_raw = [
        ["e-dataset", "dataset", "trainer", "dataset", "dataset"],
        ["e-model", "model", "trainer", "model", "model"],
        ["e-opt", "opt", "trainer", "optimizer", "optimizer"],
        ["e-loss", "loss", "trainer", "loss", "loss"],
        ["e-accuracy", "obs-accuracy", "trainer", "observables", "observables"],
        ["e-layer1", "obs-layer1", "trainer", "observables", "observables"],
        ["e-layer0", "obs-layer0", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _run(**kw: Any) -> dict[str, Any]:
    nodes, edges, trainer_id = _graph(**kw)
    events = list(iter_trainer_events(nodes, edges, trainer_id))
    complete = [event for event in events if event.get("type") == "complete"]
    assert complete, f"no complete event; got {[event.get('type') for event in events]}"
    return complete[0]


def test_associative_recall_attention_phase_transitions_and_perfect_accuracy() -> None:
    done = _run()
    metrics = done["observable_metric_histories"]
    for key in ("obs-accuracy", "obs-accuracy::test", "obs-layer1", "obs-layer0"):
        assert len(metrics[key]) == 2, (key, metrics[key])
        assert all(math.isfinite(value) for value in metrics[key]), (key, metrics[key])
    assert metrics["obs-accuracy"][-1] >= 0.9
    assert metrics["obs-accuracy::test"][-1] >= 0.9

    # The template's layer-1 key-to-value and layer-0 induction-head scores
    # both transition sharply while the recall task is solved.
    for key in ("obs-layer1", "obs-layer0"):
        history = metrics[key]
        assert history[0] < 0.4, (key, history)
        assert history[-1] - history[0] > 0.3, (key, history)
        assert history[-1] > 0.9, (key, history)
