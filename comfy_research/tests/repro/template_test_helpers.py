from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.schemas.graph import Edge, GraphDocument, Node
from comfy_research.schemas.saved_graph_library import SavedGraphEntry


_REPO = Path(__file__).resolve().parents[3]
_TEMPLATES = _REPO / "data" / "graph_library" / "templates"
_VISION_BUILD = "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"


def load_template(slug: str) -> tuple[SavedGraphEntry, list[Node], list[Edge]]:
    raw = json.loads((_TEMPLATES / f"{slug}.json").read_text(encoding="utf-8"))
    entry = SavedGraphEntry.model_validate(raw)
    document = GraphDocument.model_validate(entry.document)
    return entry, list(document.nodes), list(document.edges)


def node_by_type(nodes: list[Node], node_type: str) -> Node:
    matches = [node for node in nodes if node.type == node_type]
    assert len(matches) == 1, f"expected one {node_type}, got {len(matches)}"
    return matches[0]


def nodes_by_type(nodes: list[Node], node_type: str) -> list[Node]:
    return [node for node in nodes if node.type == node_type]


def has_edge(
    edges: list[Edge],
    source: str,
    target: str,
    source_handle: str,
    target_handle: str,
) -> bool:
    return any(
        edge.source == source
        and edge.target == target
        and edge.sourceHandle == source_handle
        and edge.targetHandle == target_handle
        for edge in edges
    )


def _synthetic_cifar(
    _kind: object,
    _data: dict,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    image_size = 28 if _data.get("preprocessing") == "center_crop_28_per_image_whiten" else 32
    x_train = rng.normal(size=(train_size, 3, image_size, image_size)).astype(np.float32)
    y_train = np.arange(train_size, dtype=np.int64) % 10
    if test_size <= 0:
        return x_train, y_train, None, None
    x_test = rng.normal(size=(test_size, 3, image_size, image_size)).astype(np.float32)
    y_test = np.arange(test_size, dtype=np.int64) % 10
    return x_train, y_train, x_test, y_test


def run_reduced_cpu_smoke(nodes: list[Node], edges: list[Edge]) -> None:
    smoke_nodes = [node.model_copy(deep=True) for node in nodes]
    smoke_edges = [edge.model_copy(deep=True) for edge in edges]
    for node in smoke_nodes:
        data = node.data or {}
        if node.type == "cifar10_dataset":
            data["trainSize"] = 8
            data["testSize"] = 4
        elif node.type == "vgg11_cifar_model":
            data["seed"] = 0
        elif node.type == "trainer":
            data.update(
                trainingLengthMode="steps",
                trainingSteps=1,
                trainingEpochs=1,
                logFrequency=1,
                logSchedule="fixed_interval",
                batchSize=4,
                minibatchSampling="independent_step",
                computeDevice="cpu",
                remoteGpu=False,
            )
        elif node.type == "cyclic_batch_schedule":
            data.update(batchMin=4, batchMax=8, refBatchSize=4, cycleLengthEpochs=2)
        elif node.type == "cyclic_lr_schedule":
            data.update(refBatchSize=4, cycleLengthEpochs=2)

    trainers = nodes_by_type(smoke_nodes, "trainer")
    assert trainers
    with patch(_VISION_BUILD, side_effect=_synthetic_cifar):
        for trainer in trainers:
            events = list(iter_trainer_events(smoke_nodes, smoke_edges, trainer.id))
            assert events[-1]["type"] == "complete"
            complete = events[-1]
            assert complete["loss_history"]
            for key in ("loss_history", "test_loss_history", "reg_loss_history"):
                assert all(math.isfinite(float(value)) for value in complete[key])
