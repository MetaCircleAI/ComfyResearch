"""prepare_trainer_run accepts cyclic schedule nodes on the Jastrzębski Fig 1 left templates.

Pins the paper-repro arithmetic (trainSize=50000, ref batch 128):
- CBS: 300 data epochs, batch 128..640 over 10-epoch cycles -> 48720 optimizer steps.
- CLR: 300 epochs x steps_per_epoch(50000, 128)=391 -> 117300 optimizer steps.

validate_only=True: the vision dataset build (CIFAR download) must be skipped.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import prepare_trainer_run  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

_TRAINER_EPOCHS = {
    "computeDevice": "cpu",
    "logFrequency": 10,
    "trainingLengthMode": "epochs",
    "trainingEpochs": 300,
}
_CIFAR = {"trainSize": 50000, "testSize": 0, "initSeed": 0, "seed": 0, "samplingMode": "fixed"}


def _base_graph(trainer_data: dict) -> tuple[list[Node], list[Edge]]:
    nodes = [
        Node(id="dataset", type=NodeKind.cifar10_dataset, data=dict(_CIFAR)),
        Node(id="model", type=NodeKind.vgg11_cifar_model, data={"seed": 0}),
        Node(id="optimizer", type=NodeKind.sgd_optimizer, data={"learningRate": 0.001, "momentum": 0.9}),
        Node(id="loss", type=NodeKind.cross_entropy_loss, data={}),
        Node(id="trainer", type=NodeKind.trainer, data=dict(trainer_data)),
    ]
    edges = [
        Edge(id="e-ds", source="dataset", target="trainer", sourceHandle="dataset", targetHandle="dataset"),
        Edge(id="e-m", source="model", target="trainer", sourceHandle="model", targetHandle="model"),
        Edge(id="e-o", source="optimizer", target="trainer", sourceHandle="optimizer", targetHandle="optimizer"),
        Edge(id="e-l", source="loss", target="trainer", sourceHandle="loss", targetHandle="loss"),
    ]
    return nodes, edges


def _cbs_graph() -> tuple[list[Node], list[Edge]]:
    nodes, edges = _base_graph({**_TRAINER_EPOCHS, "batchSize": -1})
    nodes.append(
        Node(
            id="cbs",
            type=NodeKind.cyclic_batch_schedule,
            data={
                "batchMin": 128,
                "batchMax": 640,
                "cycleLengthEpochs": 10,
                "refBatchSize": 128,
                "cycleLengthSteps": 0,
                "scheduleMode": "discrete_epoch",
            },
        )
    )
    edges.append(
        Edge(id="e-cbs", source="cbs", target="trainer", sourceHandle="batch_schedule", targetHandle="batch_schedule")
    )
    return nodes, edges


def _clr_graph() -> tuple[list[Node], list[Edge]]:
    nodes, edges = _base_graph({**_TRAINER_EPOCHS, "batchSize": 128})
    nodes.append(
        Node(
            id="clr",
            type=NodeKind.cyclic_lr_schedule,
            data={
                "lrMin": 0.001,
                "lrMax": 0.005,
                "cycleLengthEpochs": 10,
                "refBatchSize": 128,
                "cycleLengthSteps": 0,
                "scheduleMode": "discrete_epoch",
            },
        )
    )
    edges.append(
        Edge(id="e-clr", source="clr", target="optimizer", sourceHandle="lr_schedule", targetHandle="lr_schedule")
    )
    return nodes, edges


def test_fig1_cyclic_validate_only_compiles() -> None:
    cbs_nodes, cbs_edges = _cbs_graph()
    clr_nodes, clr_edges = _clr_graph()
    with patch(
        "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"
    ) as mocked:
        ctx_cbs = prepare_trainer_run(cbs_nodes, cbs_edges, "trainer", validate_only=True)
        ctx_clr = prepare_trainer_run(clr_nodes, clr_edges, "trainer", validate_only=True)
        mocked.assert_not_called()
    # CBS (Fig 1 left): batch cycles 128..640, LR fixed.
    assert ctx_cbs.cyclic_batch_cycle_steps > 0
    assert ctx_cbs.cyclic_lr_cycle_steps == 0
    assert ctx_cbs.train_batch_size == 128  # seeded from batchMin
    assert ctx_cbs.training_data_epochs == 300
    assert ctx_cbs.training_steps == 48720
    # CLR (Fig 1 left): LR cycles 0.001..0.005, batch fixed at 128.
    assert ctx_clr.cyclic_lr_cycle_steps > 0
    assert ctx_clr.cyclic_batch_cycle_steps == 0
    assert ctx_clr.training_data_epochs == 300
    assert ctx_clr.training_steps == 117300
