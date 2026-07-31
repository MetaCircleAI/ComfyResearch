"""Trainer-level trainSeed keeps a fixed dataset and initialization intact."""
from __future__ import annotations

import torch

from comfy_research.engine.runs.trainer_run import prepare_trainer_run
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _graph(train_seed: int) -> tuple[list[Node], list[Edge]]:
    nodes = [
        Node(id="dataset", type=NodeKind.linear_dataset, data={"inputDim": 4, "outputDim": 2, "trainSize": 16, "testSize": 8, "seed": 7}),
        Node(id="model", type=NodeKind.mlp_model, data={"inputDim": 4, "outputDim": 2, "depth": 1, "width": 8, "seed": 0}),
        Node(id="optimizer", type=NodeKind.adam_optimizer, data={"learningRate": 0.001}),
        Node(id="loss", type=NodeKind.mse_loss, data={}),
        Node(id="trainer", type=NodeKind.trainer, data={"trainingSteps": 2, "batchSize": 4, "trainSeed": train_seed}),
    ]
    edges = [
        Edge(id="dataset-edge", source="dataset", sourceHandle="dataset", target="trainer", targetHandle="dataset"),
        Edge(id="model-edge", source="model", sourceHandle="model", target="trainer", targetHandle="model"),
        Edge(id="optimizer-edge", source="optimizer", sourceHandle="optimizer", target="trainer", targetHandle="optimizer"),
        Edge(id="loss-edge", source="loss", sourceHandle="loss", target="trainer", targetHandle="loss"),
    ]
    return nodes, edges


def test_train_seed_changes_training_rng_not_dataset_or_initialization() -> None:
    left = prepare_trainer_run(*_graph(0), "trainer")
    right = prepare_trainer_run(*_graph(1), "trainer")

    torch.testing.assert_close(left.x_t, right.x_t)
    torch.testing.assert_close(left.y_t, right.y_t)
    for left_param, right_param in zip(left.model.parameters(), right.model.parameters(), strict=True):
        torch.testing.assert_close(left_param, right_param)
    assert left.minibatch_perm_seed != right.minibatch_perm_seed
    assert left.run_seed == 0
    assert right.run_seed == 1


def test_legacy_run_seed_defaults_to_model_seed() -> None:
    nodes, edges = _graph(-1)
    trainer = next(node for node in nodes if node.id == "trainer")
    trainer.data = {key: value for key, value in (trainer.data or {}).items() if key != "trainSeed"}

    context = prepare_trainer_run(nodes, edges, "trainer")

    assert context.run_seed == 0
