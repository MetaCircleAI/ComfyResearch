"""
Template:
    repro: Keskar Fig 2+3 SB/LB

Purpose:
    Compare the template's small- and large-batch CIFAR-10 training stacks and
    feed their checkpoints into the Fig 3 parametric path sampler.
"""

from __future__ import annotations

import pytest

from comfy_research.tests.repro.template_test_helpers import (
    has_edge,
    load_template,
    node_by_type,
    nodes_by_type,
    run_reduced_cpu_smoke,
)


pytestmark = pytest.mark.repro


def test_template_baseline() -> None:
    entry, nodes, edges = load_template("repro-keskar-fig23-sb-lb")
    assert entry.name == "repro: Keskar Fig 2+3 SB/LB"

    dataset = node_by_type(nodes, "cifar10_dataset")
    model = node_by_type(nodes, "keskar_c1_c2_cnn_model")
    optimizer = node_by_type(nodes, "adam_optimizer")
    loss = node_by_type(nodes, "cross_entropy_loss")
    sampler = node_by_type(nodes, "parametric_path_sampler")
    assert dataset.data["trainSize"] == 50000
    assert dataset.data["testSize"] == 10000
    assert model.data["architecture"] == "c1"
    assert model.data["seed"] == 0
    assert optimizer.data["learningRate"] == pytest.approx(0.001)
    assert loss.data["lossScale"] == 1
    assert sampler.data["alphaMin"] == -1
    assert sampler.data["alphaMax"] == 2
    assert sampler.data["alphaSteps"] == 25

    trainers = {int(node.data["batchSize"]): node for node in nodes_by_type(nodes, "trainer")}
    assert set(trainers) == {256, 5000}
    assert trainers[256].data["trainingSteps"] == 19600
    assert trainers[5000].data["trainingSteps"] == 1000
    assert all(node.data["trainingEpochs"] == 100 for node in trainers.values())
    assert all(has_edge(edges, dataset.id, node.id, "dataset", "dataset") for node in trainers.values())
    assert all(has_edge(edges, model.id, node.id, "model", "model") for node in trainers.values())
    assert has_edge(edges, dataset.id, sampler.id, "dataset", "dataset")
    assert has_edge(edges, model.id, sampler.id, "model", "model")

    checkpoints = nodes_by_type(nodes, "model_checkpoint")
    assert len(checkpoints) == 2
    assert all(
        any(has_edge(edges, trainer.id, checkpoint.id, "checkpoint", "model_checkpoint") for trainer in trainers.values())
        for checkpoint in checkpoints
    )
    assert {edge.targetHandle for edge in edges if edge.target == sampler.id and edge.source in {n.id for n in checkpoints}} == {
        "checkpoint_sb",
        "checkpoint_lb",
    }


def test_template_smoke_run() -> None:
    """Both reduced SB/LB stacks complete on CPU without downloading CIFAR-10."""
    _entry, nodes, edges = load_template("repro-keskar-fig23-sb-lb")
    run_reduced_cpu_smoke(nodes, edges)
