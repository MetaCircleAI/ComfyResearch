"""
Template:
    repro: Jastrzębski Fig 1 cyclic CBS vs CLR

Purpose:
    Compare cyclic batch size and cyclic learning rate using the template's
    VGG-11 + BatchNorm CIFAR-10 setup.
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
    entry, nodes, edges = load_template("repro-jastrzbski-fig1-vgg11")
    assert entry.name == "repro: Jastrzębski Fig 1 cyclic CBS vs CLR"

    dataset = node_by_type(nodes, "cifar10_dataset")
    model = node_by_type(nodes, "vgg11_cifar_model")
    cbs = node_by_type(nodes, "cyclic_batch_schedule")
    clr = node_by_type(nodes, "cyclic_lr_schedule")
    assert dataset.data["trainSize"] == 45000
    assert dataset.data["testSize"] == 10000
    assert dataset.data["initSeed"] == 777
    assert dataset.data["trainingRecipe"] == "jastrzbski_fig1"
    assert model.data["seed"] == [0, 1, 2, 3, 4]
    assert cbs.data["batchMin"] == 128
    assert cbs.data["batchMax"] == 640
    assert cbs.data["cycleLengthEpochs"] == 10
    assert cbs.data["scheduleMode"] == "square_epoch"
    assert clr.data["lrMin"] == pytest.approx(0.001)
    assert clr.data["lrMax"] == pytest.approx(0.005)
    assert clr.data["cycleLengthEpochs"] == 10
    assert clr.data["scheduleMode"] == "square_epoch"

    trainers = nodes_by_type(nodes, "trainer")
    assert len(trainers) == 2
    trainer_cbs = next(node for node in trainers if "CBS" in node.data["instanceTitle"])
    trainer_clr = next(node for node in trainers if "CLR" in node.data["instanceTitle"])
    assert trainer_cbs.data["trainingSteps"] == 63450
    assert trainer_clr.data["trainingSteps"] == 105600
    assert all(node.data["trainingEpochs"] == 300 for node in trainers)
    assert has_edge(edges, cbs.id, trainer_cbs.id, "batch_schedule", "batch_schedule")
    clr_optimizer = next(
        node for node in nodes_by_type(nodes, "sgd_optimizer") if "CLR" in node.data["instanceTitle"]
    )
    assert has_edge(edges, clr.id, clr_optimizer.id, "lr_schedule", "lr_schedule")
    assert all(has_edge(edges, dataset.id, node.id, "dataset", "dataset") for node in trainers)
    assert all(has_edge(edges, model.id, node.id, "model", "model") for node in trainers)


def test_template_smoke_run() -> None:
    """Reduced CBS and CLR stacks complete on CPU without downloading CIFAR-10."""
    _entry, nodes, edges = load_template("repro-jastrzbski-fig1-vgg11")
    run_reduced_cpu_smoke(nodes, edges)
