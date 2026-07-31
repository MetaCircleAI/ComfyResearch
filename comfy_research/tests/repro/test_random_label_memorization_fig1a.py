"""Baseline and graph-level smoke tests for the random-label memorization template."""

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
    entry, nodes, edges = load_template("repro-random-label-memorization-fig1a")
    assert entry.name == "repro: Zhang et al. Figure 1(a) random-label memorization"

    datasets = nodes_by_type(nodes, "cifar10_dataset")
    trainers = nodes_by_type(nodes, "trainer")
    model = node_by_type(nodes, "small_inception_cifar_model")
    optimizer = node_by_type(nodes, "sgd_optimizer")
    schedule = node_by_type(nodes, "lr_schedule")
    assert len(datasets) == len(trainers) == 5
    assert model.data["seed"] == 1706
    assert optimizer.data["learningRate"] == pytest.approx(0.1)
    assert optimizer.data["momentum"] == pytest.approx(0.9)
    assert optimizer.data["weightDecay"] == 0
    assert schedule.data["lrSchedule"] == "exponential_epoch"
    assert schedule.data["exponentialDecayFactor"] == pytest.approx(0.95)
    assert {node.data["inputTransform"] for node in datasets} == {
        "none",
        "shuffled_pixels",
        "random_pixels",
        "gaussian",
    }
    assert sorted(node.data["labelCorruption"] for node in datasets) == [0, 0, 0, 0, 1]
    assert all(node.data["trainingSteps"] == 25_000 for node in trainers)
    assert all(node.data["logFrequency"] == 100 for node in trainers)
    assert all(node.data["logAggregation"] == "interval_sample_mean" for node in trainers)
    assert all(node.data["testEvaluation"] == "final_only" for node in trainers)
    assert all(node.data["minibatchSampling"] == "epoch_shuffle" for node in trainers)
    assert all(node.data["minibatchSeed"] == 1707 for node in trainers)
    assert all(node.data["remoteGpu"] is True for node in trainers)
    assert all(node.data["subsetSeed"] == 1703 for node in datasets)
    assert all(node.data["classBalanced"] is False for node in datasets)
    assert all(
        has_edge(edges, model.id, trainer.id, "model", "model") for trainer in trainers
    )
    assert has_edge(edges, schedule.id, optimizer.id, "lr_schedule", "lr_schedule")


def test_template_smoke_run() -> None:
    _entry, nodes, edges = load_template("repro-random-label-memorization-fig1a")
    run_reduced_cpu_smoke(nodes, edges)
