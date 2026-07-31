"""Baseline and graph-level smoke tests for the Information Bottleneck Figure 3 template."""

from __future__ import annotations

import pytest

from comfy_research.tests.repro.template_test_helpers import (
    has_edge,
    load_template,
    nodes_by_type,
    run_reduced_cpu_smoke,
)


pytestmark = pytest.mark.repro

_TEMPLATE_SLUG = "dafb8339-a932-4b10-b3b6-185fc53a5a4f"


def test_template_baseline() -> None:
    entry, nodes, edges = load_template(_TEMPLATE_SLUG)
    assert entry.name == "repro: Information-plane interactive approximation (Figure 3 protocol)"
    assert entry.tier == "small"

    datasets = nodes_by_type(nodes, "information_bottleneck_dataset")
    trainers = nodes_by_type(nodes, "trainer")
    assert len(datasets) == len(trainers) == 3
    assert {node.data["trainSize"] for node in datasets} == {205, 1843, 3482}
    assert all(node.data["rule"] == "verified_local_var_u" for node in datasets)
    assert len(nodes_by_type(nodes, "linear_layer")) == 18
    assert len(nodes_by_type(nodes, "activation_layer")) == 15
    assert len(nodes_by_type(nodes, "observable_information_plane")) == 3
    assert len(nodes_by_type(nodes, "binary_cross_entropy_with_logits_loss")) == 3
    assert len(nodes_by_type(nodes, "idnns_initialization")) == 3
    assert all(node.data["trainingSteps"] == 10_000 for node in trainers)
    assert all(node.data["trainingEpochs"] == 10_000 for node in trainers)
    assert all(node.data["logSchedule"] == "idnns_logspace" for node in trainers)
    assert all(node.data["logSamples"] == 1800 for node in trainers)
    assert all(node.data["minibatchSampling"] == "affine_epoch" for node in trainers)
    assert all(node.data["computeDevice"] == "cuda" and node.data["remoteGpu"] for node in trainers)
    assert {node.id: node.data["batchSize"] for node in trainers} == {
        "ib-05-trainer": 205,
        "ib-45-trainer": 256,
        "ib-85-trainer": 256,
    }

    for key, seed in (("05", 1708), ("45", 1748), ("85", 1788)):
        panel_linears = sorted(
            (
                node
                for node in nodes_by_type(nodes, "linear_layer")
                if node.id.startswith(f"ib-{key}-")
            ),
            key=lambda node: node.id,
        )
        assert [
            (node.data["inFeatures"], node.data["outFeatures"])
            for node in panel_linears
        ] == [(12, 10), (10, 8), (8, 6), (6, 4), (4, 2), (2, 1)]
        observable = next(node for node in nodes if node.id == f"ib-{key}-observable")
        assert observable.data["binning"] == "idnns_equal_points"
        assert observable.data["outputMapping"] == "probability"
        initialization = next(node for node in nodes if node.id == f"ib-{key}-initialization")
        assert initialization.data["seed"] == seed
        assert has_edge(
            edges,
            f"ib-{key}-dataset",
            f"ib-{key}-trainer",
            "dataset",
            "dataset",
        )
        assert has_edge(
            edges,
            f"ib-{key}-observable",
            f"ib-{key}-trainer",
            "observables",
            "observables",
        )
        assert has_edge(
            edges,
            f"ib-{key}-initialization",
            f"ib-{key}-linear-6",
            "initialization",
            "initialization",
        )


def test_template_smoke_run() -> None:
    _entry, nodes, edges = load_template(_TEMPLATE_SLUG)
    run_reduced_cpu_smoke(nodes, edges)
