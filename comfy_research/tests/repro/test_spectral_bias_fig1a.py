"""Baseline and graph-level smoke tests for the Spectral Bias Figure 1(a) template."""

from __future__ import annotations

import numpy as np
import pytest

from comfy_research.engine.reproductions.spectral_bias import (
    SpectralBiasProtocol,
    train_spectral_bias_repeat,
)
from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.tests.repro.template_test_helpers import (
    has_edge,
    load_template,
    node_by_type,
    nodes_by_type,
    run_reduced_cpu_smoke,
)


pytestmark = pytest.mark.repro


def test_template_baseline() -> None:
    entry, nodes, edges = load_template("repro-spectral-bias-fig1a")
    assert entry.name == "repro: Rahaman Spectral Bias Figure 1(a)"

    dataset = node_by_type(nodes, "symbolic_func_dataset")
    model = node_by_type(nodes, "mlp_model")
    optimizer = node_by_type(nodes, "adam_optimizer")
    trainer = node_by_type(nodes, "trainer")
    components = nodes_by_type(nodes, "observable_fourier_component")
    layer_norm = node_by_type(nodes, "observable_layer_spectral_norm")

    assert dataset.data["trainSize"] == 200
    assert dataset.data["inputDistribution"] == "linspace_0_1_endpoint_excluded"
    assert dataset.data["evaluationPrecision"] == "float64"
    assert model.data["depth"] == 5
    assert model.data["width"] == 256
    assert optimizer.data["learningRate"] == pytest.approx(3e-4)
    assert trainer.data["trainingSteps"] == 80_000
    assert trainer.data["batchSize"] == -1
    assert trainer.data["logTiming"] == "pre_update"
    assert trainer.data["remoteGpu"] is True
    assert {node.data["frequency"] for node in components} == set(range(5, 51, 5))
    assert layer_norm.data["estimator"] == "author_figure1"
    assert layer_norm.data["startVector"] == "seeded_gaussian"
    assert layer_norm.data["seed"] == 3703
    assert all(
        has_edge(edges, node.id, trainer.id, "observables", "observables")
        for node in [*components, layer_norm]
    )


def test_template_smoke_run() -> None:
    _entry, nodes, edges = load_template("repro-spectral-bias-fig1a")
    run_reduced_cpu_smoke(nodes, edges)


def test_short_single_phase_run_matches_formal_engine() -> None:
    _entry, nodes, edges = load_template("repro-spectral-bias-fig1a")
    for node in nodes:
        if node.id == "sbf-model":
            node.data["width"] = 16
        elif node.id == "sbf-trainer":
            node.data.update(
                {
                    "trainingSteps": 6,
                    "logFrequency": 5,
                    "computeDevice": "cpu",
                    "remoteGpu": False,
                }
            )
    complete = next(
        event
        for event in iter_trainer_events(nodes, edges, "sbf-trainer")
        if event["type"] == "complete"
    )
    protocol = SpectralBiasProtocol(
        repeats=1,
        iterations=6,
        record_every=5,
        width=16,
    )
    formal, _model = train_spectral_bias_repeat(
        protocol,
        repeat_index=0,
        device="cpu",
    )
    assert complete["step_ticks"] == formal.iterations.tolist()
    np.testing.assert_array_equal(complete["loss_history"], formal.train_loss)
    for index, frequency in enumerate(protocol.frequencies):
        np.testing.assert_allclose(
            complete["observable_metric_histories"][f"sbf-f{frequency}"],
            formal.normalized_amplitudes[:, index],
            atol=2e-8,
            rtol=0,
        )
    graph_layer_norms = np.stack(
        [
            complete["observable_metric_histories"][
                f"sbf-layer-norm::layer::{index}"
            ]
            for index in range(1, 7)
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        graph_layer_norms,
        formal.author_layer_norms,
    )
