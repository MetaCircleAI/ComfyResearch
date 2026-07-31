from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from collections import defaultdict
from types import SimpleNamespace

from comfy_research.engine.reproductions.spectral_bias import (
    PaperSpectralBiasMlp,
    SpectralBiasProtocol,
    author_power_iteration_estimate,
    make_phase_target,
    normalized_dft_amplitudes,
    singular_norm_power_estimate,
    train_spectral_bias_repeat,
)
from comfy_research.nodes.definitions.observables.layer_spectral_norm import (
    record as record_layer_spectral_norm,
)


def test_phase_target_is_reproducible_and_has_exact_target_bins() -> None:
    protocol = SpectralBiasProtocol(iterations=2, repeats=1)
    x_a, target_a, phases_a = make_phase_target(protocol, phase_seed=1703)
    x_b, target_b, phases_b = make_phase_target(protocol, phase_seed=1703)
    np.testing.assert_array_equal(x_a, x_b)
    np.testing.assert_array_equal(target_a, target_b)
    np.testing.assert_array_equal(phases_a, phases_b)
    assert x_a[0] == 0.0
    assert x_a[-1] == pytest.approx(199 / 200)
    normalized = normalized_dft_amplitudes(
        target_a,
        frequencies=protocol.frequencies,
        amplitudes=protocol.amplitudes,
    )
    np.testing.assert_allclose(normalized, np.ones(10), atol=2e-7, rtol=0)


def test_paper_model_depth_means_total_linear_layers() -> None:
    model = PaperSpectralBiasMlp(width=8, linear_layers=6, seed=4)
    linears = [module for module in model if isinstance(module, nn.Linear)]
    relus = [module for module in model if isinstance(module, nn.ReLU)]
    assert len(linears) == 6
    assert len(relus) == 5
    assert linears[0].in_features == 1
    assert linears[-1].out_features == 1


def test_author_estimator_preserves_released_notebook_scale_quirk() -> None:
    width = 7
    value = 2.5
    matrix = torch.eye(width, dtype=torch.float64) * value
    author_generator = torch.Generator().manual_seed(1)
    singular_generator = torch.Generator().manual_seed(1)
    author = author_power_iteration_estimate(
        matrix, iterations=3, generator=author_generator
    )
    conventional = singular_norm_power_estimate(
        matrix, iterations=3, generator=singular_generator
    )
    assert float(author) == pytest.approx(width * value, rel=1e-12)
    assert float(conventional) == pytest.approx(value, rel=1e-12)


def test_small_repeat_shapes_and_resume_match_uninterrupted() -> None:
    full_protocol = SpectralBiasProtocol(
        samples=120,
        frequencies=(5, 10),
        amplitudes=(1.0, 1.0),
        repeats=1,
        iterations=4,
        record_every=1,
        width=8,
        linear_layers=3,
        power_iterations=2,
    )
    full, full_model = train_spectral_bias_repeat(
        full_protocol, repeat_index=0, device="cpu"
    )

    prefix_protocol = SpectralBiasProtocol(
        samples=120,
        frequencies=(5, 10),
        amplitudes=(1.0, 1.0),
        repeats=1,
        iterations=2,
        record_every=1,
        width=8,
        linear_layers=3,
        power_iterations=2,
    )
    captured: dict[str, object] = {}

    def checkpoint(
        completed: int,
        model: PaperSpectralBiasMlp,
        optimizer: torch.optim.Optimizer,
        recorded: dict[str, list[np.ndarray] | list[float] | list[int]],
        author_generator: torch.Generator,
        singular_generator: torch.Generator,
    ) -> None:
        captured.update(
            {
                "repeat_index": 0,
                "completed_iterations": completed,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "recorded": recorded,
                "author_generator_state": author_generator.get_state(),
                "singular_generator_state": singular_generator.get_state(),
            }
        )

    train_spectral_bias_repeat(
        prefix_protocol,
        repeat_index=0,
        device="cpu",
        checkpoint_every=2,
        checkpoint=checkpoint,
    )
    resumed, resumed_model = train_spectral_bias_repeat(
        full_protocol,
        repeat_index=0,
        device="cpu",
        resume_state=captured,
    )

    assert full.iterations.tolist() == [0, 1, 2, 3]
    assert full.normalized_amplitudes.shape == (4, 2)
    assert full.author_layer_norms.shape == (4, 3)
    assert full.singular_layer_norms.shape == (4, 3)
    assert full.prediction_iterations.tolist() == [4]
    np.testing.assert_array_equal(resumed.iterations, full.iterations)
    np.testing.assert_allclose(resumed.train_loss, full.train_loss, rtol=0, atol=0)
    np.testing.assert_allclose(
        resumed.normalized_amplitudes, full.normalized_amplitudes, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        resumed.author_layer_norms, full.author_layer_norms, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        resumed.singular_layer_norms, full.singular_layer_norms, rtol=0, atol=0
    )
    for expected, actual in zip(full_model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_layer_spectral_norm_observable_records_one_series_per_linear() -> None:
    model = PaperSpectralBiasMlp(width=8, linear_layers=3, seed=2)
    histories: defaultdict[str, list[float]] = defaultdict(list)
    recorder = SimpleNamespace(model=model, observable_metric_histories=histories)
    node = SimpleNamespace(
        id="norms",
        data={"estimator": "author_figure1", "powerIterations": 2},
    )
    record_layer_spectral_norm(recorder, node)
    record_layer_spectral_norm(recorder, node)
    assert len(histories["norms"]) == 2
    assert [len(histories[f"norms::layer::{index}"]) for index in range(1, 4)] == [2, 2, 2]
    assert np.isfinite(histories["norms::layer::1"]).all()
    assert np.isfinite(histories["norms::layer::2"]).all()
    assert np.isfinite(histories["norms::layer::3"]).all()
