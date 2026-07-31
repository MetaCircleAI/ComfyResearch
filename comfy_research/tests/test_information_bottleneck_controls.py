from __future__ import annotations

from comfy_research.engine.reproductions.information_bottleneck_controls import (
    SUPPLEMENTAL_CONTROLS,
    supplemental_control_map,
)


def test_supplemental_matrix_covers_declared_method_controls() -> None:
    controls = supplemental_control_map()
    assert len(controls) == len(SUPPLEMENTAL_CONTROLS) == 6
    groups = {control.comparison_group for control in controls.values()}
    assert groups == {"activation", "binning_resolution", "optimizer_noise"}
    assert {control.activation for control in controls.values()} >= {"tanh", "relu"}
    assert {control.bins for control in controls.values()} >= {30, 100}


def test_optimizer_noise_pair_changes_only_batch_size() -> None:
    controls = supplemental_control_map()
    mini = controls["noise_tanh_adaptive30_sgd_minibatch"]
    full = controls["noise_tanh_adaptive30_sgd_fullbatch"]
    assert mini.optimizer == full.optimizer == "sgd"
    assert mini.learning_rate == full.learning_rate == 1e-2
    assert mini.activation == full.activation == "tanh"
    assert mini.binning == full.binning == "adaptive_minmax"
    assert mini.bins == full.bins == 30
    assert mini.batch_size == 256
    assert full.batch_size == 3277
