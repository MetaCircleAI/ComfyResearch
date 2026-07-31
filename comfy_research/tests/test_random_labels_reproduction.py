from __future__ import annotations

import numpy as np
import pytest

from comfy_research.engine.reproductions.random_labels import (
    CONDITION_ORDER,
    RandomLabelsProtocol,
    condition_transform,
    epoch_permutation,
    learning_rate_for_step,
    numpy_sha256,
    prepare_condition_arrays,
    steps_per_epoch,
    train_random_label_condition,
)


def test_condition_mapping_matches_figure1a() -> None:
    assert CONDITION_ORDER == (
        "true_labels",
        "random_labels",
        "shuffled_pixels",
        "random_pixels",
        "gaussian",
    )
    assert condition_transform("true_labels") == ("none", 0.0)
    assert condition_transform("random_labels") == ("none", 1.0)
    assert condition_transform("shuffled_pixels") == ("shuffled_pixels", 0.0)
    assert condition_transform("random_pixels") == ("random_pixels", 0.0)
    assert condition_transform("gaussian") == ("gaussian", 0.0)


def test_epoch_shuffle_is_deterministic_and_covers_each_example() -> None:
    first = epoch_permutation(23, seed=17, epoch=4)
    second = epoch_permutation(23, seed=17, epoch=4)
    other = epoch_permutation(23, seed=17, epoch=5)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(23))
    assert not np.array_equal(first, other)


def test_paper_lr_decay_uses_complete_data_epochs() -> None:
    protocol = RandomLabelsProtocol(condition="true_labels", train_size=50_000, batch_size=128)
    epoch_steps = steps_per_epoch(50_000, 128)
    assert epoch_steps == 391
    assert learning_rate_for_step(protocol, epoch_steps - 1) == pytest.approx(0.1)
    assert learning_rate_for_step(protocol, epoch_steps) == pytest.approx(0.095)
    assert learning_rate_for_step(protocol, 2 * epoch_steps) == pytest.approx(0.1 * 0.95**2)


def test_random_labels_are_fixed_and_fingerprinted() -> None:
    rng = np.random.default_rng(9)
    x_train = rng.random((12, 3, 32, 32), dtype=np.float32)
    y_train = np.arange(12, dtype=np.int64) % 10
    x_test = rng.random((4, 3, 32, 32), dtype=np.float32)
    y_test = np.arange(4, dtype=np.int64)
    protocol = RandomLabelsProtocol(
        condition="random_labels",
        train_size=12,
        test_size=4,
        steps=2,
        transform_seed=23,
    )
    first = prepare_condition_arrays((x_train, y_train, x_test, y_test), protocol)
    second = prepare_condition_arrays((x_train, y_train, x_test, y_test), protocol)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert numpy_sha256(first[0]) == numpy_sha256(second[0])
    assert first[0].shape == (12, 3, 28, 28)
    np.testing.assert_array_equal(first[3], y_test)



def test_training_enables_and_restores_deterministic_algorithms() -> None:
    import torch

    before = torch.are_deterministic_algorithms_enabled()
    rng = np.random.default_rng(23)
    arrays = (
        rng.normal(size=(2, 3, 28, 28)).astype(np.float32),
        np.array([0, 1], dtype=np.int64),
        None,
        None,
    )
    protocol = RandomLabelsProtocol(
        condition="true_labels",
        train_size=2,
        test_size=0,
        steps=1,
        batch_size=2,
        log_every=1,
        full_eval_every=1,
        eval_batch_size=2,
        deterministic_algorithms=True,
    )
    result, _model = train_random_label_condition(protocol, arrays, device=torch.device("cpu"))
    assert result.step_ticks.tolist() == [1]
    assert torch.are_deterministic_algorithms_enabled() is before
