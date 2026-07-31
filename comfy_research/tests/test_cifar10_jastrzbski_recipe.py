from __future__ import annotations

import numpy as np
import torch

from comfy_research.engine.datasets.vision_datasets_runtime import (
    cifar10_jastrzbski_split_indices,
    cifar10_pixel_mean_global_std,
)
from comfy_research.engine.trainer.dataset_helpers import (
    _cifar10_crop_flip_standardize,
    _take_epoch_shuffled_minibatch,
)
from comfy_research.engine.trainer.dataset_materialize import _materialize_vision
from comfy_research.schemas.graph import Node, NodeKind


def test_jastrzbski_split_uses_legacy_random_state_permutation() -> None:
    train, held_out = cifar10_jastrzbski_split_indices(10, 6, seed=777)
    expected = np.random.RandomState(777).permutation(10)
    assert train.tolist() == expected[:6].tolist()
    assert held_out.tolist() == expected[6:].tolist()


def test_jastrzbski_statistics_are_pixel_mean_and_one_global_std() -> None:
    x = np.array(
        [
            [[[0.0, 1.0], [0.5, 0.25]]],
            [[[1.0, 0.0], [0.5, 0.75]]],
        ],
        dtype=np.float32,
    )
    mean, std = cifar10_pixel_mean_global_std(x)
    assert np.allclose(mean, x.astype(np.float64).mean(axis=0))
    expected = np.sqrt(np.square(x.astype(np.float64) - mean).mean())
    assert np.isclose(std, expected)


def test_epoch_shuffle_visits_each_example_once_and_changes_by_epoch() -> None:
    x = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    y = torch.arange(10)

    def epoch_values(epoch: int) -> list[int]:
        values: list[int] = []
        for step_in_epoch in range(3):
            xb, _ = _take_epoch_shuffled_minibatch(
                x,
                y,
                4,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                run_seed=3,
            )
            values.extend(int(v) for v in xb[:, 0])
        return values

    first = epoch_values(0)
    second = epoch_values(1)
    assert sorted(first) == list(range(10))
    assert sorted(second) == list(range(10))
    assert first != second


def test_cifar_crop_flip_standardize_is_seeded_and_shape_preserving() -> None:
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 100
    mean = torch.zeros((3, 4, 4))
    g1 = torch.Generator().manual_seed(9)
    g2 = torch.Generator().manual_seed(9)
    a = _cifar10_crop_flip_standardize(x, mean, 2.0, generator=g1, padding=1)
    b = _cifar10_crop_flip_standardize(x, mean, 2.0, generator=g2, padding=1)
    assert a.shape == x.shape
    assert torch.equal(a, b)


def test_vision_materializer_keeps_train_raw_and_standardizes_test(monkeypatch) -> None:
    train = np.array([[[[0.0, 1.0]]], [[[1.0, 0.0]]]], dtype=np.float32)
    test = np.array([[[[0.5, 0.5]]]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    monkeypatch.setattr(
        "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays",
        lambda *args: (train.copy(), labels.copy(), test.copy(), labels[:1].copy()),
    )
    monkeypatch.setattr(
        "comfy_research.engine.trainer.dataset_materialize.infer_vision_input_channels_height_width",
        lambda *args: (1, 1, 2),
    )
    arrays = _materialize_vision(
        Node(id="ds", type=NodeKind.cifar10_dataset, data={}),
        {"trainingRecipe": "jastrzbski_fig1"},
        2,
        1,
        np.random.default_rng(0),
    )

    assert np.array_equal(arrays.x_np, train)
    assert arrays.extras["training_recipe"] == "jastrzbski_fig1"
    assert np.allclose(arrays.x_test_np, 0.0)
