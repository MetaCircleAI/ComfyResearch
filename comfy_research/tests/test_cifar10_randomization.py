from __future__ import annotations

import numpy as np
import pytest

import comfy_research.engine.datasets.vision_datasets_runtime as runtime

from comfy_research.engine.datasets.vision_datasets_runtime import (
    _apply_cifar10_preprocessing,
    _apply_cifar10_randomization,
)
from comfy_research.engine.models.vision_models import infer_vision_input_channels_height_width
from comfy_research.schemas.graph import NodeKind


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.arange(8 * 2 * 2 * 3, dtype=np.float32).reshape(8, 2, 2, 3) / 100.0
    y_train = np.arange(8, dtype=np.int64)
    x_test = x_train[:2].copy()
    y_test = y_train[:2].copy()
    return x_train, y_train, x_test, y_test


def test_full_label_corruption_is_seeded_and_keeps_test_labels() -> None:
    arrays = _arrays()
    first = _apply_cifar10_randomization(
        *arrays,
        input_transform="none",
        label_corruption=1.0,
        rng=np.random.default_rng(17),
    )
    second = _apply_cifar10_randomization(
        *arrays,
        input_transform="none",
        label_corruption=1.0,
        rng=np.random.default_rng(17),
    )

    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[3], arrays[3])
    assert not np.array_equal(first[1], arrays[1])


def test_shared_pixel_shuffle_uses_one_permutation_for_every_example() -> None:
    arrays = _arrays()
    x_train, _, x_test, _ = _apply_cifar10_randomization(
        *arrays,
        input_transform="shuffled_pixels",
        label_corruption=0.0,
        rng=np.random.default_rng(3),
    )

    assert x_test is not None
    np.testing.assert_array_equal(x_train[:2], x_test)
    np.testing.assert_array_equal(np.sort(x_train.reshape(8, -1), axis=1), np.sort(arrays[0].reshape(8, -1), axis=1))


def test_random_pixels_and_gaussian_preserve_shapes_and_dtype() -> None:
    arrays = _arrays()
    for mode in ("random_pixels", "gaussian"):
        x_train, y_train, x_test, y_test = _apply_cifar10_randomization(
            *arrays,
            input_transform=mode,
            label_corruption=0.0,
            rng=np.random.default_rng(5),
        )
        assert x_train.shape == arrays[0].shape
        assert x_train.dtype == np.float32
        assert x_test is not None and x_test.shape == arrays[2].shape
        np.testing.assert_array_equal(y_train, arrays[1])
        np.testing.assert_array_equal(y_test, arrays[3])


def test_paper_preprocessing_center_crops_and_whitens_each_image() -> None:
    x = np.arange(2 * 3 * 32 * 32, dtype=np.float32).reshape(2, 3, 32, 32) / 255.0
    processed = _apply_cifar10_preprocessing(x, "center_crop_28_per_image_whiten")

    assert processed.shape == (2, 3, 28, 28)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(processed.mean(axis=(1, 2, 3)), 0.0, atol=2e-6)
    np.testing.assert_allclose(processed.std(axis=(1, 2, 3)), 1.0, atol=2e-6)


def test_paper_preprocessing_uses_tensorflow_adjusted_stddev_floor() -> None:
    constant = np.full((1, 3, 32, 32), 0.25, dtype=np.float32)
    processed = _apply_cifar10_preprocessing(
        constant, "center_crop_28_per_image_whiten"
    )
    np.testing.assert_array_equal(processed, np.zeros((1, 3, 28, 28), dtype=np.float32))


def test_cifar10_input_dims_follow_preprocessing_output_shape() -> None:
    assert infer_vision_input_channels_height_width(
        NodeKind.cifar10_dataset, {"preprocessing": "none"}
    ) == (3, 32, 32)
    assert infer_vision_input_channels_height_width(
        NodeKind.cifar10_dataset,
        {"preprocessing": "center_crop_28_per_image_whiten"},
    ) == (3, 28, 28)


def test_build_pipeline_rejects_diffusion_scaling_after_paper_whitening(monkeypatch) -> None:
    arrays = (
        np.zeros((2, 3, 32, 32), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        None,
        None,
    )
    monkeypatch.setattr(runtime, "_load_cifar10_official", lambda **_kwargs: arrays)

    with pytest.raises(runtime.HTTPException, match="per-image whitening"):
        runtime.build_cifar10_arrays(
            {
                "trainingRecipe": "jastrzbski_fig1",
                "preprocessing": "center_crop_28_per_image_whiten",
                "normalize": "minus_one_to_one",
            },
            train_n=2,
            test_n=0,
            rng=np.random.default_rng(99),
        )


def test_build_pipeline_uses_init_seed_for_randomization_not_loader_rng(monkeypatch) -> None:
    source = np.arange(2 * 3 * 32 * 32, dtype=np.float32).reshape(2, 3, 32, 32) / 8192.0
    labels = np.array([1, 2], dtype=np.int64)

    def load(**kwargs):
        # The loader's sampling may consume a varying amount of caller RNG state.
        kwargs["rng"].random(31)
        return source.copy(), labels.copy(), None, None

    monkeypatch.setattr(runtime, "_load_cifar10_official", load)
    data = {
        "trainingRecipe": "jastrzbski_fig1",
        "initSeed": 7,
        "inputTransform": "gaussian",
        "normalize": "minus_one_to_one",
    }
    first = runtime.build_cifar10_arrays(data, 2, 0, np.random.default_rng(1))
    second = runtime.build_cifar10_arrays(data, 2, 0, np.random.default_rng(2))

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], labels)
    assert first[0].dtype == np.float32
    assert float(first[0].min()) >= -1.0
    assert float(first[0].max()) <= 1.0
