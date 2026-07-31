from __future__ import annotations

import numpy as np
import torch

from comfy_research.engine.datasets.vision_datasets_runtime import _cifar10_subset_indices, denormalize_cifar10_images
from comfy_research.engine.models.diffusion_score_model import diffusion_noise_mse_eval_mean, diffusion_noise_mse_loss
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node
from comfy_research.schemas.graph import Node, NodeKind


def test_unet_ddpm_builder_preserves_nchw_shape() -> None:
    node = Node(
        id="unet",
        type=NodeKind.unet_ddpm_model,
        data={"inChannels": 3, "baseChannels": 8, "channelMult": "1,2", "timeEmbedDim": 16, "diffusionTimesteps": 8, "imageSize": 8},
    )
    model = build_model_for_node(node, ModelBuildContext(input_channels=3, image_size=8))
    output = model(torch.zeros(2, 3, 8, 8), torch.tensor([0, 7]))
    assert tuple(output.shape) == (2, 3, 8, 8)
    assert model.max_timesteps == 8


def test_diffusion_loss_supports_unet_nchw_batches() -> None:
    node = Node(
        id="unet",
        type=NodeKind.unet_ddpm_model,
        data={"inChannels": 3, "baseChannels": 8, "channelMult": "1,2", "timeEmbedDim": 16, "diffusionTimesteps": 8, "imageSize": 8},
    )
    model = build_model_for_node(node, ModelBuildContext(input_channels=3, image_size=8))
    images = torch.randn(2, 3, 8, 8)

    loss = diffusion_noise_mse_loss(model, images, torch.Generator().manual_seed(0), timesteps=8)
    eval_loss = diffusion_noise_mse_eval_mean(model, images, timesteps=8, num_noise_draws=1)

    assert torch.isfinite(loss)
    assert eval_loss > 0


def test_cifar_balanced_subset_and_denormalization_are_deterministic() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    first = _cifar10_subset_indices(labels, 20, seed=7, class_balanced=True)
    second = _cifar10_subset_indices(labels, 20, seed=7, class_balanced=True)
    assert np.array_equal(first, second)
    assert np.bincount(labels[first], minlength=10).tolist() == [2] * 10
    assert np.allclose(denormalize_cifar10_images(np.array([[[[-1.0, 1.0]]]], dtype=np.float32), "minus_one_to_one"), [[[[0.0, 1.0]]]])
