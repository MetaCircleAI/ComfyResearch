"""
Template:
    repro: The Emergence of Reproducibility and Generalizability in Diffusion Models

Purpose:
    Same-initialization / different-training-seed CIFAR-10 DDPM workflow with
    deterministic same-noise sampling and reproducibility observables.
"""

from __future__ import annotations

import base64
import io
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from comfy_research.api import diffusion_repro as diffusion_api
from comfy_research.engine.models.unet_ddpm_model import build_unet_ddpm_from_md
from comfy_research.engine.runs.trainer_run import iter_trainer_events
from comfy_research.tests.repro.template_test_helpers import (
    has_edge,
    load_template,
    node_by_type,
    nodes_by_type,
)


pytestmark = pytest.mark.repro

_VISION_BUILD = "comfy_research.engine.trainer.dataset_materialize.build_vision_numpy_arrays"


def _synthetic_cifar(
    _kind: object,
    _data: dict,
    train_size: int,
    test_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    x_train = rng.normal(size=(train_size, 3, 32, 32)).astype(np.float32)
    y_train = np.arange(train_size, dtype=np.int64) % 10
    x_test = rng.normal(size=(test_size, 3, 32, 32)).astype(np.float32)
    y_test = np.arange(test_size, dtype=np.int64) % 10
    return x_train, y_train, x_test, y_test


def _checkpoint_b64(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict()}, buffer)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _reduced_graph() -> tuple[list, list]:
    _entry, nodes, edges = load_template("repro-diffusion-same-init-different-seed")
    nodes = [node.model_copy(deep=True) for node in nodes]
    edges = [edge.model_copy(deep=True) for edge in edges]
    for node in nodes:
        data = node.data or {}
        if node.type == "cifar10_dataset":
            data.update(trainSize=8, testSize=4, imageSize=32, normalize="minus_one_to_one")
        elif node.type == "unet_ddpm_model":
            data.update(baseChannels=8, channelMult="1", timeEmbedDim=16, diffusionTimesteps=8, imageSize=32)
        elif node.type == "trainer":
            data.update(trainingLengthMode="steps", trainingSteps=1, logFrequency=1, batchSize=4, computeDevice="cpu", remoteGpu=False)
        elif node.type == "deterministic_diffusion_sampler":
            data.update(sampleCount=2, numSteps=2)
    return nodes, edges


def test_template_baseline() -> None:
    entry, nodes, edges = load_template("repro-diffusion-same-init-different-seed")
    assert entry.tier == "small"

    dataset = node_by_type(nodes, "cifar10_dataset")
    assert dataset.data["trainSize"] == 4096
    assert dataset.data["testSize"] == 128
    assert dataset.data["subsetSeed"] == 0

    models = nodes_by_type(nodes, "unet_ddpm_model")
    assert len(models) == 2
    assert {model.data["seed"] for model in models} == {0}
    assert all(model.data["baseChannels"] == 128 for model in models)

    trainers = sorted(nodes_by_type(nodes, "trainer"), key=lambda node: node.id)
    assert [trainer.data["seed"] for trainer in trainers] == [0, 1]
    assert all(trainer.data["trainingSteps"] == 5000 for trainer in trainers)
    assert all(trainer.data["batchSize"] == 256 for trainer in trainers)

    samplers = nodes_by_type(nodes, "deterministic_diffusion_sampler")
    assert len(samplers) == 2
    assert all(sampler.data["sampleCount"] == 64 for sampler in samplers)
    assert all(sampler.data["numSteps"] == 50 for sampler in samplers)

    assert node_by_type(nodes, "observable_paired_generation_similarity")
    assert node_by_type(nodes, "observable_rp_score_sscd")
    assert len(nodes_by_type(nodes, "observable_nearest_train_gl")) == 2
    for trainer in trainers:
        assert has_edge(edges, dataset.id, trainer.id, "dataset", "dataset")


def test_template_smoke_run(tmp_path, monkeypatch) -> None:
    """Reduced template trains one step, samples both endpoints, and runs the observables."""
    nodes, edges = _reduced_graph()

    with patch(_VISION_BUILD, side_effect=_synthetic_cifar):
        for trainer in nodes_by_type(nodes, "trainer"):
            events = list(iter_trainer_events(nodes, edges, trainer.id))
            assert events[-1]["type"] == "complete"
            assert all(math.isfinite(float(value)) for value in events[-1]["loss_history"])

    model_data = next(node.data for node in nodes if node.id == "wf-model-a")
    model_a = build_unet_ddpm_from_md(model_data)
    model_b = build_unet_ddpm_from_md(model_data)
    with torch.no_grad():
        for parameter in model_b.parameters():
            parameter.add_(0.001)
    for node in nodes:
        if node.id == "wf-ckpt-a":
            node.data.update(checkpointSource="memory", memoryCheckpoint_b64=_checkpoint_b64(model_a))
        elif node.id == "wf-ckpt-b":
            node.data.update(checkpointSource="memory", memoryCheckpoint_b64=_checkpoint_b64(model_b))

    api_nodes = [node.model_dump(mode="json") for node in nodes]
    api_edges = [edge.model_dump(mode="json") for edge in edges]
    monkeypatch.setattr(diffusion_api, "_SAMPLE_DIR", tmp_path)
    monkeypatch.setattr(
        diffusion_api,
        "build_cifar10_arrays",
        lambda *_args, **_kwargs: (
            np.zeros((8, 3, 32, 32), dtype=np.float32),
            np.arange(8, dtype=np.int64) % 10,
            np.zeros((4, 3, 32, 32), dtype=np.float32),
            np.arange(4, dtype=np.int64) % 10,
        ),
    )

    first = diffusion_api.sample(diffusion_api.SamplerRequest(nodes=api_nodes, edges=api_edges, sampler_node_id="wf-sampler-a"))
    second = diffusion_api.sample(diffusion_api.SamplerRequest(nodes=api_nodes, edges=api_edges, sampler_node_id="wf-sampler-b"))
    for node in api_nodes:
        if node["id"] == "wf-sampler-a":
            node["data"]["runId"] = first["runId"]
        elif node["id"] == "wf-sampler-b":
            node["data"]["runId"] = second["runId"]

    paired = diffusion_api.paired_similarity(diffusion_api.ObservableRequest(nodes=api_nodes, edges=api_edges, observable_node_id="wf-paired"))
    rp = diffusion_api.rp_score(diffusion_api.ObservableRequest(nodes=api_nodes, edges=api_edges, observable_node_id="wf-rp"))
    nearest = diffusion_api.nearest_train(diffusion_api.ObservableRequest(nodes=api_nodes, edges=api_edges, observable_node_id="wf-nearest-a"))

    assert first["previewGrid"].startswith("data:image/png;base64,")
    assert paired["imageGrid"].startswith("data:image/png;base64,")
    assert math.isfinite(float(paired["meanMae"]))
    assert math.isfinite(float(rp["rp"]))
    assert nearest["imageGrid"].startswith("data:image/png;base64,")
    assert math.isfinite(float(nearest["glScore"]))
