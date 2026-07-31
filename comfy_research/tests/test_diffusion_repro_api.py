from __future__ import annotations

import base64
import asyncio
import io
import json

import numpy as np
import pytest
import torch

from comfy_research.api import diffusion_repro as api
from comfy_research.engine.models.unet_ddpm_model import build_unet_ddpm_from_md


def _checkpoint_b64(model) -> str:
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict()}, buffer)
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def _graph(checkpoint_b64: str) -> tuple[list[dict], list[dict]]:
    nodes = [
        {"id": "dataset", "type": "cifar10_dataset", "data": {"trainSize": 8, "normalize": "minus_one_to_one"}},
        {"id": "model", "type": "unet_ddpm_model", "data": {"inChannels": 3, "baseChannels": 8, "channelMult": "1", "timeEmbedDim": 16, "diffusionTimesteps": 8, "imageSize": 8}},
        {"id": "trainer", "type": "trainer", "data": {}},
        {"id": "checkpoint", "type": "model_checkpoint", "data": {"checkpointSource": "file", "checkpoint_b64": checkpoint_b64}},
        {"id": "sampler", "type": "deterministic_diffusion_sampler", "data": {"noiseSeed": 3, "sampleCount": 2, "numSteps": 2}},
        {"id": "paired", "type": "observable_paired_generation_similarity", "data": {}},
        {"id": "rp", "type": "observable_rp_score_sscd", "data": {"threshold": 0.95}},
    ]
    edges = [
        {"id": "dataset-trainer", "source": "dataset", "sourceHandle": "dataset", "target": "trainer", "targetHandle": "dataset"},
        {"id": "model-trainer", "source": "model", "sourceHandle": "model", "target": "trainer", "targetHandle": "model"},
        {"id": "trainer-checkpoint", "source": "trainer", "sourceHandle": "checkpoint", "target": "checkpoint", "targetHandle": "model_checkpoint"},
        {"id": "checkpoint-sampler", "source": "checkpoint", "sourceHandle": "model", "target": "sampler", "targetHandle": "checkpoint"},
    ]
    return nodes, edges


def test_sampler_and_paired_metrics_are_deterministic(tmp_path, monkeypatch) -> None:
    model = build_unet_ddpm_from_md({"inChannels": 3, "baseChannels": 8, "channelMult": "1", "timeEmbedDim": 16, "diffusionTimesteps": 8, "imageSize": 8})
    nodes, edges = _graph(_checkpoint_b64(model))
    monkeypatch.setattr(api, "_SAMPLE_DIR", tmp_path)

    first = api.sample(api.SamplerRequest(nodes=nodes, edges=edges, sampler_node_id="sampler"))
    second = api.sample(api.SamplerRequest(nodes=nodes, edges=edges, sampler_node_id="sampler"))
    stream = api.sample_stream(api.SamplerRequest(nodes=nodes, edges=edges, sampler_node_id="sampler"))

    async def stream_events() -> list[dict[str, object]]:
        chunks = [chunk async for chunk in stream.body_iterator]
        return [json.loads(line) for line in "".join(chunks).splitlines()]

    events = asyncio.run(stream_events())
    for node in nodes:
        if node["id"] == "sampler":
            node["data"]["runId"] = first["runId"]
    nodes.append({"id": "sampler-b", "type": "deterministic_diffusion_sampler", "data": {"runId": second["runId"]}})
    edges.extend([
        {"id": "a-paired", "source": "sampler", "sourceHandle": "samples", "target": "paired", "targetHandle": "sampler_a"},
        {"id": "b-paired", "source": "sampler-b", "sourceHandle": "samples", "target": "paired", "targetHandle": "sampler_b"},
        {"id": "a-rp", "source": "sampler", "sourceHandle": "samples", "target": "rp", "targetHandle": "sampler_a"},
        {"id": "b-rp", "source": "sampler-b", "sourceHandle": "samples", "target": "rp", "targetHandle": "sampler_b"},
    ])
    paired = api.paired_similarity(api.ObservableRequest(nodes=nodes, edges=edges, observable_node_id="paired"))
    rp = api.rp_score(api.ObservableRequest(nodes=nodes, edges=edges, observable_node_id="rp"))
    assert paired["meanMae"] == pytest.approx(0.0)
    assert paired["meanMse"] == pytest.approx(0.0)
    assert rp["rp"] == pytest.approx(1.0)
    assert first["previewGrid"].startswith("data:image/png;base64,")
    assert [event["type"] for event in events] == ["progress", "progress", "complete"]
    assert paired["imageGrid"].startswith("data:image/png;base64,")
    assert paired["histogramPng"].startswith("data:image/png;base64,")
    assert rp["histogramPng"].startswith("data:image/png;base64,")
    expected_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    assert first["metadata"]["device"] == expected_device
    assert paired["device"] == expected_device
    assert rp["device"] == expected_device


def test_nearest_train_uses_the_runtime_device(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "_SAMPLE_DIR", tmp_path)
    images = np.zeros((2, 3, 8, 8), dtype=np.float32)
    images[1] = 1.0
    run_id = api._save_run(images, {})
    nodes = [
        {"id": "dataset", "type": "cifar10_dataset", "data": {"trainSize": 4, "normalize": "zero_one"}},
        {"id": "sampler", "type": "deterministic_diffusion_sampler", "data": {"runId": run_id}},
        {"id": "nearest", "type": "observable_nearest_train_gl", "data": {"glThreshold": 0.95}},
    ]
    edges = [
        {"id": "sample-nearest", "source": "sampler", "sourceHandle": "samples", "target": "nearest", "targetHandle": "generated"},
        {"id": "dataset-nearest", "source": "dataset", "sourceHandle": "dataset", "target": "nearest", "targetHandle": "train_dataset"},
    ]
    train = np.zeros((4, 3, 8, 8), dtype=np.float32)
    train[1] = 1.0
    monkeypatch.setattr(
        api,
        "build_cifar10_arrays",
        lambda *_args, **_kwargs: (train, np.zeros(4), train, np.zeros(4)),
    )

    result = api.nearest_train(api.ObservableRequest(nodes=nodes, edges=edges, observable_node_id="nearest"))

    assert result["glScore"] == pytest.approx(0.5)
    assert result["imageGrid"].startswith("data:image/png;base64,")
    assert result["histogramPng"].startswith("data:image/png;base64,")
    assert result["device"] == ("cuda:0" if torch.cuda.is_available() else "cpu")
