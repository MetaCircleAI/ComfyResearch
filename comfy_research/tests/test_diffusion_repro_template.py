"""Regression checks for the asset-free diffusion reproducibility template."""

from __future__ import annotations

import json
from pathlib import Path

from comfy_research.engine.runs.trainer_run import prepare_trainer_run
from comfy_research.schemas.graph import GraphDocument
from comfy_research.schemas.saved_graph_library import SavedGraphEntry


_TEMPLATES = Path(__file__).resolve().parents[2] / "data/graph_library/templates"
_SLUG = "repro-diffusion-same-init-different-seed"


def _load(slug: str) -> tuple[SavedGraphEntry, GraphDocument]:
    entry = SavedGraphEntry.model_validate(
        json.loads((_TEMPLATES / f"{slug}.json").read_text(encoding="utf-8"))
    )
    return entry, GraphDocument.model_validate(entry.document)


def test_diffusion_repro_template_is_small_complete_and_asset_free() -> None:
    entry, document = _load(_SLUG)
    by_id = {node.id: node for node in document.nodes}

    assert entry.tier == "small"
    assert len(document.nodes) == 21
    assert len(document.edges) == 22
    assert by_id["wf-dataset"].data["trainSize"] == 4_096
    assert all(by_id[f"wf-trainer-{suffix}"].data["trainingSteps"] == 5_000 for suffix in ("a", "b"))
    assert by_id["wf-model-a"].data["seed"] == by_id["wf-model-b"].data["seed"] == 0
    assert by_id["wf-trainer-a"].data["seed"] == 0
    assert by_id["wf-trainer-b"].data["seed"] == 1
    assert all(not node.data.get("checkpoint_b64") for node in document.nodes if node.type.value == "model_checkpoint")
    assert all(not node.data.get("memoryCheckpoint_b64") for node in document.nodes if node.type.value == "model_checkpoint")
    assert all(not node.data.get("runId") for node in document.nodes if node.type.value == "deterministic_diffusion_sampler")
    assert all(not node.data.get("previewGrid") for node in document.nodes if node.type.value == "deterministic_diffusion_sampler")
    assert all(not node.data.get("histogramPng") for node in document.nodes)
    assert all(not node.data.get("imageGrid") for node in document.nodes)
    assert all(not node.data.get("lossHistory") for node in document.nodes)
    assert all(not node.data.get("testLossHistory") for node in document.nodes)


def test_diffusion_repro_template_wires_the_complete_two_endpoint_protocol() -> None:
    _, document = _load(_SLUG)
    edges = {(edge.source, edge.sourceHandle, edge.target, edge.targetHandle) for edge in document.edges}

    expected = {
        ("wf-dataset", "dataset", "wf-trainer-a", "dataset"),
        ("wf-dataset", "dataset", "wf-trainer-b", "dataset"),
        ("wf-model-a", "model", "wf-trainer-a", "model"),
        ("wf-model-b", "model", "wf-trainer-b", "model"),
        ("wf-opt-a", "optimizer", "wf-trainer-a", "optimizer"),
        ("wf-opt-b", "optimizer", "wf-trainer-b", "optimizer"),
        ("wf-loss-a", "loss", "wf-trainer-a", "loss"),
        ("wf-loss-b", "loss", "wf-trainer-b", "loss"),
        ("wf-trainer-a", "checkpoint", "wf-ckpt-a", "model_checkpoint"),
        ("wf-trainer-b", "checkpoint", "wf-ckpt-b", "model_checkpoint"),
        ("wf-ckpt-a", "model", "wf-sampler-a", "checkpoint"),
        ("wf-ckpt-b", "model", "wf-sampler-b", "checkpoint"),
        ("wf-sampler-a", "samples", "wf-paired", "sampler_a"),
        ("wf-sampler-b", "samples", "wf-paired", "sampler_b"),
        ("wf-sampler-a", "samples", "wf-rp", "sampler_a"),
        ("wf-sampler-b", "samples", "wf-rp", "sampler_b"),
        ("wf-sampler-a", "samples", "wf-nearest-a", "generated"),
        ("wf-sampler-b", "samples", "wf-nearest-b", "generated"),
        ("wf-dataset", "dataset", "wf-nearest-a", "train_dataset"),
        ("wf-dataset", "dataset", "wf-nearest-b", "train_dataset"),
    }

    assert expected <= edges


def test_diffusion_repro_template_compiles_without_loading_cifar10() -> None:
    _, document = _load(_SLUG)

    context = prepare_trainer_run(document.nodes, document.edges, "wf-trainer-a", validate_only=True)

    assert context.trainer_task == "diffusion_noise"
    assert context.training_steps == 5_000
