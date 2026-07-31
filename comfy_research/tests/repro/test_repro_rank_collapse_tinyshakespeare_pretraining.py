"""
phenomenon:
    Real-text pretraining can exhibit non-monotonic final-hidden representation
    geometry analogous to the paper's warmup, expansion, and compression phases.

settings:
    TinyShakespeare word-level LM, vocab 256, context 32, 4000 windows, a
    4-layer d=4 causal Transformer, AdamW(lr=5e-3, wd=1e-3), next-token CE,
    batch 32, 20k updates, and final-hidden RankMe plus alphaReQ. Weight decay
    is the sensitive knob: at wd=1e-2 the decay term flattens the narrow
    spectrum before the transient expansion forms, so the collapse stays inside
    the noise band.

result:
    Baseline locks the real-text audit graph and claim boundary. Smoke shortens
    only an in-memory copy to 3 steps / 32 windows, uses an explicit local real
    text cache, runs the native Trainer to complete, and requires finite loss,
    RankMe, eigenvalues and alphaReQ. Smoke does not assert the three phases.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.models.token_transformer_model import TokenTransformerModel
from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import Edge, GraphDocument, Node
from comfy_research.schemas.saved_graph_library import SavedGraphEntry


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "data" / "graph_library" / "templates" / "repro-rank-collapse-tinyshakespeare-pretraining.json"


def _load() -> SavedGraphEntry:
    return SavedGraphEntry.model_validate(json.loads(TEMPLATE_PATH.read_text(encoding="utf-8")))


def _type(node: Node) -> str:
    return node.type.value if hasattr(node.type, "value") else str(node.type)


def _one(document: GraphDocument, node_type: str) -> Node:
    matches = [node for node in document.nodes if _type(node) == node_type]
    assert len(matches) == 1
    return matches[0]


def _edge_signature(
    document: GraphDocument,
    edge: Edge,
) -> tuple[str, str | None, str, str | None]:
    node_types = {node.id: _type(node) for node in document.nodes}
    return (
        node_types[edge.source],
        edge.sourceHandle,
        node_types[edge.target],
        edge.targetHandle,
    )


def _finite(values) -> bool:
    return bool(values) and all(math.isfinite(float(value)) for value in values)


def test_template_baseline() -> None:
    entry = _load()
    document = entry.document
    assert entry.id == "repro-rank-collapse-tinyshakespeare-pretraining"
    assert entry.name == "repro: Rank Collapse TinyShakespeare spectral audit"
    expected = {
        "tinyshakespeare_lm_dataset", "transformer_token_model", "adamw_optimizer",
        "cross_entropy_loss", "observable_representation_rankme",
        "observable_representation_alpha_req", "trainer",
    }
    assert expected <= {_type(node) for node in document.nodes}
    assert "paper_repro_runner" not in {_type(node) for node in document.nodes}
    assert (next(node for node in document.nodes if node.id == "rct-note").data or {}).get("__collapsed") is True
    for node in document.nodes:
        if node.id in {"rct-dataset", "rct-model", "rct-optimizer", "rct-loss", "rct-rankme", "rct-alpha", "rct-trainer"}:
            assert (node.data or {}).get("__collapsed") is not True

    dataset = _one(document, "tinyshakespeare_lm_dataset")
    model = _one(document, "transformer_token_model")
    optimizer = _one(document, "adamw_optimizer")
    loss = _one(document, "cross_entropy_loss")
    trainer = _one(document, "trainer")
    assert dataset.data == {"vocabSize": 256, "contextLength": 32, "trainSize": 4000, "testSize": 0,
                            "seed": 0, "initSeed": 0, "stride": 1,
                            "instanceTitle": "TinyShakespeare real-text LM corpus"}
    model_data = model.data or {}
    assert model_data["vocabSize"] == 256
    assert model_data["contextLength"] == 32
    assert model_data["modelDim"] == 4
    assert model_data["numLayers"] == 4
    assert model_data["numHeads"] == 4
    assert model_data["ffDim"] == 64
    assert model_data["activation"] == "gelu"
    assert model_data["encoderBackend"] == "stable"
    assert model_data["encoderDropout"] == pytest.approx(0.0)
    assert model_data["tieEmbeddingLmHead"] == "yes"
    assert model_data["causalAttention"] == "yes"
    assert model_data["seed"] == 0

    optimizer_data = optimizer.data or {}
    assert optimizer_data["learningRate"] == pytest.approx(5e-3)
    assert optimizer_data["beta1"] == pytest.approx(0.9)
    assert optimizer_data["beta2"] == pytest.approx(0.99)
    assert optimizer_data["epsilon"] == pytest.approx(1e-8)
    # Weight decay is the knob that decides whether the collapse is visible at
    # all: at 1e-2 the decay term flattens the narrow d=4 spectrum before the
    # transient expansion can form. The reproduction locks it at 1e-3.
    assert optimizer_data["weightDecay"] == pytest.approx(1e-3)

    loss_data = loss.data or {}
    assert loss_data["lossScale"] == pytest.approx(1.0)
    assert loss_data["labelSmoothing"] == pytest.approx(0.0)

    trainer_data = trainer.data or {}
    assert trainer_data["trainingSteps"] == 20_000
    assert trainer_data["logFrequency"] == 100
    assert trainer_data["batchSize"] == 32
    assert trainer_data["computeDevice"] == "cpu"
    assert trainer_data["gradClipMaxNorm"] == pytest.approx(1.0)
    for observable_type in ("observable_representation_rankme", "observable_representation_alpha_req"):
        data = _one(document, observable_type).data or {}
        assert data["representationId"] == "lm_head::input"
        assert data["tokenPositionsAsSamples"] is True

    signatures = {_edge_signature(document, edge) for edge in document.edges}
    assert {
        ("tinyshakespeare_lm_dataset", "dataset", "trainer", "dataset"),
        ("transformer_token_model", "model", "trainer", "model"),
        ("adamw_optimizer", "optimizer", "trainer", "optimizer"),
        ("cross_entropy_loss", "loss", "trainer", "loss"),
        (
            "observable_representation_rankme",
            "observables",
            "trainer",
            "observables",
        ),
        (
            "observable_representation_alpha_req",
            "observables",
            "trainer",
            "observables",
        ),
        ("trainer", "loss_results", "training_visualization", "tensor_list"),
        ("trainer", "observable_results", "observable_viz", "tensor"),
    } <= signatures


def test_template_smoke_run_uses_real_text_in_memory_copy(tmp_path) -> None:
    original = TEMPLATE_PATH.read_text(encoding="utf-8")
    document = copy.deepcopy(_load().document)
    dataset = _one(document, "tinyshakespeare_lm_dataset")
    trainer = _one(document, "trainer")
    assert dataset.data is not None and trainer.data is not None
    (tmp_path / "input.txt").write_text(("to be or not to be that is the question\n" * 512), encoding="utf-8")
    dataset.data["cacheDir"] = str(tmp_path)
    dataset.data["trainSize"] = 32
    trainer.data["trainingSteps"] = 3
    trainer.data["logFrequency"] = 1

    context = prepare_trainer_run(list(document.nodes), list(document.edges), trainer.id)
    assert context.trainer_task == "token_classification"
    assert isinstance(context.model, TokenTransformerModel)
    assert tuple(context.x_t.shape) == (32, 32)
    assert tuple(context.y_t.shape) == (32, 32)
    complete = [event for event in iter_trainer_events_from_context(context) if event.get("type") == "complete"]
    assert len(complete) == 1
    payload = complete[0]
    assert payload["step_ticks"] == [0, 1, 2, 3]
    assert _finite(payload["loss_history"])
    histories = payload["observable_metric_histories"]
    rank_id = _one(document, "observable_representation_rankme").id
    alpha_id = _one(document, "observable_representation_alpha_req").id
    assert _finite(histories[rank_id])
    assert _finite(histories[f"{rank_id}::eig::0"])
    assert _finite(histories[f"{rank_id}::eig::1"])
    assert _finite(histories[alpha_id])
    assert TEMPLATE_PATH.read_text(encoding="utf-8") == original
