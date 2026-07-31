"""Prepare-level characterization golden over the trainer branch tree.

Densifies the test net for the trainer preparation decomposition:
every vector/token model family x dataset shape that prepare_trainer_run's
build tree dispatches on, including teacher paths, testSize=0 variants (the
conditional-placeholder hazard class), CE slot masking, and stable error
paths. Snapshots structure only (task, classes, tensor kind/shape/dtype,
flags) -- fast, prepare-only, no training.

Bootstrapped from the refactored main branch (behavior-identical by the
existing goldens). Regenerate ONLY on deliberate behavior changes:
COMFYRESEARCH_UPDATE_TRAINER_PREPARE_GOLDEN=1 python -m pytest -q \
    comfy_research/tests/test_trainer_prepare_golden.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from fastapi import HTTPException  # noqa: E402

from comfy_research.engine.runs.trainer_run import prepare_trainer_run  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "trainer_prepare_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_TRAINER_PREPARE_GOLDEN"

TRAINER = {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 1, "logFrequency": 1}
LINEAR_DS = {"inputDim": 3, "outputDim": 2, "trainSize": 8, "testSize": 4, "noiseLevel": 0, "seed": 0, "samplingMode": "fixed"}
MLP = {"inputDim": 3, "outputDim": 2, "depth": 2, "width": 6, "activation": "relu", "seed": 0}


def _std_edges() -> list[list[str]]:
    return [
        ["e-ds", "dataset", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-o", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
    ]


def _gr(dataset: dict, model: dict, loss_type: str, *, trainer_extra: dict | None = None,
        extra_nodes: list[dict] | None = None, extra_edges: list[list[str]] | None = None) -> dict:
    return {
        "nodes": [
            {"id": "dataset", **dataset},
            {"id": "model", **model},
            {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
            {"id": "loss", "type": loss_type, "data": {}},
            {"id": "trainer", "type": "trainer", "data": {**TRAINER, **(trainer_extra or {})}},
            *(extra_nodes or []),
        ],
        "edges": _std_edges() + (extra_edges or []),
    }


def _fixtures() -> dict[str, dict]:
    lin = lambda **kw: {"type": "linear_dataset", "data": {**LINEAR_DS, **kw}}  # noqa: E731
    fx: dict[str, dict] = {}

    # --- F1c vector-model family sweep over linear_dataset ---
    fx["vec_mlp"] = _gr(lin(), {"type": "mlp_model", "data": dict(MLP)}, "mse_loss")
    fx["vec_mlp_test0"] = _gr(lin(testSize=0), {"type": "mlp_model", "data": dict(MLP)}, "mse_loss")
    fx["vec_gated_mlp"] = _gr(lin(), {"type": "gated_mlp_model", "data": dict(MLP)}, "mse_loss")
    fx["vec_moe_mlp"] = _gr(lin(), {"type": "moe_mlp_model", "data": {**MLP, "numExperts": 2}}, "mse_loss")
    fx["vec_kan"] = _gr(lin(), {"type": "kan_model", "data": {"inputDim": 3, "outputDim": 2, "width": 4, "grid": 3, "k": 3, "seed": 0}}, "mse_loss")
    fx["err_residual_ln_mse"] = _gr(lin(), {"type": "residual_ln_model", "data": dict(MLP)}, "mse_loss")

    # --- numeric sequence models (kepler dataset drives [B,T,D] inputs) ---
    kepler = {"type": "kepler_2d_dataset", "data": {"contextLength": 4, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}
    fx["vec_numeric_transformer"] = _gr(kepler, {"type": "numeric_transformer_model", "data": {"inputDim": 2, "outputDim": 2, "contextLength": 4, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "mse_loss")
    fx["vec_numeric_hyena"] = _gr(kepler, {"type": "numeric_hyena_model", "data": {"inputDim": 2, "outputDim": 2, "contextLength": 4, "modelDim": 16, "numLayers": 1, "seed": 0}}, "mse_loss")

    # --- spatiotemporal / diffusion ---
    pde = {"type": "diffusion_pde_dataset", "data": {"contextFrames": 4, "channels": 1, "gridSize": 8, "trainSize": 8, "testSize": 4, "initSeed": 0, "samplingMode": "fixed"}}
    fx["vec_mpp_pde"] = _gr(pde, {"type": "mpp_spatiotemporal_model", "data": {"contextFrames": 4, "channels": 1, "gridSize": 8, "patchSize": 4, "inputDim": 256, "outputDim": 256, "embedDim": 32, "depth": 1, "numHeads": 2, "ffRatio": 2, "seed": 0}}, "mse_loss")
    fx["vec_afno_pde"] = _gr(pde, {"type": "afno_lite_spatiotemporal_model", "data": {"contextFrames": 4, "channels": 1, "gridSize": 8, "inputDim": 256, "outputDim": 256, "embedDim": 32, "depth": 1, "seed": 0}}, "mse_loss")
    fx["diffusion_score"] = _gr(
        {"type": "random_noise_dataset", "data": {"inputDim": 4, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"type": "diffusion_score_model", "data": {"inputDim": 4, "width": 8, "depth": 1, "seed": 0}},
        "diffusion_mse_loss",
    )
    # --- teacher paths (F1c full-model and F1b atomic tip) ---
    teacher_nodes = [
        {"id": "tmodel", "type": "mlp_model", "data": {"inputDim": 3, "outputDim": 2, "depth": 1, "width": 4, "activation": "relu", "seed": 1}},
        {"id": "sampler", "type": "input_sampler", "data": {"numSamples": 8, "seed": 0}},
        {"id": "sampler_te", "type": "input_sampler", "data": {"numSamples": 4, "seed": 1}},
        {"id": "rid", "type": "random_input_distribution", "data": {"inputDim": 3, "distribution": "normal"}},
    ]
    teacher_edges = [
        ["e-t", "tmodel", "dataset", "model", "model"],
        ["e-s", "sampler", "dataset", "sampler", "train_input"],
        ["e-ste", "sampler_te", "dataset", "sampler", "test_input"],
        ["e-r1", "rid", "sampler", "distribution", "distribution"],
        ["e-r2", "rid", "sampler_te", "distribution", "distribution"],
    ]
    teacher_ds = {"type": "teacher_dataset", "data": {"trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}
    fx["teacher_mlp_student"] = _gr(
        teacher_ds, {"type": "mlp_model", "data": dict(MLP)}, "mse_loss",
        extra_nodes=teacher_nodes, extra_edges=teacher_edges,
    )
    fx["teacher_atomic_student"] = _gr(
        teacher_ds, {"type": "linear_layer", "data": {"inFeatures": 3, "outFeatures": 2, "seed": 0}}, "mse_loss",
        extra_nodes=teacher_nodes, extra_edges=teacher_edges,
    )
    # --- cross_entropy_dense: memorization + slot masking + flattened vision ---
    mem_a = {"type": "memorization_a_dataset", "data": {"inputDim": 3, "outputDim": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}
    fx["mem_a_ce"] = _gr(mem_a, {"type": "mlp_model", "data": {**MLP, "outputDim": 3}}, "cross_entropy_loss")
    fx["mem_a_ce_slotmask"] = _gr(
        {"type": "memorization_a_dataset", "data": {"inputDim": 4, "outputDim": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"type": "mlp_model", "data": {**MLP, "inputDim": 4, "outputDim": 6}},
        "cross_entropy_loss",
    )
    fx["mem_a_ce_slotmask"]["nodes"][3]["data"] = {"lossMaskMode": "last_context", "lossMaskContextLength": 2}
    fx["vision_flatten_mlp_ce"] = _gr(
        {"type": "mnist_dataset", "data": {"trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed", "flattenOutput": True}},
        {"type": "mlp_model", "data": {**MLP, "inputDim": 784, "outputDim": 10}},
        "cross_entropy_loss",
    )

    # --- token family sweep ---
    tok_ds = {"type": "token_prediction_dataset", "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}
    fx["tok_mlp"] = _gr(tok_ds, {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss")
    fx["tok_transformer"] = _gr(tok_ds, {"type": "transformer_token_model", "data": {"vocabSize": 5, "contextLength": 3, "modelDim": 8, "numLayers": 1, "numHeads": 2, "seed": 0}}, "cross_entropy_loss")
    fx["tok_attention_only"] = _gr(tok_ds, {"type": "attention_only_model", "data": {"vocabSize": 5, "contextLength": 3, "modelDim": 8, "numHeads": 2, "seed": 0}}, "cross_entropy_loss")
    fx["tok_linear_attention"] = _gr(tok_ds, {"type": "linear_attention_model", "data": {"vocabSize": 5, "contextLength": 3, "modelDim": 8, "numHeads": 2, "seed": 0}}, "cross_entropy_loss")
    fx["tok_unigram_stream"] = _gr(
        {"type": "unigram_dataset", "data": {"vocabSize": 5, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}},
        {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}},
        "cross_entropy_loss",
    )
    fx["tok_modular_addition"] = _gr(
        {"type": "modular_addition_dataset", "data": {"modulus": 7, "trainFraction": 0.6, "seed": 0, "samplingMode": "fixed"}},
        {"type": "transformer_token_model", "data": {"vocabSize": 7, "contextLength": 2, "modelDim": 8, "numLayers": 1, "numHeads": 2, "seed": 0}},
        "cross_entropy_loss",
    )
    fx["tok_test0"] = _gr(
        {"type": "token_prediction_dataset", "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8, "testSize": 0, "seed": 0, "samplingMode": "fixed"}},
        {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}},
        "cross_entropy_loss",
    )

    # memB routed to the TOKEN task falls through the family chain and is
    # OVERWRITTEN by the fallback circle-walk draws -- characterized shipped
    # behavior. Do not "fix"
    # silently: flipping this is a post- user decision.
    fx["tok_memb_fallthrough"] = _gr(
        {"type": "memorization_b_dataset", "data": {"vocabSize": 6, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"type": "mlp_token_model", "data": {"vocabSize": 6, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}},
        "cross_entropy_loss",
    )

    # --- stable error paths (characterize the 4xx contract too) ---
    fx["err_token_model_mse"] = _gr(tok_ds, {"type": "transformer_token_model", "data": {"vocabSize": 5, "contextLength": 3, "modelDim": 8, "numLayers": 1, "numHeads": 2, "seed": 0}}, "mse_loss")
    fx["err_mem_a_mse"] = _gr(mem_a, {"type": "mlp_model", "data": dict(MLP)}, "mse_loss")

    # Divergence fixtures pin the branch-copy differences the
    # unified materialize must preserve (test_dataset_materialize_dispatch.py
    # KNOWN_DIVERGENCES maps each divergence to these entries). ---
    mixer_nodes = [
        {"id": "ds_a", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "noiseLevel": 0, "seed": 0, "samplingMode": "fixed"}},
        {"id": "ds_b", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "noiseLevel": 0, "seed": 1, "samplingMode": "fixed"}},
    ]
    mixer_edges = [["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]
    mixer_ds = {"type": "dataset_mixer", "data": {"trainTotalSamples": 8, "testTotalSamples": 4, "proportionA": 0.5, "seed": 0}}
    # teacher dim-mismatch: "Student model dimensions" (vector) vs "Student chain I/O" (atomic)
    fx["err_teacher_dim_mismatch_vec"] = _gr(
        teacher_ds, {"type": "mlp_model", "data": {**MLP, "inputDim": 4}}, "mse_loss",
        extra_nodes=teacher_nodes, extra_edges=teacher_edges,
    )
    fx["err_teacher_dim_mismatch_atomic"] = _gr(
        teacher_ds, {"type": "linear_layer", "data": {"inFeatures": 4, "outFeatures": 2, "seed": 0}}, "mse_loss",
        extra_nodes=teacher_nodes, extra_edges=teacher_edges,
    )
    # mixer dim-mismatch: "student model dimensions" (vector) vs "student chain dimensions" (atomic)
    fx["err_mixer_dim_mismatch_vec"] = _gr(
        mixer_ds, {"type": "mlp_model", "data": {**MLP, "outputDim": 1}}, "mse_loss",
        extra_nodes=mixer_nodes, extra_edges=mixer_edges,
    )
    fx["err_mixer_dim_mismatch_atomic"] = _gr(
        mixer_ds, {"type": "linear_layer", "data": {"inFeatures": 3, "outFeatures": 1, "seed": 0}}, "mse_loss",
        extra_nodes=mixer_nodes, extra_edges=mixer_edges,
    )
    # memB on the dense VECTOR path with testSize > 0: draws BOTH splits then
    # raises the test chain's terminal 400 (the atomic copy returns cleanly)
    fx["err_memb_vector_testsplit"] = _gr(
        {"type": "memorization_b_dataset", "data": {"vocabSize": 4, "inputDim": 4, "outputDim": 4, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"type": "mlp_model", "data": {"inputDim": 4, "outputDim": 4, "depth": 1, "width": 6, "activation": "relu", "seed": 0}},
        "cross_entropy_loss",
    )
    # circle_random_walk FAMILY path (vocab default 10-sourcing branch), the
    # counterpart of the fallback circle-walk that tok_memb_fallthrough hits
    fx["tok_circle_walk"] = _gr(
        {"type": "circle_random_walk_dataset", "data": {"vocabSize": 6, "contextLength": 3, "rightStepProb": 0.5, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
        {"type": "transformer_token_model", "data": {"vocabSize": 6, "contextLength": 3, "modelDim": 8, "numLayers": 1, "numHeads": 2, "seed": 0}},
        "cross_entropy_loss",
    )
    return fx


def _desc(t: Any) -> Any:
    if t is None:
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype)}


def _capture(fixture: dict) -> dict[str, Any]:
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n.get("data") or {})) for n in fixture["nodes"]]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in fixture["edges"]]
    try:
        ctx = prepare_trainer_run(nodes, edges, "trainer")
    except HTTPException as e:
        return {"error": {"status": e.status_code, "detail": str(e.detail)}}
    return {
        "trainer_task": str(ctx.trainer_task),
        "model_class": type(ctx.model).__name__,
        "criterion_class": type(ctx.criterion).__name__,
        "x": _desc(ctx.x_t),
        "y": _desc(ctx.y_t),
        "x_test": _desc(ctx.x_test_t),
        "y_test": _desc(ctx.y_test_t),
        "depth": ctx.depth,
        "test_size": ctx.test_size,
        "train_streaming": ctx.train_streaming,
        "token_lm_seq_len": ctx.token_lm_seq_len,
        "hessian_oversized_mode": ctx.hessian_oversized_mode,
    }


def test_trainer_prepare_golden() -> None:
    current = {name: _capture(fx) for name, fx in sorted(_fixtures().items())}
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"Updated snapshot at {SNAPSHOT_PATH}")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"bootstrap snapshot with {UPDATE_ENV}=1")
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(expected) == set(current), (
        f"fixture set changed: only in expected={sorted(set(expected) - set(current))}, "
        f"only in current={sorted(set(current) - set(expected))}"
    )
    for name in sorted(expected):
        assert expected[name] == current[name], (
            f"{name} diverged:\nexpected={json.dumps(expected[name], indent=1, sort_keys=True)}\n"
            f"current={json.dumps(current[name], indent=1, sort_keys=True)}"
        )
