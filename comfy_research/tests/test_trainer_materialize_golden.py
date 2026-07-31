from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import prepare_trainer_run  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "trainer_materialize_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_TRAINER_MATERIALIZE_GOLDEN"

TRAINER = {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 1, "logFrequency": 1}
ADAM = {"learningRate": 0.01}


def _node(node_id: str, node_type: str, data: dict[str, Any] | None = None) -> Node:
    return Node(id=node_id, type=NodeKind(node_type), data=dict(data or {}))


def _edge(edge_id: str, source: str, target: str, source_handle: str, target_handle: str) -> Edge:
    return Edge(id=edge_id, source=source, target=target, sourceHandle=source_handle, targetHandle=target_handle)


def _trainer_edges() -> list[list[str]]:
    return [
        ["dataset-trainer", "dataset", "trainer", "dataset", "dataset"],
        ["model-trainer", "model", "trainer", "model", "model"],
        ["optimizer-trainer", "optimizer", "trainer", "optimizer", "optimizer"],
        ["loss-trainer", "loss", "trainer", "loss", "loss"],
    ]


def _graph(dataset_nodes: list[dict], model: dict, loss_type: str, extra_edges: list[list[str]] | None = None) -> dict:
    nodes = [
        *dataset_nodes,
        {"id": "model", **model},
        {"id": "optimizer", "type": "adam_optimizer", "data": ADAM},
        {"id": "loss", "type": loss_type, "data": {}},
        {"id": "trainer", "type": "trainer", "data": dict(TRAINER)},
    ]
    return {"nodes": nodes, "edges": _trainer_edges() + (extra_edges or [])}


def _fixed_fixtures() -> dict[str, dict[str, Any]]:
    lin = {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "noiseLevel": 0, "seed": 0, "samplingMode": "fixed"}
    mlp2_1 = {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}
    tok_model = {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}
    return {
        "linear_mlp_mse": _graph(
            [{"id": "dataset", "type": "linear_dataset", "data": dict(lin)}], mlp2_1, "mse_loss"),
        "linear_xor2d_mlp_mse": _graph(
            [{"id": "dataset", "type": "linear_dataset", "data": {**lin, "targetKind": "xor2d"}}], mlp2_1, "mse_loss"),
        "token_prediction_mlp_ce": _graph(
            [{"id": "dataset", "type": "token_prediction_dataset", "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            tok_model, "cross_entropy_loss"),
        "memorization_a_mlp_ce": _graph(
            [{"id": "dataset", "type": "memorization_a_dataset", "data": {"inputDim": 4, "outputDim": 4, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "mlp_model", "data": {"inputDim": 4, "outputDim": 4, "depth": 1, "width": 6, "activation": "relu", "seed": 0}}, "cross_entropy_loss"),
        # memorization_b resolves to mse_regression here (unlike memorization_a -> cross_entropy_dense);
        # its mse test-split branch rejects memorization_b, so pin train-only (testSize=0). The
        # materialization path is still exercised; memorization_a covers the CE-dense test split.
        "memorization_b_mlp_trainonly": _graph(
            [{"id": "dataset", "type": "memorization_b_dataset", "data": {"vocabSize": 4, "inputDim": 4, "outputDim": 4, "trainSize": 8, "testSize": 0, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "mlp_model", "data": {"inputDim": 4, "outputDim": 4, "depth": 1, "width": 6, "activation": "relu", "seed": 0}}, "cross_entropy_loss"),
        "kepler_numeric_transformer_mse": _graph(
            [{"id": "dataset", "type": "kepler_2d_dataset", "data": {"contextLength": 4, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "numeric_transformer_model", "data": {"inputDim": 2, "outputDim": 2, "contextLength": 4, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "mse_loss"),
        "bigram_mlp_token_ce": _graph(
            [{"id": "dataset", "type": "bigram_low_rank_dataset", "data": {"vocabSize": 8, "rank": 3, "trainSize": 8, "testSize": 4, "seed": 0, "initSeed": 0, "alpha": 1.0, "samplingMode": "fixed"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 8, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "modular_addition_mlp_token_ce": _graph(
            [{"id": "dataset", "type": "modular_addition_dataset", "data": {"modulus": 7, "trainFraction": 0.5, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 7, "embedDim": 4, "tokensPerInput": 2, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "dataset_mixer_mlp_mse": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer", "data": {"trainTotalSamples": 8, "testTotalSamples": 4, "proportionA": 0.5, "seed": 0}},
                {"id": "ds_a", "type": "linear_dataset", "data": dict(lin)},
                {"id": "ds_b", "type": "linear_dataset", "data": {**lin, "seed": 1}},
            ],
            mlp2_1, "mse_loss",
            extra_edges=[["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]),
        "dataset_mixer_b_mlp_mse": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer_b", "data": {"interpolationLambda": 0.5, "seed": 0}},
                {"id": "ds_a", "type": "linear_dataset", "data": dict(lin)},
                {"id": "ds_b", "type": "linear_dataset", "data": dict(lin)},
            ],
            mlp2_1, "mse_loss",
            extra_edges=[["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]),
        "pcfg_transformer_token_ce": _graph(
            [{"id": "dataset", "type": "pcfg_dataset", "data": {"seed": 0, "dataSource": "synthetic", "trainSize": 8, "testSize": 4, "samplingMode": "fixed"}}],
            {"type": "transformer_token_model", "data": {"vocabSize": 32, "contextLength": 16, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "cross_entropy_loss"),
        # Non-vision leaf dataset configurations verified by the materialization probes.
        "uniform_motion_numeric_transformer_mse": _graph(
            [{"id": "dataset", "type": "uniform_linear_motion_dataset", "data": {"positionDim": 2, "contextLength": 2, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "numeric_transformer_model", "data": {"inputDim": 2, "outputDim": 2, "contextLength": 2, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "mse_loss"),
        "random_noise_mlp_mse": _graph(
            [{"id": "dataset", "type": "random_noise_dataset", "data": {"inputDim": 3, "outputDim": 2, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "mlp_model", "data": {"inputDim": 3, "outputDim": 2, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss"),
        "symbolic_func_mlp_mse": _graph(
            [{"id": "dataset", "type": "symbolic_func_dataset", "data": {"equationLatex": "x_1^2 + x_2", "inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}}],
            {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss"),
        "diffusion_pde_mpp_mse": _graph(
            [{"id": "dataset", "type": "diffusion_pde_dataset", "data": {"contextFrames": 4, "channels": 1, "gridSize": 8, "trainSize": 8, "testSize": 4, "initSeed": 0, "samplingMode": "fixed"}}],
            {"type": "mpp_spatiotemporal_model", "data": {"contextFrames": 4, "channels": 1, "gridSize": 8, "patchSize": 4, "inputDim": 256, "outputDim": 256, "embedDim": 32, "depth": 1, "numHeads": 2, "ffRatio": 2, "seed": 0}}, "mse_loss"),
        # Mixer RNG branches: interpolation endpoints, non-uniform proportions, and nesting.
        "mixer_b_lambda_1_0": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer_b", "data": {"interpolationLambda": 1.0, "seed": 0}},
                {"id": "ds_a", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
                {"id": "ds_b", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 1, "samplingMode": "fixed"}},
            ],
            {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss",
            extra_edges=[["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]),
        "mixer_b_lambda_0_0": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer_b", "data": {"interpolationLambda": 0.0, "seed": 0}},
                {"id": "ds_a", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
                {"id": "ds_b", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 1, "samplingMode": "fixed"}},
            ],
            {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss",
            extra_edges=[["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]),
        "mixer_proportion_0_25": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer", "data": {"proportionA": 0.25, "trainTotalSamples": 8, "testTotalSamples": 4, "seed": 0}},
                {"id": "ds_a", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
                {"id": "ds_b", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 1, "samplingMode": "fixed"}},
            ],
            {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss",
            extra_edges=[["a-mix", "ds_a", "dataset", "dataset", "dataset_a"], ["b-mix", "ds_b", "dataset", "dataset", "dataset_b"]]),
        "nested_concat_over_mixer_b": _graph(
            [
                {"id": "dataset", "type": "dataset_mixer", "data": {"proportionA": 0.5, "trainTotalSamples": 8, "testTotalSamples": 4, "seed": 0}},
                {"id": "mb", "type": "dataset_mixer_b", "data": {"interpolationLambda": 0.5, "seed": 0}},
                {"id": "l1", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "fixed"}},
                {"id": "l2", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 1, "samplingMode": "fixed"}},
                {"id": "l3", "type": "linear_dataset", "data": {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "seed": 2, "samplingMode": "fixed"}},
            ],
            {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}, "mse_loss",
            extra_edges=[
                ["mb-mix", "mb", "dataset", "dataset", "dataset_a"], ["l3-mix", "l3", "dataset", "dataset", "dataset_b"],
                ["l1-mb", "l1", "mb", "dataset", "dataset_a"], ["l2-mb", "l2", "mb", "dataset", "dataset_b"],
            ]),
    }


def _tensor_desc(t: Any) -> Any:
    if t is None:
        return {"kind": "none"}
    arr = t.detach().cpu().numpy()
    return {"kind": "ndarray", "dtype": str(arr.dtype), "shape": list(arr.shape), "data": arr.tolist()}


def _capture(ctx: Any, streaming: bool) -> dict[str, Any]:
    expected_test_size = 0 if ctx.x_test_t is None else int(ctx.x_test_t.shape[0])
    assert int(ctx.test_size) == expected_test_size, f"ctx.test_size {ctx.test_size} != tensor {expected_test_size}"
    out: dict[str, Any] = {
        "trainer_task": str(ctx.trainer_task),
        "train_streaming": bool(ctx.train_streaming),
        "test_size": int(ctx.test_size),
        "tensors": {
            "x_t": _tensor_desc(ctx.x_t),
            "y_t": _tensor_desc(ctx.y_t),
            "x_test_t": _tensor_desc(ctx.x_test_t),
            "y_test_t": _tensor_desc(ctx.y_test_t),
        },
    }
    if streaming:
        tr0_x, tr0_y = ctx.train_materialize(0)
        tr1_x, tr1_y = ctx.train_materialize(1)
        te0_x, te0_y = ctx.test_materialize(0)
        out["stream"] = {
            "train_0": [_tensor_desc(tr0_x), _tensor_desc(tr0_y)],
            "train_1": [_tensor_desc(tr1_x), _tensor_desc(tr1_y)],
            "test_0": [_tensor_desc(te0_x), _tensor_desc(te0_y)],
        }
    return out


def _build_graph_objects(fixture: dict[str, Any]) -> tuple[list[Node], list[Edge]]:
    nodes = [_node(str(n["id"]), str(n["type"]), n.get("data")) for n in fixture["nodes"]]
    edges = [_edge(*raw) for raw in fixture["edges"]]
    return nodes, edges


def _streaming_fixtures() -> dict[str, dict[str, Any]]:
    lin_stream = {"inputDim": 2, "outputDim": 1, "trainSize": 8, "testSize": 4, "noiseLevel": 0, "seed": 0, "samplingMode": "streaming"}
    mlp2_1 = {"type": "mlp_model", "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 4, "activation": "relu", "seed": 0}}
    tok_model = {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}
    return {
        "linear_mlp_mse_streaming": _graph(
            [{"id": "dataset", "type": "linear_dataset", "data": dict(lin_stream)}], mlp2_1, "mse_loss"),
        "token_prediction_mlp_ce_streaming": _graph(
            [{"id": "dataset", "type": "token_prediction_dataset", "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            tok_model, "cross_entropy_loss"),
        # Token-arm streaming branches; configs are verified to stream and redraw.
        "memorization_b_stream": _graph(
            [{"id": "dataset", "type": "memorization_b_dataset", "data": {"vocabSize": 4, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 4, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "unigram_stream": _graph(
            [{"id": "dataset", "type": "unigram_dataset", "data": {"vocabSize": 5, "contextLength": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "bigram_stream": _graph(
            [{"id": "dataset", "type": "bigram_low_rank_dataset", "data": {"vocabSize": 8, "rank": 3, "trainSize": 8, "testSize": 4, "seed": 0, "initSeed": 0, "alpha": 1.0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 8, "embedDim": 4, "tokensPerInput": 1, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "circle_random_walk_stream": _graph(
            [{"id": "dataset", "type": "circle_random_walk_dataset", "data": {"vocabSize": 16, "contextLength": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 16, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "circular_motion_stream": _graph(
            [{"id": "dataset", "type": "circular_motion_dataset", "data": {"vocabSize": 16, "contextLength": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "transformer_multi_token_model", "data": {"vocabSize": 16, "contextLength": 3, "tokensPerPosition": 2, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "cross_entropy_loss"),
        "associative_recall_stream": _graph(
            [{"id": "dataset", "type": "in_context_associative_recall_dataset", "data": {"vocabSize": 16, "numPairs": 3, "trainSize": 8, "testSize": 4, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 16, "embedDim": 4, "tokensPerInput": 7, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "modular_addition_stream": _graph(
            [{"id": "dataset", "type": "modular_addition_dataset", "data": {"modulus": 7, "trainFraction": 0.5, "seed": 0, "samplingMode": "streaming"}}],
            {"type": "mlp_token_model", "data": {"vocabSize": 7, "embedDim": 4, "tokensPerInput": 2, "depth": 1, "width": 6, "activation": "relu", "tieWeights": "no", "seed": 0}}, "cross_entropy_loss"),
        "pcfg_stream": _graph(
            [{"id": "dataset", "type": "pcfg_dataset", "data": {"seed": 0, "dataSource": "synthetic", "trainSize": 8, "testSize": 4, "samplingMode": "streaming"}}],
            {"type": "transformer_token_model", "data": {"vocabSize": 32, "contextLength": 16, "modelDim": 16, "numHeads": 1, "numLayers": 1, "ffDim": 32, "seed": 0}}, "cross_entropy_loss"),
    }


def _all_fixtures() -> dict[str, tuple[dict[str, Any], bool]]:
    combined: dict[str, tuple[dict[str, Any], bool]] = {name: (fx, False) for name, fx in _fixed_fixtures().items()}
    combined.update({name: (fx, True) for name, fx in _streaming_fixtures().items()})
    return combined


def _build_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for name, (fx, streaming) in _all_fixtures().items():
        nodes, edges = _build_graph_objects(fx)
        ctx = prepare_trainer_run(nodes, edges, "trainer")
        snap[name] = _capture(ctx, streaming)
    return snap


def _compare(expected: Any, actual: Any, path: str) -> None:
    if isinstance(expected, dict) and "kind" in expected:
        assert expected["kind"] == actual["kind"], f"{path}: kind {actual['kind']} != {expected['kind']}"
        if expected["kind"] == "none":
            return
        assert actual["shape"] == expected["shape"], f"{path}: shape {actual['shape']} != {expected['shape']}"
        assert actual["dtype"] == expected["dtype"], f"{path}: dtype {actual['dtype']} != {expected['dtype']}"
        np.testing.assert_allclose(np.array(actual["data"]), np.array(expected["data"]), rtol=1e-12, atol=1e-12, err_msg=path)
        return
    if isinstance(expected, dict):
        assert set(expected) == set(actual), f"{path}: keys {set(actual)} != {set(expected)}"
        for k in expected:
            _compare(expected[k], actual[k], f"{path}/{k}")
        return
    if isinstance(expected, list):
        assert len(expected) == len(actual), f"{path}: length mismatch"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _compare(e, a, f"{path}[{i}]")
        return
    assert expected == actual, f"{path}: {actual!r} != {expected!r}"


def test_trainer_materialize_golden() -> None:
    current = _build_snapshot()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"Updated snapshot at {SNAPSHOT_PATH}")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"bootstrap snapshot with {UPDATE_ENV}=1")
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(expected) == set(current), "fixture set changed; re-bootstrap the snapshot"
    for name in expected:
        _compare(expected[name], current[name], name)


def test_streaming_fixtures_redraw_between_steps() -> None:
    for name, (fx, streaming) in _all_fixtures().items():
        if not streaming:
            continue
        nodes, edges = _build_graph_objects(fx)
        ctx = prepare_trainer_run(nodes, edges, "trainer")
        assert ctx.train_streaming is True, f"{name}: expected streaming mode"
        x0, _ = ctx.train_materialize(0)
        x1, _ = ctx.train_materialize(1)
        assert not torch.equal(x0, x1), f"{name}: streaming train_materialize(0) == (1); path not redrawing"


def test_vision_classification_prepare_with_zero_test_size() -> None:
    """Regression: vision build stage must tolerate testSize=0.

    In the pre-split inline code x_test_t/y_test_t fell back to earlier-stage
    ``None`` placeholders when the vision branch skipped the test split; the
    extracted stage must preserve that (prologue-read from state), not crash
    with UnboundLocalError.
    """
    fixture = _graph(
        [
            {
                "id": "dataset",
                "type": "gaussian_blob_dataset",
                "data": {"trainSize": 8, "testSize": 0, "seed": 0, "samplingMode": "fixed"},
            }
        ],
        {"type": "resnet_model", "data": {"seed": 0}},
        "cross_entropy_loss",
    )
    nodes, edges = _build_graph_objects(fixture)
    ctx = prepare_trainer_run(nodes, edges, "trainer")
    assert ctx.trainer_task == "vision_classification"
    assert ctx.x_test_t is None and ctx.y_test_t is None
    assert ctx.x_t.shape[0] == 8
