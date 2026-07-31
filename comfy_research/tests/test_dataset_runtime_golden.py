from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import HTTPException

from comfy_research.engine.datasets import dataset_runtime as dr
from comfy_research.schemas.graph import Edge, Node, NodeKind

SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "dataset_runtime_golden.json"
UPDATE_ENV = "COMFYRESEARCH_UPDATE_DATASET_RUNTIME_GOLDEN"


def _node(node_id: str, node_type: str, data: dict[str, Any] | None = None) -> Node:
    return Node(id=node_id, type=NodeKind(node_type), data=dict(data or {}))


def _edge(edge_id: str, source: str, target: str, target_handle: str) -> Edge:
    return Edge(id=edge_id, source=source, target=target, sourceHandle="dataset", targetHandle=target_handle)


# --- Toy-language materialization for offline and synthetic datasets ---
# scan_dataset and cogs_dataset are intentionally excluded: their builders iterate
# hash-seed-dependent string sets, so their token ordering (thus array values) varies
# across processes under Python hash randomization — un-snapshottable without globally
# pinning PYTHONHASHSEED. They still go through the identical TOY_LM_BUILDERS lookup as
# the 11 pinned kinds, so the shared registry path is covered transitively.
def _toy_lm_fixtures() -> dict[str, tuple[str, dict[str, Any], int, int]]:
    base = {"seed": 0, "dataSource": "synthetic"}
    kinds = [
        "pcfg_dataset", "dyck_dataset", "ngram_language_dataset", "formal_language_suite_dataset",
        "listops_dataset", "tinystories_dataset", "phi1_style_dataset",
        "biography_lm_dataset", "relation_tuple_dataset", "synthetic_playground_dataset",
        "multi_hop_fact_chain_dataset",
    ]
    return {k: (k, dict(base), 6, 4) for k in kinds}


# --- Split sizes for special types, defaults, and error paths ---
def _split_size_fixtures() -> dict[str, tuple[list[Node], list[Edge], str]]:
    fx: dict[str, tuple[list[Node], list[Edge], str]] = {}
    fx["plain_default"] = ([_node("d", "linear_dataset", {"trainSize": 800, "testSize": 200})], [], "d")
    fx["mixer"] = (
        [_node("m", "dataset_mixer", {"trainTotalSamples": 1000, "testTotalSamples": 250})], [], "m",
    )
    fx["modular_addition"] = (
        [_node("d", "modular_addition_dataset", {"modulus": 59, "trainFraction": 0.3})], [], "d",
    )
    a = _node("a", "linear_dataset", {"trainSize": 800, "testSize": 200})
    b = _node("b", "linear_dataset", {"trainSize": 800, "testSize": 200})
    mb = _node("mb", "dataset_mixer_b", {})
    fx["mixer_b_ok"] = (
        [mb, a, b],
        [_edge("e1", "a", "mb", "dataset_a"), _edge("e2", "b", "mb", "dataset_b")],
        "mb",
    )
    return fx


def _split_size_error_fixtures() -> dict[str, tuple[list[Node], list[Edge], str, int, str]]:
    mb = _node("mb", "dataset_mixer_b", {})
    a = _node("a", "linear_dataset", {"trainSize": 800, "testSize": 200})
    b_bad = _node("b", "linear_dataset", {"trainSize": 400, "testSize": 200})
    return {
        "mixer_b_missing": (
            [mb], [], "mb", 400,
            "dataset_mixer_b requires both dataset_a and dataset_b connections.",
        ),
        "mixer_b_mismatch": (
            [mb, a, b_bad],
            [_edge("e1", "a", "mb", "dataset_a"), _edge("e2", "b", "mb", "dataset_b")],
            "mb", 400,
            "dataset_mixer_b requires dataset_a and dataset_b to have matching train/test sizes.",
        ),
        "mixer_b_cycle": (
            [mb, _node("b", "linear_dataset", {"trainSize": 800, "testSize": 200})],
            [_edge("e1", "mb", "mb", "dataset_a"), _edge("e2", "b", "mb", "dataset_b")],
            "mb", 400,
            "dataset_mixer_b graph contains a cycle.",
        ),
    }


def _describe(obj: Any) -> Any:
    if obj is None:
        return {"kind": "none"}
    if isinstance(obj, np.ndarray):
        return {"kind": "ndarray", "dtype": str(obj.dtype), "shape": list(obj.shape), "data": obj.tolist()}
    if isinstance(obj, (list, tuple)):
        return {"kind": "seq", "items": [_describe(x) for x in obj]}
    if isinstance(obj, (int, np.integer)):
        return {"kind": "int", "value": int(obj)}
    if isinstance(obj, (float, np.floating)):
        return {"kind": "float", "value": float(obj)}
    raise TypeError(f"Unsnapshottable type: {type(obj)!r}")


def _build_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {"toy_lm": {}, "split_sizes": {}}
    for name, (kind, dd, tr, te) in _toy_lm_fixtures().items():
        snap["toy_lm"][name] = _describe(dr._materialize_toy_language_lm_numpy(NodeKind(kind), dd, tr, te))
    for name, (nodes, edges, root) in _split_size_fixtures().items():
        nmap = {n.id: n for n in nodes}
        snap["split_sizes"][name] = _describe(dr._declared_dataset_split_sizes(nmap[root], edges, nmap))
    return snap


def _compare(expected: Any, actual: Any, path: str) -> None:
    assert expected["kind"] == actual["kind"], f"{path}: kind {actual['kind']} != {expected['kind']}"
    kind = expected["kind"]
    if kind == "none":
        return
    if kind == "int":
        assert actual["value"] == expected["value"], f"{path}: {actual['value']} != {expected['value']}"
        return
    if kind == "float":
        np.testing.assert_allclose(actual["value"], expected["value"], rtol=1e-12, atol=1e-12, err_msg=path)
        return
    if kind == "seq":
        assert len(actual["items"]) == len(expected["items"]), f"{path}: length mismatch"
        for i, (e, a) in enumerate(zip(expected["items"], actual["items"])):
            _compare(e, a, f"{path}[{i}]")
        return
    if kind == "ndarray":
        assert actual["shape"] == expected["shape"], f"{path}: shape {actual['shape']} != {expected['shape']}"
        assert actual["dtype"] == expected["dtype"], f"{path}: dtype {actual['dtype']} != {expected['dtype']}"
        np.testing.assert_allclose(np.array(actual["data"]), np.array(expected["data"]), rtol=1e-12, atol=1e-12, err_msg=path)
        return
    raise AssertionError(f"{path}: unknown kind {kind!r}")


def test_dataset_runtime_golden() -> None:
    current = _build_snapshot()
    if os.environ.get(UPDATE_ENV) == "1":
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"Updated snapshot at {SNAPSHOT_PATH}")
    if not SNAPSHOT_PATH.exists():
        pytest.skip(f"bootstrap snapshot with {UPDATE_ENV}=1")
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(expected["toy_lm"]) == set(current["toy_lm"]), "toy_lm fixture set changed; re-bootstrap"
    assert set(expected["split_sizes"]) == set(current["split_sizes"]), "split_sizes fixture set changed; re-bootstrap"
    for group in ("toy_lm", "split_sizes"):
        for name in expected[group]:
            _compare(expected[group][name], current[group][name], f"{group}/{name}")


def test_toy_lm_unsupported_kind_raises_500() -> None:
    with pytest.raises(HTTPException) as exc:
        dr._materialize_toy_language_lm_numpy(NodeKind.linear_dataset, {"seed": 0}, 4, 2)
    assert exc.value.status_code == 500
    assert exc.value.detail == f"Internal: unsupported toy language dataset {NodeKind.linear_dataset}"


@pytest.mark.parametrize("name", list(_split_size_error_fixtures()))
def test_split_size_error_semantics(name: str) -> None:
    nodes, edges, root, status, detail = _split_size_error_fixtures()[name]
    nmap = {n.id: n for n in nodes}
    with pytest.raises(HTTPException) as exc:
        dr._declared_dataset_split_sizes(nmap[root], edges, nmap)
    assert exc.value.status_code == status
    assert exc.value.detail == detail


def test_dataset_runtime_does_not_import_trainer_run() -> None:
    import ast

    tree = ast.parse(Path(dr.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    offenders = [m for m in imported if "trainer_run" in m]
    assert not offenders, f"dataset_runtime must not import trainer_run; found {offenders}"


def test_trainer_run_reexports_dataset_runtime_verbs() -> None:
    pytest.importorskip("torch")
    import importlib

    trainer_run = importlib.import_module("comfy_research.engine.runs.trainer_run")
    for name in ("_declared_dataset_split_sizes", "_materialize_toy_language_lm_numpy"):
        fn = getattr(trainer_run, name, None)
        assert fn is not None, f"trainer_run no longer re-exports {name}"
        # Must be the re-exported object, not a re-defined local function.
        assert fn.__module__ == "comfy_research.engine.datasets.dataset_runtime", (
            f"trainer_run.{name} must be re-exported from dataset_runtime, "
            f"got {getattr(fn, '__module__', None)!r}"
        )


def test_toy_lm_registry_maps_each_kind_to_its_builder() -> None:
    # Catches any mis-mapping in TOY_LM_BUILDERS (including scan/cogs, which are
    # excluded from the value-golden), independent of hash-seed non-determinism.
    expected = {
        NodeKind.pcfg_dataset: "build_pcfg_lm_arrays_from_seed",
        NodeKind.dyck_dataset: "build_dyck_lm_arrays_from_seed",
        NodeKind.ngram_language_dataset: "build_ngram_lm_arrays_from_seed",
        NodeKind.formal_language_suite_dataset: "build_formal_suite_lm_arrays_from_seed",
        NodeKind.scan_dataset: "build_scan_arrays_from_seed",
        NodeKind.cogs_dataset: "build_cogs_arrays_from_seed",
        NodeKind.listops_dataset: "build_listops_arrays_from_seed",
        NodeKind.tinystories_dataset: "tiny_stories_lm_from_seed",
        NodeKind.phi1_style_dataset: "phi_style_lm_from_seed",
        NodeKind.biography_lm_dataset: "build_biography_lm_arrays_from_seed",
        NodeKind.relation_tuple_dataset: "build_relation_tuple_lm_arrays_from_seed",
        NodeKind.synthetic_playground_dataset: "build_synthetic_playground_lm_arrays_from_seed",
        NodeKind.multi_hop_fact_chain_dataset: "build_multi_hop_fact_chain_lm_arrays_from_seed",
        NodeKind.tinyshakespeare_lm_dataset: "tiny_shakespeare_lm_from_seed",
    }
    assert {k: fn.__name__ for k, fn in dr.TOY_LM_BUILDERS.items()} == expected


@pytest.mark.parametrize("kind", ["scan_dataset", "cogs_dataset"])
def test_hash_variant_toy_lm_registry_path_runs(kind: str) -> None:
    # scan/cogs are excluded from the value-golden (hash-seed non-determinism);
    # smoke-check that their registry path resolves to a working builder and
    # returns token sequences of the expected structure.
    x, y, x_te, y_te = dr._materialize_toy_language_lm_numpy(
        NodeKind(kind), {"seed": 0, "dataSource": "synthetic"}, 6, 4
    )
    for arr in (x, y, x_te, y_te):
        assert isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.dtype == np.int64
    assert x.shape[0] == 6 and y.shape[0] == 6
    assert x_te.shape[0] == 4 and y_te.shape[0] == 4


# --- Leaf-sampling registry guards ---
EXPECTED_LEAF_KINDS = {
    NodeKind.uniform_linear_motion_dataset, NodeKind.kepler_2d_dataset,
    NodeKind.diffusion_pde_dataset, NodeKind.reaction_diffusion_dataset, NodeKind.advection_dataset,
    NodeKind.memorization_a_dataset, NodeKind.memorization_b_dataset,
    NodeKind.random_noise_dataset, NodeKind.linear_dataset, NodeKind.symbolic_func_dataset,
}


def test_leaf_handlers_key_set_is_exact_and_vision_free() -> None:
    assert set(dr.LEAF_SAMPLE_HANDLERS) == EXPECTED_LEAF_KINDS
    from comfy_research.engine.runs.trainer_run import _VISION_DATASET_TYPES

    assert not (set(dr.LEAF_SAMPLE_HANDLERS) & set(_VISION_DATASET_TYPES))


def test_leaf_handler_routing() -> None:
    h = dr.LEAF_SAMPLE_HANDLERS
    assert h[NodeKind.diffusion_pde_dataset] is h[NodeKind.reaction_diffusion_dataset] is h[NodeKind.advection_dataset]
    # random_noise must NOT resolve to the linear-like handler (precedence subtlety).
    assert h[NodeKind.random_noise_dataset] is not h[NodeKind.linear_dataset]


def _mat_rng():
    return np.random.default_rng(0)


def test_leaf_memorization_a_non_ce_raises() -> None:
    node = _node("d", "memorization_a_dataset", {"inputDim": 4, "outputDim": 4})
    with pytest.raises(HTTPException) as exc:
        dr.LEAF_SAMPLE_HANDLERS[NodeKind.memorization_a_dataset](
            node, 4, _mat_rng(), trainer_task="mse_regression", eff_task="mse_regression")
    assert exc.value.status_code == 400
    assert "memorization_a_dataset is only valid with cross_entropy_loss" in exc.value.detail


def test_leaf_memorization_b_non_ce_raises_else_detail() -> None:
    node = _node("d", "memorization_b_dataset", {"vocabSize": 4})
    with pytest.raises(HTTPException) as exc:
        dr.LEAF_SAMPLE_HANDLERS[NodeKind.memorization_b_dataset](
            node, 4, _mat_rng(), trainer_task="mse_regression", eff_task="mse_regression")
    assert exc.value.status_code == 400
    assert exc.value.detail == dr._ELSE_RAISE_DETAIL


def test_leaf_symbolic_missing_equation_wraps_error() -> None:
    node = _node("sym1", "symbolic_func_dataset", {"inputDim": 2, "outputDim": 1, "instanceTitle": "MySym"})
    with pytest.raises(HTTPException) as exc:
        dr.LEAF_SAMPLE_HANDLERS[NodeKind.symbolic_func_dataset](
            node, 4, _mat_rng(), trainer_task="mse_regression", eff_task="mse_regression")
    assert exc.value.status_code == 400
    assert "equationLatex is required" in exc.value.detail
    assert "Triggered by symbolic_func_dataset 'MySym'." in exc.value.detail


# --- Mixer combinator guards ---
def test_mixer_b_dispatch_maps_are_leaf_only() -> None:
    combos = {NodeKind.dataset_mixer, NodeKind.dataset_mixer_b}
    assert not (set(dr._MIXER_B_INPUT_DRAWERS) & combos), "input drawers must not contain combinators"
    assert not (set(dr._MIXER_B_OUTPUT_SAMPLERS) & combos), "output samplers must not contain combinators"
    assert set(dr._MIXER_B_INPUT_DRAWERS) == {
        NodeKind.uniform_linear_motion_dataset, NodeKind.kepler_2d_dataset,
        NodeKind.diffusion_pde_dataset, NodeKind.reaction_diffusion_dataset, NodeKind.advection_dataset,
        NodeKind.memorization_b_dataset, NodeKind.memorization_a_dataset,
    }
    assert set(dr._MIXER_B_OUTPUT_SAMPLERS) == {
        NodeKind.memorization_b_dataset, NodeKind.memorization_a_dataset, NodeKind.random_noise_dataset,
        NodeKind.linear_dataset, NodeKind.symbolic_func_dataset,
    }


def test_mixer_b_output_random_noise_not_linear_like() -> None:
    assert dr._MIXER_B_OUTPUT_SAMPLERS[NodeKind.random_noise_dataset] is not dr._MIXER_B_OUTPUT_SAMPLERS[NodeKind.linear_dataset]
    assert dr._MIXER_B_INPUT_DRAWERS[NodeKind.diffusion_pde_dataset] is dr._MIXER_B_INPUT_DRAWERS[NodeKind.advection_dataset]


def _noop_leaf(*_a, **_k):
    raise AssertionError("leaf_sampler should not be reached for these error cases")


def test_mixer_proportion_out_of_range_raises() -> None:
    m = _node("m", "dataset_mixer", {"proportionA": 1.5})
    a = _node("a", "linear_dataset", {"inputDim": 2, "outputDim": 1})
    b = _node("b", "linear_dataset", {"inputDim": 2, "outputDim": 1})
    edges = [_edge("ea", "a", "m", "dataset_a"), _edge("eb", "b", "m", "dataset_b")]
    with pytest.raises(HTTPException) as exc:
        dr._sample_dataset_arrays_for_mixer(m, 8, edges, {"m": m, "a": a, "b": b}, "mse_regression", leaf_sampler=_noop_leaf)
    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_mixer proportionA must be in [0, 1]."


def test_mixer_b_lambda_out_of_range_raises() -> None:
    m = _node("m", "dataset_mixer_b", {"interpolationLambda": 2.0})
    a = _node("a", "linear_dataset", {"inputDim": 2, "outputDim": 1})
    b = _node("b", "linear_dataset", {"inputDim": 2, "outputDim": 1})
    edges = [_edge("ea", "a", "m", "dataset_a"), _edge("eb", "b", "m", "dataset_b")]
    with pytest.raises(HTTPException) as exc:
        dr._sample_dataset_arrays_for_mixer(m, 8, edges, {"m": m, "a": a, "b": b}, "mse_regression", leaf_sampler=_noop_leaf)
    assert exc.value.status_code == 400
    assert exc.value.detail == "dataset_mixer_b interpolationLambda must be in [0, 1]."


def test_mixer_b_child_dataset_mixer_forbidden() -> None:
    mb = _node("mb", "dataset_mixer_b", {"interpolationLambda": 0.5})
    inner = _node("inner", "dataset_mixer", {"proportionA": 0.5})
    b = _node("b", "linear_dataset", {"inputDim": 2, "outputDim": 1})
    edges = [_edge("e1", "inner", "mb", "dataset_a"), _edge("e2", "b", "mb", "dataset_b")]
    with pytest.raises(HTTPException) as exc:
        dr._sample_dataset_arrays_for_mixer(mb, 8, edges, {"mb": mb, "inner": inner, "b": b}, "mse_regression", leaf_sampler=_noop_leaf)
    assert exc.value.status_code == 400
    assert "dataset_mixer_b cannot use dataset_mixer" in exc.value.detail


# Streaming token-arm registry guards.
def test_streaming_token_handlers_key_set() -> None:
    non_toy = {
        NodeKind.memorization_b_dataset, NodeKind.unigram_dataset, NodeKind.token_prediction_dataset,
        NodeKind.bigram_low_rank_dataset, NodeKind.circle_random_walk_dataset, NodeKind.circular_motion_dataset,
        NodeKind.in_context_associative_recall_dataset, NodeKind.modular_addition_dataset,
    }
    assert set(dr.STREAMING_TOKEN_SAMPLE_HANDLERS) == non_toy | set(dr.TOY_LM_BUILDERS)
    # fallback is not a registry key; combinators never in it
    assert NodeKind.dataset_mixer not in dr.STREAMING_TOKEN_SAMPLE_HANDLERS
    assert NodeKind.dataset_mixer_b not in dr.STREAMING_TOKEN_SAMPLE_HANDLERS


def test_dataset_runtime_is_torch_free() -> None:
    import ast

    tree = ast.parse(Path(dr.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(m == "torch" or (m or "").startswith("torch.") for m in imported), "dataset_runtime must stay torch-free"
