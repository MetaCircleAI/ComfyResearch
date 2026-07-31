"""provider 等价测试。

期望值由**独立重算**给出(直接调 engine builder + 原分支的补形逻辑),与 provider
路径(preview: post_dataset_tensor;materializer: DatasetMaterializeContext)对比。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

pytest.importorskip("torch")

from comfy_research.api.dataset_tensor import DatasetTensorRequest, post_dataset_tensor
from comfy_research.nodes import registry
from comfy_research.nodes.provider_types import DatasetMaterializeContext
from comfy_research.schemas.graph import NodeKind

PDE_TYPES = ("advection_dataset", "diffusion_pde_dataset", "reaction_diffusion_dataset")
# direct-arrays 家族(trio 经 ai4science alias 命中 linear 的 provider)
DIRECT_TYPES = ("linear_dataset", "random_noise_dataset", "memorization_a_dataset", "memorization_b_dataset")
TOY_LANGUAGE_TYPES = (
    "biography_lm_dataset", "cogs_dataset", "dyck_dataset", "formal_language_suite_dataset",
    "listops_dataset", "multi_hop_fact_chain_dataset", "ngram_language_dataset", "pcfg_dataset",
    "phi1_style_dataset", "relation_tuple_dataset", "scan_dataset", "synthetic_playground_dataset",
    "tinystories_dataset",
)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _defaults_for(t: str) -> dict:
    registry.load_definitions()
    return {f.key: f.default for f in registry.DATASET_DEFS[t].fields}


def _expected_arrays(t: str, data: dict):
    """独立重算:engine builder 直调 + 原 preview 分支的补形规则。"""
    if t in PDE_TYPES:
        from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays, pde_field_flat_dims

        rng = np.random.default_rng(int(data.get("initSeed", 0)))
        x_tr, y_tr, x_te, y_te = build_pde_field_arrays(
            NodeKind(t), rng, data, data, int(data["trainSize"]), int(data["testSize"])
        )
        _, _, _, flat = pde_field_flat_dims(data)
        if x_te is None:
            x_te = np.zeros((0, flat), dtype=np.float32)
        if y_te is None:
            y_te = np.zeros((0, flat), dtype=np.float32)
        return x_tr, y_tr, x_te, y_te


def _expected_direct(t: str, data: dict):
    from comfy_research.engine.datasets.dataset_preview_helpers import build_linear_or_symbolic_arrays

    return build_linear_or_symbolic_arrays(t, data)


@pytest.mark.parametrize("t", DIRECT_TYPES)
@pytest.mark.parametrize("split,key", [("train", "input"), ("train", "output"), ("test", "input"), ("test", "output")])
def test_direct_arrays_preview_provider(t: str, split: str, key: str) -> None:
    data = _defaults_for(t)
    if split == "test" and int(data.get("testSize", 0)) <= 0:
        pytest.skip("default testSize=0 — endpoint 400s by design")
    resp = post_dataset_tensor(
        DatasetTensorRequest(dataset_node_id="n1", dataset_node_type=t, dataset_data=data, split=split, tensor_key=key)
    )
    x_tr, y_tr, x_te, y_te = _expected_direct(t, data)
    want_map = {"train": {"input": x_tr, "output": y_tr}, "test": {"input": x_te, "output": y_te}}
    want = np.ascontiguousarray(want_map[split][key], dtype=np.float32)
    got = np.frombuffer(resp.body, dtype=np.float32)
    assert _sha(got) == _sha(want.reshape(-1)), (t, split, key)




def test_linear_materializer_with_test_split_and_ds_test_raw() -> None:
    """ds_test_raw 必须穿透 hook——test_size>0 且 ds_test_raw 非空时
    与 engine 直调等价(xor2d targetKind 走 dd_test 语义)。"""
    from comfy_research.engine.trainer.dataset_materialize import _materialize_linear_like
    from comfy_research.nodes.registry import dataset_defs_materializers

    data = {**_defaults_for("linear_dataset"), "targetKind": "xor2d", "inputDim": 2, "outputDim": 1}
    sentinel = object()
    rng = np.random.default_rng(11)
    arrays = dataset_defs_materializers()["linear_dataset"](
        DatasetMaterializeContext(
            ds_type=NodeKind("linear_dataset"), rng=rng, dd_train=data, dd_test=data,
            train_size=64, test_size=16, input_dim=2, output_dim=1, ds_test_raw=sentinel,
        )
    )
    rng2 = np.random.default_rng(11)
    want = _materialize_linear_like(
        rng2, data, data, sentinel, 64, 16, 2, 1,
        "standard_normal", 0.25, True, "standard_normal", 0.25, True,
    )
    assert _sha(arrays.x_np) == _sha(want.x_np)
    assert _sha(arrays.y_np) == _sha(want.y_np)
    assert _sha(arrays.x_test_np) == _sha(want.x_test_np)
    assert _sha(arrays.y_test_np) == _sha(want.y_test_np)


@pytest.mark.parametrize("t", PDE_TYPES)
@pytest.mark.parametrize("split,key", [("train", "input"), ("train", "output"), ("test", "input"), ("test", "output")])
def test_preview_provider_matches_independent_recompute(t: str, split: str, key: str) -> None:
    data = _defaults_for(t)
    resp = post_dataset_tensor(
        DatasetTensorRequest(
            dataset_node_id="n1", dataset_node_type=t, dataset_data=data,
            split=split, tensor_key=key,
        )
    )
    x_tr, y_tr, x_te, y_te = _expected_arrays(t, data)
    want = {"train": {"input": x_tr, "output": y_tr}, "test": {"input": x_te, "output": y_te}}[split][key]
    want32 = np.ascontiguousarray(want, dtype=np.float32)
    got = np.frombuffer(resp.body, dtype=np.float32)
    assert got.size == want32.size
    assert _sha(got) == _sha(want32.reshape(-1)), (t, split, key)
    # header 元数据也钉住:shape 头回归 sha 检不出来。
    import json as _json

    assert _json.loads(resp.headers["x-tensor-shape"]) == list(want32.shape), (t, split, key)


@pytest.mark.parametrize("t", PDE_TYPES)
def test_materializer_provider_matches_independent_recompute(t: str) -> None:
    """ctx 接线钉死:ds_type 传 NodeKind 原值、dims 透传;输出与 engine builder
    直调等价(shape/dtype/sha256)。"""
    data = _defaults_for(t)
    mat = registry.dataset_defs_materializers()[t]
    rng = np.random.default_rng(7)
    # dims 由调用方(_materialize_dense_families)解析;这里给定值,只验证透传。
    if t in PDE_TYPES:
        from comfy_research.engine.datasets.pde_field_dataset_runtime import pde_field_flat_dims

        _, _, _, flat = pde_field_flat_dims(data)
        input_dim = output_dim = flat
    else:
        input_dim, output_dim = int(data["inputDim"]), int(data["outputDim"])
    arrays = mat(
        DatasetMaterializeContext(
            ds_type=NodeKind(t), rng=rng, dd_train=data, dd_test=data,
            train_size=int(data["trainSize"]), test_size=int(data["testSize"]),
            input_dim=input_dim, output_dim=output_dim,
        )
    )
    rng2 = np.random.default_rng(7)
    if t in PDE_TYPES:
        from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays

        x_tr, y_tr, x_te, y_te = build_pde_field_arrays(
            NodeKind(t), rng2, data, data, int(data["trainSize"]), int(data["testSize"])
        )
    assert arrays.input_dim == input_dim and arrays.output_dim == output_dim
    for got, want in ((arrays.x_np, x_tr), (arrays.y_np, y_tr)):
        assert got.shape == want.shape and got.dtype == want.dtype
        assert _sha(got) == _sha(want)
    if x_te is not None:
        assert _sha(arrays.x_test_np) == _sha(x_te)
        assert _sha(arrays.y_test_np) == _sha(y_te)


def test_materializer_hook_threads_ds_test_raw_source_contract() -> None:
    """hook 若丢 ds_test_raw=ds_test_raw,provider 单测仍绿——源契约钉住。"""
    from pathlib import Path

    src = Path("comfy_research/engine/trainer/dataset_materialize.py").read_text()
    assert "ds_test_raw=ds_test_raw" in src


def test_all_migrated_dataset_defs_anchor_committed_manifest() -> None:
    """全部已迁 dataset 的逐键+键序锚点(参数化总锚,后续批自动覆盖)。"""
    import json
    from pathlib import Path

    from comfy_research.nodes.generate import def_to_entry_dataset

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(Path("comfy_research/generated/node_manifest.json").read_text())}
    assert registry.DATASET_DEFS, "no dataset defs registered"
    for t, d in sorted(registry.DATASET_DEFS.items()):
        got = def_to_entry_dataset(d)
        want = committed[t]
        assert got == want, t
        assert list(got) == list(want), t


@pytest.mark.parametrize("t", TOY_LANGUAGE_TYPES)
@pytest.mark.parametrize("split,key", [("train", "input"), ("train", "output"), ("test", "input"), ("test", "output")])
def test_toy_language_preview_provider(t: str, split: str, key: str) -> None:
    """provider = TOY_LM_BUILDERS builder 直调(literal 分支原体)。"""
    from comfy_research.engine.datasets.dataset_runtime import TOY_LM_BUILDERS

    data = _defaults_for(t)
    resp = post_dataset_tensor(
        DatasetTensorRequest(dataset_node_id="n1", dataset_node_type=t, dataset_data=data, split=split, tensor_key=key)
    )
    x_tr, y_tr, x_te, y_te = TOY_LM_BUILDERS[NodeKind(t)](data, int(data["trainSize"]), int(data["testSize"]))
    want = {"train": {"input": x_tr, "output": y_tr}, "test": {"input": x_te, "output": y_te}}[split][key]
    got = np.frombuffer(resp.body, dtype=np.float32)
    want32 = np.ascontiguousarray(want, dtype=np.float32)
    assert _sha(got) == _sha(want32.reshape(-1)), (t, split, key)
    import json as _json

    assert _json.loads(resp.headers["x-tensor-shape"]) == list(want32.shape), (t, split, key)


@pytest.mark.parametrize("t", TOY_LANGUAGE_TYPES)
def test_toy_language_provider_returns_arrays_not_responses(t: str) -> None:
    """provider 只回 4 数组——word-inspect 归 post-dispatch 块所有。"""
    from comfy_research.nodes.registry import dataset_defs_previews
    from comfy_research.nodes.provider_types import DatasetPreviewRequest

    out = dataset_defs_previews()[t](
        DatasetPreviewRequest(original_type=t, effective_type=t, data={**_defaults_for(t), "inspectFormat": "word"})
    )
    assert isinstance(out, tuple) and len(out) == 4
    assert all(isinstance(a, np.ndarray) for a in out)


def test_toy_language_word_inspect_preserved() -> None:
    """word-inspect 在 post-dispatch 块、capability 再判——provider 迁移后行为不变
    (JSONResponse + X-Dataset-Inspect: word,lines 与独立重算一致)。"""
    import json as _json

    from comfy_research.engine.datasets.toy_language_inspect import toy_language_word_inspect_lines

    t = "scan_dataset"
    data = {**_defaults_for(t), "inspectFormat": "word"}
    resp = post_dataset_tensor(
        DatasetTensorRequest(dataset_node_id="n1", dataset_node_type=t, dataset_data=data, split="train", tensor_key="input")
    )
    assert resp.headers["x-dataset-inspect"] == "word"
    body = _json.loads(resp.body)
    lines, shape_list, _note = toy_language_word_inspect_lines(t, data, "train", "input")
    assert body["lines"] == lines
    assert body["shape"] == shape_list


VISION_TYPES = ("mnist_dataset", "gaussian_blob_dataset", "shape_world_dataset", "hole_counting_dataset")
SINGLETON_PREVIEW_TYPES = (
    "token_prediction_dataset", "bigram_low_rank_dataset", "circle_random_walk_dataset",
    "circular_motion_dataset", "in_context_associative_recall_dataset", "modular_addition_dataset",
    "uniform_linear_motion_dataset",
)
TOKEN_STAY_HAND_TYPES = SINGLETON_PREVIEW_TYPES[:6] + ("unigram_dataset",)


def test_stay_hand_families_have_no_providers() -> None:
    """documented-exception 姿势 guard:
    - toy-language:token 分支共享 ctx-override/text-heavy 回退;
    - vision:branch provider + cross_entropy_dense flatten 任务条件双路;
    两族的任何 materializer provider 都会绕过共享逻辑,FATAL。vision 也无 preview
    provider(独立 gallery API)。"""
    from comfy_research.nodes.registry import dataset_defs_materializers, dataset_defs_previews

    mats = set(dataset_defs_materializers())
    leaked = (set(TOY_LANGUAGE_TYPES) | set(VISION_TYPES)) & mats
    assert not leaked, sorted(leaked)
    vision_previews = set(VISION_TYPES) & set(dataset_defs_previews())
    assert not vision_previews, sorted(vision_previews)


@pytest.mark.parametrize("t", SINGLETON_PREVIEW_TYPES)
@pytest.mark.parametrize("split,key", [("train", "input"), ("train", "output"), ("test", "input"), ("test", "output")])
def test_singleton_preview_provider(t: str, split: str, key: str) -> None:
    """provider = engine 移层 _build_*_arrays(data) 原体。"""
    from comfy_research.engine.datasets import dataset_preview_helpers as h

    fn = {
        "token_prediction_dataset": h._build_token_arrays,
        "bigram_low_rank_dataset": h._build_bigram_low_rank_arrays,
        "circle_random_walk_dataset": h._build_circle_random_walk_arrays,
        "circular_motion_dataset": h._build_circular_motion_arrays,
        "in_context_associative_recall_dataset": h._build_associative_recall_arrays,
        "modular_addition_dataset": h._build_modular_addition_arrays,
        "uniform_linear_motion_dataset": h._build_uniform_linear_motion_arrays,
    }[t]
    data = _defaults_for(t)
    if split == "test" and int(data.get("testSize", 0)) <= 0 and "testSize" in data:
        pytest.skip("default testSize=0")
    resp = post_dataset_tensor(
        DatasetTensorRequest(dataset_node_id="n1", dataset_node_type=t, dataset_data=data, split=split, tensor_key=key)
    )
    x_tr, y_tr, x_te, y_te = fn(data)
    pick = {"train": {"input": x_tr, "output": y_tr}, "test": {"input": x_te, "output": y_te}}[split][key]
    want = np.ascontiguousarray(pick, dtype=np.float32)
    got = np.frombuffer(resp.body, dtype=np.float32)
    assert _sha(got) == _sha(want.reshape(-1)), (t, split, key)
    import json as _json

    assert _json.loads(resp.headers["x-tensor-shape"]) == list(want.shape), (t, split, key)


@pytest.mark.parametrize("t", ("kepler_2d_dataset", "uniform_linear_motion_dataset", "symbolic_func_dataset"))
def test_regression_singleton_materializer(t: str) -> None:
    """regression 三型 materializer = arm 原体(sha256 vs engine 直调)。"""
    from comfy_research.nodes.registry import dataset_defs_materializers

    data = _defaults_for(t)
    rng = np.random.default_rng(5)
    dims = (int(data.get("inputDim", 2)), int(data.get("outputDim", 1)))
    arrays = dataset_defs_materializers()[t](
        DatasetMaterializeContext(
            ds_type=NodeKind(t), rng=rng, dd_train=data, dd_test=data,
            train_size=32, test_size=8, input_dim=dims[0], output_dim=dims[1], ds_test_raw=None,
        )
    )
    rng2 = np.random.default_rng(5)
    if t == "kepler_2d_dataset":
        from comfy_research.engine.datasets.synthetic_dataset_builders import _build_kepler_2d_arrays

        x, y, xt, yt = _build_kepler_2d_arrays(rng2, data, data, 32, 8)
    elif t == "uniform_linear_motion_dataset":
        from comfy_research.engine.datasets.synthetic_dataset_builders import _build_uniform_linear_motion_arrays

        x, y, xt, yt = _build_uniform_linear_motion_arrays(rng2, data, data, 32, 8)
    else:
        from comfy_research.engine.trainer.dataset_materialize import _materialize_symbolic_func

        want = _materialize_symbolic_func(
            rng2, data, data, None, 32, 8, dims[0], dims[1],
            "standard_normal", 0.0, False, "standard_normal", 0.0, False,
        )
        x, y, xt, yt = want.x_np, want.y_np, want.x_test_np, want.y_test_np
    assert _sha(arrays.x_np) == _sha(np.ascontiguousarray(x))
    assert _sha(arrays.y_np) == _sha(np.ascontiguousarray(y))
    if xt is not None:
        assert _sha(arrays.x_test_np) == _sha(np.ascontiguousarray(xt))
        assert _sha(arrays.y_test_np) == _sha(np.ascontiguousarray(yt))


def test_token_singletons_stay_hand_and_crl_has_no_providers() -> None:
    """token 分支无 hook 且不加(toy-language 姿势);crl 双缺口 by design。"""
    from comfy_research.nodes.registry import dataset_defs_materializers, dataset_defs_previews

    mats = set(dataset_defs_materializers())
    assert not (set(TOKEN_STAY_HAND_TYPES) & mats), sorted(set(TOKEN_STAY_HAND_TYPES) & mats)
    assert "crl_env_config" not in mats
    assert "crl_env_config" not in set(dataset_defs_previews())


def test_teacher_preview_provider_four_arrays() -> None:
    """teacher preview provider ≡ engine 移层体,四数组 sha256,
    graph fixture 含 model + train_input + test_input。"""
    from comfy_research.engine.datasets.dataset_preview_helpers import _build_teacher_arrays

    nodes = [
        {"id": "teach", "type": "teacher_dataset", "data": {"samplingMode": "fixed"}},
        {"id": "tm", "type": "mlp_model", "data": {"inputDim": 4, "outputDim": 2, "depth": 1, "width": 8, "activation": "relu", "seed": 3}},
        {"id": "sam", "type": "input_sampler", "data": {"numSamples": 64}},
        {"id": "sam2", "type": "input_sampler", "data": {"numSamples": 16}},
        {"id": "rid", "type": "random_input_distribution", "data": {"inputDim": 4, "inputDistribution": "standard_normal", "noiseDistribution": "deterministic", "noiseLevel": 0, "seed": 3}},
    ]
    edges = [
        {"id": "e1", "source": "tm", "target": "teach", "sourceHandle": "model", "targetHandle": "model"},
        {"id": "e2", "source": "sam", "target": "teach", "sourceHandle": "sample_tensor", "targetHandle": "train_input"},
        {"id": "e3", "source": "sam2", "target": "teach", "sourceHandle": "sample_tensor", "targetHandle": "test_input"},
        {"id": "e4", "source": "rid", "target": "sam", "sourceHandle": "input_distribution", "targetHandle": "distribution"},
        {"id": "e5", "source": "rid", "target": "sam2", "sourceHandle": "input_distribution", "targetHandle": "distribution"},
    ]
    data = {"samplingMode": "fixed"}
    want = _build_teacher_arrays("teach", data, nodes, edges)
    for split, key, idx in [("train", "input", 0), ("train", "output", 1), ("test", "input", 2), ("test", "output", 3)]:
        resp = post_dataset_tensor(
            DatasetTensorRequest(
                dataset_node_id="teach", dataset_node_type="teacher_dataset", dataset_data=data,
                graph_nodes=nodes, graph_edges=edges, split=split, tensor_key=key,
            )
        )
        got = np.frombuffer(resp.body, dtype=np.float32)
        w = np.ascontiguousarray(want[idx], dtype=np.float32)
        assert _sha(got) == _sha(w.reshape(-1)), (split, key)
