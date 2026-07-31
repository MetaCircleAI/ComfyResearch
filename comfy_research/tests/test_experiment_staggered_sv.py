"""Exp A (staggered SV): CI 短训结构与突变敏感性测试。

自建图,不引用 template JSON 的 node-id;现象级验证在 scripts/phenomena/staggered_sv.py。
"""
from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


def _graph(*, amplitude: float | None = 0.2, steps: int = 4) -> tuple[list[Node], list[Edge], str]:
    nodes_raw: list[dict[str, Any]] = [
        {"id": "ds", "type": "linear_dataset",
         "data": {"inputDim": 8, "outputDim": 8, "inputDistribution": "standard_normal",
                  "outputDistribution": "additive_gaussian", "noiseLevel": 0,
                  "trainSize": 64, "testSize": 0, "seed": 42}},
        {"id": "model", "type": "mlp_model",
         "data": {"inputDim": 8, "outputDim": 8, "depth": 2, "width": 8,
                  "activation": "identity", "seed": 7}},
        {"id": "opt", "type": "sgd_optimizer",
         "data": {"learningRate": 0.005, "momentum": 0, "weightDecay": 0}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "obs-sv", "type": "observable_weight_product_sv", "data": {"topK": 4}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": steps, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-ds", "ds", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-obs", "obs-sv", "trainer", "observables", "observables"],
    ]
    if amplitude is not None:
        nodes_raw.append({"id": "saxe", "type": "saxe_initialization", "data": {"amplitude": amplitude}})
        edges_raw.append(["e-init", "saxe", "model", "initialization", "initialization"])
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _run(**kw) -> dict[str, Any]:
    nodes, edges, tid = _graph(**kw)
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    assert complete, f"no complete event; got {[e.get('type') for e in events]}"
    return complete[0]


def test_sv_observable_shapes_and_finiteness() -> None:
    done = _run()
    hists = done["observable_metric_histories"]
    assert len(hists["obs-sv"]) == 5  # step 0..4, logFrequency=1
    for i in range(4):
        row = hists[f"obs-sv::{i}"]
        assert len(row) == 5, (i, row)
        assert all(math.isfinite(v) for v in row), (i, row)
    # 每个 log step 上 sv_i 应为降序
    for t in range(5):
        col = [hists[f"obs-sv::{i}"][t] for i in range(4)]
        assert col == sorted(col, reverse=True), (t, col)


def test_saxe_amplitude_mutation_changes_sv() -> None:
    a = _run(amplitude=0.2)["observable_metric_histories"]
    b = _run(amplitude=1.0)["observable_metric_histories"]
    assert a["obs-sv::0"] != b["obs-sv::0"]


def test_removing_saxe_changes_sv() -> None:
    """禁用 saxe 后 SV 轨迹必须显著不同，证明 init node 已连接。"""
    with_saxe = _run(amplitude=0.2)["observable_metric_histories"]
    without = _run(amplitude=None)["observable_metric_histories"]
    assert with_saxe["obs-sv::0"] != without["obs-sv::0"]
    # saxe(ε=0.2, 正交)的初始有效积 top SV ≈ ε^(层数) 量级,显著小于默认 init
    assert with_saxe["obs-sv::0"][0] < without["obs-sv::0"][0]
