"""Exp B (edge of stability): CI 短训结构 + η 突变敏感性。现象级验证在 scripts/phenomena/edge_of_stability.py。"""
from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


def _graph(*, lr: float = 0.2, steps: int = 4) -> tuple[list[Node], list[Edge], str]:
    nodes_raw: list[dict[str, Any]] = [
        {"id": "ds", "type": "linear_dataset",
         "data": {"inputDim": 3, "outputDim": 1, "inputDistribution": "standard_normal",
                  "outputDistribution": "deterministic", "noiseLevel": 0,
                  "trainSize": 32, "testSize": 0, "seed": 0}},
        {"id": "model", "type": "mlp_model",
         "data": {"inputDim": 3, "outputDim": 1, "depth": 1, "width": 4,
                  "activation": "tanh", "seed": 0}},
        {"id": "opt", "type": "sgd_optimizer",
         "data": {"learningRate": lr, "momentum": 0, "weightDecay": 0}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "obs-h", "type": "observable_hessian_eigenvalues",
         "data": {"topK": 1, "order": "descending"}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": steps, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-ds", "ds", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-obs", "obs-h", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _run(**kw) -> dict[str, Any]:
    nodes, edges, tid = _graph(**kw)
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    assert complete, f"no complete event; got {[e.get('type') for e in events]}"
    return complete[0]


def test_sharpness_history_shape_and_finiteness() -> None:
    done = _run()
    hists = done["observable_metric_histories"]
    lam = hists["obs-h::0"]
    assert len(lam) == 5, lam
    # 存量行为:step-0 的记录点 loss 不带 grad,hessian 分支把 RuntimeError 吞成 NaN
    # (recorder._record_hessian_eigenvalues 的 except 兜底)。step 1+ 必须有限。
    assert math.isnan(lam[0]), lam
    assert all(math.isfinite(v) for v in lam[1:]), lam
    assert len(done["loss_history"]) == 5


def test_lr_mutation_changes_sharpness_trajectory() -> None:
    """η=0.2 与 η=0.002 的 λ_max/loss 轨迹必须有量级差异(list != 是弱断言)。"""
    hi_run = _run(lr=0.2)
    lo_run = _run(lr=0.002)
    hi = hi_run["observable_metric_histories"]["obs-h::0"]
    lo = lo_run["observable_metric_histories"]["obs-h::0"]
    # step-0 双方都是 NaN(见上);从 step 1 起比较量级差异
    assert math.isnan(hi[0]) and math.isnan(lo[0]), (hi[0], lo[0])
    assert max(abs(a - b) for a, b in zip(hi[1:], lo[1:])) > 1e-6, (hi, lo)
    hl = hi_run["loss_history"]
    ll = lo_run["loss_history"]
    assert max(abs(a - b) for a, b in zip(hl[1:], ll[1:])) > 1e-6, (hl, ll)
