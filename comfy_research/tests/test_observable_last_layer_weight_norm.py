"""observable_last_layer_weight_norm recorder — Slingshot Fig.1 signal.

两锚:(1) 对真实 mlp_model builder 产物,recorder 记录值 == 最后一个严格
``nn.Linear``(即分类头,out_features==output_dim)的 ‖W‖₂;(2) 无 Linear
模型记 NaN。附一条 e2e:6 层 MLP 迷你训练流,history 有限且为正。
"""
from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from comfy_research.engine.models.model_builders import (  # noqa: E402
    ModelBuildContext,
    _build_mlp_model,
)
from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.nodes.definitions.observables.last_layer_weight_norm import (  # noqa: E402
    record,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402


def _record_once(model: nn.Module) -> float:
    rec = SimpleNamespace(model=model, observable_metric_histories=defaultdict(list))
    on = SimpleNamespace(id="obs")
    record(rec, on)  # type: ignore[arg-type]
    hist = rec.observable_metric_histories["obs"]
    assert len(hist) == 1
    return hist[0]


def test_records_classification_head_norm_on_real_mlp_builder() -> None:
    model = _build_mlp_model(
        {"depth": 6, "width": 32, "activation": "relu", "seed": 0},
        ModelBuildContext(input_dim=12, output_dim=10),
    )
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) >= 2
    head = linears[-1]
    assert head.out_features == 10  # “最后一个 Linear”确为分类头
    got = _record_once(model)
    want = float(torch.linalg.vector_norm(head.weight.detach()).item())
    assert got == pytest.approx(want, rel=0, abs=0)
    # 且不是把首层当了尾层
    first = float(torch.linalg.vector_norm(linears[0].weight.detach()).item())
    assert got != pytest.approx(first)


def test_records_nan_when_model_has_no_linear() -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU())
    assert math.isnan(_record_once(model))


def test_norm_layers_with_weight_do_not_shadow_the_head() -> None:
    # LayerNorm 也有 weight;严格 isinstance(nn.Linear) 不得选中它。
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2), nn.LayerNorm(2))
    got = _record_once(model)
    want = float(torch.linalg.vector_norm(model[2].weight.detach()).item())
    assert got == pytest.approx(want, rel=0, abs=0)


def _graph() -> tuple[list[Node], list[Edge], str]:
    nodes_raw: list[dict[str, Any]] = [
        {"id": "dataset", "type": "linear_dataset",
         "data": {"inputDim": 4, "outputDim": 2, "trainSize": 16, "testSize": 8,
                  "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "mlp_model",
         "data": {"inputDim": 4, "outputDim": 2, "depth": 6, "width": 16, "activation": "relu", "seed": 0}},
        {"id": "optimizer", "type": "adam_optimizer", "data": {"learningRate": 0.001}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "obs", "type": "observable_last_layer_weight_norm", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 3, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-ds", "dataset", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-o", "optimizer", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-b", "obs", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def test_e2e_history_is_finite_and_positive() -> None:
    nodes, edges, tid = _graph()
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    assert complete, f"no complete event; got {[e.get('type') for e in events]}"
    hist = complete[0]["observable_metric_histories"]["obs"]
    assert len(hist) >= 3
    assert all(math.isfinite(v) and v > 0 for v in hist), hist
