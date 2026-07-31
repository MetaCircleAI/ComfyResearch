"""Exp C (lazy vs rich): CI 短训结构 + 三个 mutation(outputScale/去 init/tau)。

句柄按 template d2e8cdd7 的真实边表固化(rid→sampler: input_distribution→distribution;
sampler→teacher_dataset: sample_tensor→train_input)。现象级验证在
scripts/phenomena/rich_vs_lazy.py。
"""
from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

WIDTH = 16


def _graph(
    *, alpha: float = 0.01, tau: float | None = 0.01, steps: int = 4,
    batch_size: int = -1,
):
    nodes_raw: list[dict[str, Any]] = [
        {"id": "rid", "type": "random_input_distribution",
         "data": {"inputDim": 2, "inputDistribution": "standard_normal",
                  "noiseDistribution": "deterministic", "noiseLevel": 0, "seed": 0}},
        {"id": "sampler", "type": "input_sampler", "data": {"numSamples": 64}},
        {"id": "teacher", "type": "mlp_model",
         "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": 3, "activation": "relu", "seed": 42}},
        {"id": "tds", "type": "teacher_dataset", "data": {"samplingMode": "fixed"}},
        {"id": "student", "type": "mlp_model",
         "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": WIDTH, "activation": "relu",
                  "seed": 0, "outputScale": alpha}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "opt", "type": "sgd_optimizer", "data": {"learningRate": 0.01, "momentum": 0, "weightDecay": 0}},
        {"id": "obs-traj", "type": "observable_neuron_trajectory_2d", "data": {}},
        {"id": "obs-disp", "type": "observable_weight_displacement", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": batch_size, "trainingSteps": steps, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-rid", "rid", "sampler", "input_distribution", "distribution"],
        ["e-sam", "sampler", "tds", "sample_tensor", "train_input"],
        ["e-tm", "teacher", "tds", "model", "model"],
        ["e-ds", "tds", "trainer", "dataset", "dataset"],
        ["e-m", "student", "trainer", "model", "model"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-obs1", "obs-traj", "trainer", "observables", "observables"],
        ["e-obs2", "obs-disp", "trainer", "observables", "observables"],
    ]
    if tau is not None:
        nodes_raw.append({"id": "sym", "type": "symmetrized_mlp_init", "data": {"tau": tau}})
        edges_raw.append(["e-init", "sym", "student", "initialization", "initialization"])
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _run(**kw) -> dict[str, Any]:
    nodes, edges, tid = _graph(**kw)
    events = list(iter_trainer_events(nodes, edges, tid))
    complete = [e for e in events if e.get("type") == "complete"]
    assert complete, f"no complete event; got {[e.get('type') for e in events]}"
    return complete[0]


def test_structure_traj_and_displacement() -> None:
    done = _run()
    traj = done["observable_embedding_histories"]["obs-traj"]
    assert len(traj) == 5
    for snap in traj:
        assert len(snap) == WIDTH and len(snap[0]) == 2, (len(snap), len(snap[0]))
    disp = done["observable_metric_histories"]["obs-disp"]
    assert len(disp) == 5
    assert disp[0] == pytest.approx(0.0, abs=1e-9)
    assert all(math.isfinite(v) for v in disp)
    assert disp[-1] >= disp[0]


def test_fixed_teacher_dataset_uses_wired_sampler_size_for_minibatches() -> None:
    """A fixed teacher dataset may have a different size than the 800-row default."""
    done = _run(batch_size=1)
    assert len(done["loss_history"]) == 5


def _max_traj_delta(a, b) -> float:
    return max(
        abs(x - y)
        for sa, sb in zip(a, b)
        for ra, rb in zip(sa, sb)
        for x, y in zip(ra, rb)
    )


def test_output_scale_mutation() -> None:
    """α 改回 1.0 后，轨迹和 loss 须有量级差异。"""
    rich = _run(alpha=0.01)
    lazy = _run(alpha=1.0)
    assert _max_traj_delta(
        rich["observable_embedding_histories"]["obs-traj"][1:],
        lazy["observable_embedding_histories"]["obs-traj"][1:],
    ) > 1e-6
    assert max(abs(a - b) for a, b in zip(rich["loss_history"], lazy["loss_history"])) > 1e-9


def test_symmetrized_init_mutations() -> None:
    """移除 init / 改 tau 均须显著改变初始神经元位置。"""
    with_init = _run(tau=0.01)["observable_embedding_histories"]["obs-traj"][0]
    without = _run(tau=None)["observable_embedding_histories"]["obs-traj"][0]
    tau_big = _run(tau=1.0)["observable_embedding_histories"]["obs-traj"][0]
    assert _max_traj_delta([with_init], [without]) > 1e-6
    assert _max_traj_delta([with_init], [tau_big]) > 1e-6
    # 对称性:后半神经元第一层权重 = 前半(symmetrized 的可观测指纹)
    half = WIDTH // 2
    for j in range(half):
        assert with_init[j] == pytest.approx(with_init[half + j]), j


def test_symmetrized_init_rejects_non_mlp_model() -> None:
    """symmetrized_mlp_init 仅适用于 mlp_model，防止静默修改其他模型。"""
    from fastapi import HTTPException

    from comfy_research.engine.runs.trainer_run import prepare_trainer_run

    nodes_raw: list[dict[str, Any]] = [
        {"id": "ds", "type": "token_prediction_dataset",
         "data": {"vocabSize": 5, "contextLength": 3, "whichToken": -1, "trainSize": 8,
                  "testSize": 0, "seed": 0, "samplingMode": "fixed"}},
        {"id": "model", "type": "mlp_token_model",
         "data": {"vocabSize": 5, "embedDim": 4, "tokensPerInput": 3, "depth": 1, "width": 6,
                  "activation": "relu", "tieWeights": "no", "seed": 0}},
        {"id": "sym", "type": "symmetrized_mlp_init", "data": {"tau": 1.0}},
        {"id": "opt", "type": "adam_optimizer", "data": {"learningRate": 0.01}},
        {"id": "loss", "type": "cross_entropy_loss", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": 2, "logFrequency": 1}},
    ]
    edges_raw = [
        ["e-ds", "ds", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-init", "sym", "model", "initialization", "initialization"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    with pytest.raises(HTTPException) as exc:
        prepare_trainer_run(nodes, edges, "trainer")
    assert exc.value.status_code == 400
    assert "symmetrized_mlp_init requires an mlp_model" in str(exc.value.detail)
