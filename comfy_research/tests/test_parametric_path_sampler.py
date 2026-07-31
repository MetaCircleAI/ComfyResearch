"""Smoke tests for parametric_path_sampler backend."""

from __future__ import annotations

import base64
import io

import pytest
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.analysis import parametric_path_sampler as sampler_module
from comfy_research.api.parametric_path_sampler import _sampler_prefers_remote_gpu
from comfy_research.engine.analysis.parametric_path_sampler import (
    _accuracy_drop_from_path,
    _blend_state_dicts,
    _recompute_batch_norm_stats,
    run_parametric_path_sampler,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _pack_state(model: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save({"model": model.state_dict()}, buf)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _tiny_graph(*, sb_b64: str, lb_b64: str) -> tuple[list[Node], list[Edge], str]:
    nodes = [
        Node(id="ds", type=NodeKind.linear_dataset, data={"inputDim": 4, "outputDim": 2, "trainSize": 32, "testSize": 16}),
        Node(id="mlp", type=NodeKind.mlp_model, data={"inputDim": 4, "outputDim": 2, "depth": 1, "width": 8, "seed": 0}),
        Node(id="ce", type=NodeKind.mse_loss, data={}),
        Node(id="ck_sb", type=NodeKind.model_checkpoint, data={"memoryCheckpoint_b64": sb_b64}),
        Node(id="ck_lb", type=NodeKind.model_checkpoint, data={"memoryCheckpoint_b64": lb_b64}),
        Node(
            id="sampler",
            type=NodeKind.parametric_path_sampler,
            data={"alphaMin": 0.0, "alphaMax": 1.0, "alphaSteps": 3, "metric": "loss", "split": "train"},
        ),
    ]
    edges = [
        Edge(id="e1", source="ck_sb", target="sampler", sourceHandle="model", targetHandle="checkpoint_sb"),
        Edge(id="e2", source="ck_lb", target="sampler", sourceHandle="model", targetHandle="checkpoint_lb"),
        Edge(id="e3", source="mlp", target="sampler", sourceHandle="model", targetHandle="model"),
        Edge(id="e4", source="ds", target="sampler", sourceHandle="dataset", targetHandle="dataset"),
        Edge(id="e5", source="ce", target="sampler", sourceHandle="loss", targetHandle="loss"),
    ]
    return nodes, edges, "sampler"


def test_parametric_path_sampler_smoke() -> None:
    m0 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        for p in m1.parameters():
            p.add_(0.1)
    sb = _pack_state(m0)
    lb = _pack_state(m1)
    nodes, edges, sid = _tiny_graph(sb_b64=sb, lb_b64=lb)
    sampler = next(node for node in nodes if node.id == sid)
    sampler.data = {**(sampler.data or {}), "interpolationMode": "parameters_only"}
    out = run_parametric_path_sampler(nodes, edges, sid)
    assert len(out["alphaSeries"]) == 3
    assert len(out["series"]) == 4
    for s in out["series"]:
        assert len(s["values"]) == 3
        assert all(isinstance(v, float) for v in s["values"])
    assert out["series"][0]["metricId"] == "train_loss"
    assert "trainable-parameter interpolation" in out["summary"]


def test_parametric_interpolation_keeps_lb_batch_norm_buffers() -> None:
    sb = nn.BatchNorm1d(2)
    lb = nn.BatchNorm1d(2)
    with torch.no_grad():
        sb.weight.fill_(2.0)
        lb.weight.fill_(4.0)
        sb.running_mean.fill_(2.0)
        lb.running_mean.fill_(4.0)
        sb.running_var.fill_(8.0)
        lb.running_var.fill_(1.0)

    blended = _blend_state_dicts(
        sb.state_dict(),
        lb.state_dict(),
        2.0,
        trainable_keys={name for name, param in sb.named_parameters() if param.requires_grad},
    )

    torch.testing.assert_close(blended["weight"], torch.full((2,), 6.0))
    torch.testing.assert_close(blended["running_mean"], lb.running_mean)
    torch.testing.assert_close(blended["running_var"], lb.running_var)
    torch.testing.assert_close(blended["num_batches_tracked"], lb.num_batches_tracked)


def test_batch_norm_recalibration_resets_running_statistics_without_changing_weights() -> None:
    model = nn.Sequential(nn.BatchNorm1d(2))
    batch_norm = model[0]
    with torch.no_grad():
        batch_norm.weight.copy_(torch.tensor([2.0, 3.0]))
        batch_norm.bias.copy_(torch.tensor([4.0, 5.0]))
        batch_norm.running_mean.fill_(99.0)
        batch_norm.running_var.fill_(77.0)
        batch_norm.num_batches_tracked.fill_(42)

    _recompute_batch_norm_stats(model, torch.zeros(8, 2), batch_size=4, max_batches=2)

    torch.testing.assert_close(batch_norm.weight, torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(batch_norm.bias, torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(batch_norm.running_mean, torch.zeros(2))
    torch.testing.assert_close(batch_norm.running_var, torch.full((2,), 0.81))
    assert batch_norm.num_batches_tracked.item() == 2


def test_accuracy_drop_is_a_non_negative_dip_below_endpoints() -> None:
    assert _accuracy_drop_from_path([0.9, 0.6, 0.8]) == pytest.approx(0.2)
    assert _accuracy_drop_from_path([0.6, 0.8, 0.9]) == 0.0


def test_parametric_sampler_requests_paper_batch_norm_protocol(monkeypatch) -> None:
    m0 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    nodes, edges, sid = _tiny_graph(sb_b64=_pack_state(m0), lb_b64=_pack_state(m1))
    batch_norm_modes: list[bool] = []

    def fake_loss(*_args, **kwargs) -> float:
        batch_norm_modes.append(bool(kwargs.get("batch_norm_batch_stats", False)))
        return 0.0

    def fake_accuracy(*_args, **kwargs) -> float:
        batch_norm_modes.append(bool(kwargs.get("batch_norm_batch_stats", False)))
        return 1.0

    monkeypatch.setattr(sampler_module, "_batched_primary_loss_mean", fake_loss)
    monkeypatch.setattr(sampler_module, "_batched_classification_accuracy", fake_accuracy)

    out = run_parametric_path_sampler(nodes, edges, sid)

    assert sampler_module._PARAM_PATH_EVAL_BATCH == 5000
    assert batch_norm_modes and all(batch_norm_modes)
    assert "per-batch BatchNorm statistics" in out["summary"]


def test_linear_interpolation_barrier_returns_metrics_and_png() -> None:
    m0 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        for parameter in m1.parameters():
            parameter.add_(0.1)
    nodes, edges, sid = _tiny_graph(sb_b64=_pack_state(m0), lb_b64=_pack_state(m1))
    sampler = next(node for node in nodes if node.id == sid)
    sampler.type = NodeKind.observable_linear_interpolation_barrier
    sampler.data = {
        "alphaMin": 0.0,
        "alphaMax": 1.0,
        "alphaSteps": 3,
        "recomputeBnStats": False,
        "evalBatchSize": 16,
    }
    for edge in edges:
        if edge.targetHandle == "checkpoint_sb":
            edge.targetHandle = "checkpoint_a"
        elif edge.targetHandle == "checkpoint_lb":
            edge.targetHandle = "checkpoint_b"

    out = run_parametric_path_sampler(nodes, edges, sid)

    assert [series["metricId"] for series in out["series"]] == [
        "train_loss", "test_loss", "train_acc", "test_acc",
    ]
    assert len(out["alphaSeries"]) == 3
    assert out["alpha"] == out["alphaSeries"]
    assert out["train_loss"] == out["series"][0]["values"]
    assert out["test_acc"] == out["series"][3]["values"]
    assert isinstance(out["lossBarrier"], float)
    assert out["loss_barrier"] == out["lossBarrier"]
    assert isinstance(out["accuracyDrop"], float)
    assert isinstance(out["interpolationCurvePng"], str)
    assert len(out["interpolationCurvePng"]) > 100


def test_linear_interpolation_uses_a_b_checkpoint_labels_in_errors() -> None:
    m0 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    nodes, edges, sid = _tiny_graph(sb_b64=_pack_state(m0), lb_b64=_pack_state(m0))
    sampler = next(node for node in nodes if node.id == sid)
    sampler.type = NodeKind.observable_linear_interpolation_barrier
    for edge in edges:
        if edge.targetHandle == "checkpoint_sb":
            edge.targetHandle = "checkpoint_a"
        elif edge.targetHandle == "checkpoint_lb":
            edge.targetHandle = "checkpoint_b"
    edges = [edge for edge in edges if edge.targetHandle != "checkpoint_b"]

    with pytest.raises(HTTPException, match="checkpoint A and B"):
        run_parametric_path_sampler(nodes, edges, sid)


def test_bezier_mode_connectivity_optimizes_a_control_point_and_returns_both_paths() -> None:
    m0 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    m1 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    with torch.no_grad():
        for parameter in m1.parameters():
            parameter.add_(0.1)
    nodes, edges, sid = _tiny_graph(sb_b64=_pack_state(m0), lb_b64=_pack_state(m1))
    sampler = next(node for node in nodes if node.id == sid)
    sampler.type = NodeKind.observable_bezier_mode_connectivity
    sampler.data = {
        "alphaSteps": 3,
        "curveOptimizationSteps": 2,
        "curveSamplesPerStep": 1,
        "curveBatchSize": 8,
        "curveLearningRate": 0.01,
        "recomputeBnStats": False,
        "evalBatchSize": 16,
    }
    for edge in edges:
        if edge.targetHandle == "checkpoint_sb":
            edge.targetHandle = "checkpoint_a"
        elif edge.targetHandle == "checkpoint_lb":
            edge.targetHandle = "checkpoint_b"

    out = run_parametric_path_sampler(nodes, edges, sid)

    assert out["alphaSeries"] == [0.0, 0.5, 1.0]
    assert len(out["linearTestLoss"]) == 3
    assert len(out["bezierTestLoss"]) == 3
    assert out["linearTestLoss"][0] == pytest.approx(out["bezierTestLoss"][0])
    assert out["linearTestLoss"][-1] == pytest.approx(out["bezierTestLoss"][-1])
    assert isinstance(out["linearLossBarrier"], float)
    assert isinstance(out["bezierLossBarrier"], float)


def test_reserved_temp_node_id_collision_rejected() -> None:
    """Preexisting graph ids colliding with injected temp trainer/opt ids must 400."""
    nodes, edges, sid = _tiny_graph(sb_b64="", lb_b64="")
    nodes.append(
        Node(id=f"{sid}::__param_path_trainer", type=NodeKind.trainer, data={"trainingSteps": 1})
    )
    with pytest.raises(HTTPException) as ei:
        run_parametric_path_sampler(nodes, edges, sid)
    assert ei.value.status_code == 400
    assert "reserved" in str(ei.value.detail)


def test_reserved_id_edge_reference_rejected() -> None:
    """Preexisting edges targeting the reserved temp trainer id must 400 (they would
    otherwise remain in run_edges and feed/duplicate the injected temp trainer)."""
    nodes, edges, sid = _tiny_graph(sb_b64="", lb_b64="")
    edges.append(
        Edge(
            id="user-edge",
            source="ds",
            sourceHandle="dataset",
            target=f"{sid}::__param_path_trainer",
            targetHandle="dataset",
        )
    )
    with pytest.raises(HTTPException) as ei:
        run_parametric_path_sampler(nodes, edges, sid)
    assert ei.value.status_code == 400
    assert "reserved" in str(ei.value.detail)


def test_sampler_prefers_remote_gpu() -> None:
    local = Node(id="s", type=NodeKind.parametric_path_sampler, data={"computeDevice": "cuda:0", "remoteGpu": False})
    remote = Node(id="s", type=NodeKind.parametric_path_sampler, data={"computeDevice": "cuda", "remoteGpu": True})
    cpu = Node(id="s", type=NodeKind.parametric_path_sampler, data={"computeDevice": "cpu", "remoteGpu": True})
    assert _sampler_prefers_remote_gpu(local) is False
    assert _sampler_prefers_remote_gpu(remote) is True
    assert _sampler_prefers_remote_gpu(cpu) is False


def test_registered_ssh_kill_raises_not_false(monkeypatch) -> None:
    """abort 杀掉注册型 ssh 子进程(returncode<0)必须外抛,
    不得被 _remote_bundle_is_current 折叠为 False 后继续 bootstrap。"""
    import pytest
    from fastapi import HTTPException

    from comfy_research.remote import ssh as rssh

    class _KilledProc:
        returncode = -15
        def communicate(self, input=None, timeout=None):
            return b"", b""

    monkeypatch.setattr(rssh.subprocess, "Popen", lambda *a, **k: _KilledProc())
    with pytest.raises(HTTPException, match="aborted during ssh"):
        rssh._run_ssh_subprocess_registered(["ssh"], trainer_node_id="t1")
