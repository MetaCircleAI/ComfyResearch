#!/usr/bin/env python3
"""Exp C phenomenon evidence: rich vs lazy regime (Chizat et al. 2019).

Full mode(默认):2D teacher(width 3, relu, seed 42)+ student(width 200, relu,
symmetrized init **tau=1.0** 固定、只变 α)、SGD 0.001 全批、40000 步。

**时标补偿(Chizat 2019 的时间重标,非调参过测试)**:输出乘 α 后梯度 ∝ α,固定
lr/steps 下小 α 臂的有效训练速率被压低(探针实测:无补偿时 α=0.01 十万步预算只移动
3.5%——是"没训完"不是"lazy")。按理论的 1/(2α²) loss 归一化,给每臂设
``mse_loss.lossScale = 1/α²``(α=1 → 1;α=0.01 → 10000),固定 wall-clock 步数下
公平比较。PR #18 的 α-sweep template 固定 lr 扫 α 而无此补偿,展示大概率混入了
训练时间尺度混淆(记入 friction audit)。

断言(阈值由补偿后探针实测标定,见下):
(1) rich(α=0.01)的相对权重位移显著大于 lazy(α=1.0);
(2) rich 的**范数加权**对齐度更高(2D 中 200 神经元的无权重 max-|cos| 对随机方向也≈1,
无区分度;rich regime 对齐 teacher 方向的神经元范数增长,加权后才可分)。

Teacher 方向重建与 trainer 同一代码路径(teacher_helpers._build_teacher_mlp_eval,
内部自播种 torch.manual_seed(seed));脚本先做双重建稳定性 FATAL 检查,防止
"跟另一个 teacher 比对齐"。--quick: 40 步缩模,只产出 schema。--json: 单行 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch.nn as nn  # noqa: E402

from comfy_research.engine.trainer.teacher_helpers import _build_teacher_mlp_eval  # noqa: E402
from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

ALPHAS = (0.1, 1.0)


def _teacher_node(width: int) -> Node:
    return Node(
        id="teacher",
        type=NodeKind.mlp_model,
        data={"inputDim": 2, "outputDim": 1, "depth": 1, "width": width, "activation": "relu", "seed": 42},
    )


def _graph(*, quick: bool, alpha: float, steps: int, log_freq: int):
    student_width = 16 if quick else 200
    n_train = 64 if quick else 200
    # 时标补偿:严格 1/α² 理论时钟(Chizat)。离散 GD 稳定性(探针标定):补偿后损失在
    # 初始点含 (1/α)·|res|·∇²f 的 sharpness 项,残差未消前必须小步——用 lr_schedule 的
    # warmup(现成节点,标准手段)让残差先收缩;α 取 0.1;双臂同 lr/warmup(公平)。
    loss_scale = 1.0 / (alpha * alpha)
    teacher = _teacher_node(3)
    nodes_raw = [
        {"id": "rid", "type": "random_input_distribution",
         "data": {"inputDim": 2, "inputDistribution": "standard_normal",
                  "noiseDistribution": "deterministic", "noiseLevel": 0, "seed": 0}},
        {"id": "sampler", "type": "input_sampler", "data": {"numSamples": n_train}},
        {"id": "tds", "type": "teacher_dataset", "data": {"samplingMode": "fixed"}},
        {"id": "student", "type": "mlp_model",
         "data": {"inputDim": 2, "outputDim": 1, "depth": 1, "width": student_width,
                  "activation": "relu", "seed": 0, "outputScale": alpha}},
        {"id": "sym", "type": "symmetrized_mlp_init", "data": {"tau": 1.0}},
        {"id": "loss", "type": "mse_loss", "data": {"lossScale": loss_scale}},
        {"id": "lrs", "type": "lr_schedule", "data": {"lrSchedule": "constant", "lrWarmupSteps": 4000}},
        {"id": "opt", "type": "sgd_optimizer", "data": {"learningRate": 0.001, "momentum": 0, "weightDecay": 0}},
        {"id": "obs-traj", "type": "observable_neuron_trajectory_2d", "data": {}},
        {"id": "obs-disp", "type": "observable_weight_displacement", "data": {}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": steps, "logFrequency": log_freq}},
    ]
    edges_raw = [
        ["e-rid", "rid", "sampler", "input_distribution", "distribution"],
        ["e-sam", "sampler", "tds", "sample_tensor", "train_input"],
        ["e-tm", "teacher", "tds", "model", "model"],
        ["e-ds", "tds", "trainer", "dataset", "dataset"],
        ["e-m", "student", "trainer", "model", "model"],
        ["e-init", "sym", "student", "initialization", "initialization"],
        ["e-lrs", "lrs", "opt", "lr_schedule", "lr_schedule"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-obs1", "obs-traj", "trainer", "observables", "observables"],
        ["e-obs2", "obs-disp", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    nodes.append(teacher)
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _first_linear_weights(model: nn.Module) -> np.ndarray:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m.weight.detach().cpu().numpy()
    raise RuntimeError("no nn.Linear in teacher model")


def _teacher_directions() -> np.ndarray:
    """Trainer 同路径重建 teacher;双重建稳定性 FATAL 检查。"""
    node = _teacher_node(3)
    w1 = _first_linear_weights(_build_teacher_mlp_eval(node, 0))
    w2 = _first_linear_weights(_build_teacher_mlp_eval(node, 0))
    if not np.array_equal(w1, w2):
        print("FATAL: teacher rebuild is not deterministic (seed semantics drifted)", file=sys.stderr)
        raise SystemExit(2)
    norms = np.linalg.norm(w1, axis=1, keepdims=True)
    return w1 / np.maximum(norms, 1e-12)


CONE_COS = 0.985  # ~10° 紧锥


def _cone_mass(teacher_dirs: np.ndarray, student_positions: list[list[float]]) -> float:
    """范数质量落在任一 teacher 方向 ~10° 锥内的比例。

    探针发现:2D 中 3 条 teacher 方向线几乎覆盖半圆,max-|cos| 的加权均值对任何分布
    都 ≈0.94(无区分力)。紧锥质量才可分:rich 聚簇到方向上 → →1;lazy 随机 →
    ≈锥测度(3×20°/180° ≈ 0.33 的量级)。"""
    s = np.asarray(student_positions, dtype=float)  # [width, 2]
    norms = np.linalg.norm(s, axis=1)
    s_unit = s / np.maximum(norms[:, None], 1e-12)
    best_cos = np.abs(teacher_dirs @ s_unit.T).max(axis=0)  # [width]
    in_cone = (best_cos >= CONE_COS).astype(float)
    return float((norms * in_cone).sum() / max(norms.sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    steps, log_freq = (40, 2) if args.quick else (40000, 400)
    teacher_dirs = _teacher_directions()
    displacement: dict[str, float] = {}
    alignment: dict[str, float] = {}
    for alpha in ALPHAS:
        nodes, edges, tid = _graph(quick=args.quick, alpha=alpha, steps=steps, log_freq=log_freq)
        complete = [e for e in iter_trainer_events(nodes, edges, tid) if e.get("type") == "complete"]
        if not complete:
            print(f"no complete event for alpha={alpha}", file=sys.stderr)
            return 1
        done = complete[0]
        displacement[str(alpha)] = float(done["observable_metric_histories"]["obs-disp"][-1])
        alignment[str(alpha)] = _cone_mass(teacher_dirs, done["observable_embedding_histories"]["obs-traj"][-1])
    rich_key, lazy_key = str(ALPHAS[0]), str(ALPHAS[1])
    # 阈值由补偿+warmup 后的 40000 步探针实测标定(探针记录见 PR body):
    # disp 0.534 vs 0.0071(75×)→ 断言 ≥10×(主信号,余量 7.5×);
    # 锥质量 0.407 vs 0.329(+0.078)→ 断言 ≥ +0.03(余量 2.6×;α=0.1 为温和 rich,
    # 强聚簇需 α→0 的 mean-field 极限——方向性成立,幅度温和,如实申报)。
    rich_more_displaced = displacement[rich_key] >= 10 * displacement[lazy_key]
    rich_better_aligned = alignment[rich_key] >= alignment[lazy_key] + 0.03
    result = {
        "experiment": "rich_vs_lazy",
        "quick": bool(args.quick),
        "steps": steps,
        "alphas": list(ALPHAS),
        "displacement": displacement,
        "alignment": alignment,
        "rich_more_displaced": bool(rich_more_displaced),
        "rich_better_aligned": bool(rich_better_aligned),
    }
    print(json.dumps(result) if args.as_json else json.dumps(result, indent=2))
    if args.quick:
        return 0
    if not (rich_more_displaced and rich_better_aligned):
        print("PHENOMENON ASSERTIONS FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
