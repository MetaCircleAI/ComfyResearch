#!/usr/bin/env python3
"""Exp B phenomenon evidence: edge of stability (Cohen et al. 2021).

Full mode(默认):20→32×2→1 tanh MLP、full-batch GD η=0.2(2/η=10)、6000 步。
断言:(1) sharpness λ_max 升穿 0.95·2/η(eos_reached);(2) 穿越前 loss 基本单调降
(monotone_frac ≥ 0.95),穿越后出现振荡(up_frac ≥ 0.15);(3) 穿越后 λ_max 均值
悬停在 2/η ±30% 内。

因果说明(避免误读):progressive sharpening 把 λ_max 推向的是它的**自然上限**
(本任务约 10.5);**只有自然上限高于 2/η 时,2/η 才成为主动的稳定边界**。
η=0.2(2/η=10 < 10.5)→ λ 被截在 10 悬停 + loss 振荡;η=0.1(2/η=20 > 10.5)
→ λ 自然饱和在 ~10.5、不触边、loss 单调——这是 negative control,不是实验失败。
配对运行(λ 在 η=0.1 下冲过 10)证明 η=0.2 的 λ=10 平台是学习率摁住的,非任务内禀。

已知存量行为:λ 曲线首点为 NaN(step-0 记录点 loss 无 grad),crossing 检测天然
跳过。--quick: 40 步缩模,只产出 schema。--json: 单行 JSON。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_research.engine.runs.trainer_run import iter_trainer_events  # noqa: E402
from comfy_research.schemas.graph import Edge, Node, NodeKind  # noqa: E402

LR = 0.2


def _graph(*, quick: bool, steps: int, log_freq: int) -> tuple[list[Node], list[Edge], str]:
    if quick:
        ds = {"inputDim": 3, "outputDim": 1, "trainSize": 32}
        md = {"inputDim": 3, "outputDim": 1, "depth": 1, "width": 4}
    else:
        ds = {"inputDim": 20, "outputDim": 1, "trainSize": 200}
        md = {"inputDim": 20, "outputDim": 1, "depth": 2, "width": 32}
    nodes_raw = [
        {"id": "ds", "type": "linear_dataset",
         "data": {**ds, "inputDistribution": "standard_normal", "outputDistribution": "deterministic",
                  "noiseLevel": 0, "testSize": 0, "seed": 0}},
        {"id": "model", "type": "mlp_model", "data": {**md, "activation": "tanh", "seed": 0}},
        {"id": "opt", "type": "sgd_optimizer", "data": {"learningRate": LR, "momentum": 0, "weightDecay": 0}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "obs-h", "type": "observable_hessian_eigenvalues", "data": {"topK": 1, "order": "descending"}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": -1, "trainingSteps": steps, "logFrequency": log_freq}},
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args()
    steps, log_freq = (40, 2) if args.quick else (6000, 25)
    two_over_lr = 2.0 / LR
    nodes, edges, tid = _graph(quick=args.quick, steps=steps, log_freq=log_freq)
    complete = [e for e in iter_trainer_events(nodes, edges, tid) if e.get("type") == "complete"]
    if not complete:
        print("no complete event", file=sys.stderr)
        return 1
    done = complete[0]
    lam = np.asarray(done["observable_metric_histories"]["obs-h::0"], dtype=float)
    loss = np.asarray(done["loss_history"], dtype=float)
    cross_idx = next(
        (i for i, v in enumerate(lam) if math.isfinite(v) and v >= 0.95 * two_over_lr), None
    )
    pre_frac = post_up = post_mean = None
    if cross_idx is not None and 2 <= cross_idx <= len(loss) - 3:
        pre = np.diff(loss[: cross_idx + 1])
        pre_frac = float((pre <= 1e-12).mean())
        post = np.diff(loss[cross_idx:])
        post_up = float((post > 0).mean())
        post_mean = float(np.nanmean(lam[cross_idx:]))
    eos_reached = cross_idx is not None
    oscillates = post_up is not None and post_up >= 0.15
    result = {
        "experiment": "edge_of_stability", "quick": bool(args.quick), "steps": steps,
        "lr": LR, "two_over_lr": two_over_lr,
        "crossing_step": int(done["step_ticks"][cross_idx]) if cross_idx is not None else None,
        "pre_cross_monotone_frac": pre_frac, "post_cross_up_frac": post_up,
        "post_cross_sharpness_mean": post_mean,
        "eos_reached": bool(eos_reached), "oscillates": bool(oscillates),
        "final_sharpness": float(lam[-1]),
    }
    print(json.dumps(result) if args.as_json else json.dumps(result, indent=2))
    if args.quick:
        return 0
    hover_ok = post_mean is not None and abs(post_mean - two_over_lr) <= 0.30 * two_over_lr
    pre_ok = pre_frac is not None and pre_frac >= 0.95
    if not (eos_reached and oscillates and hover_ok and pre_ok):
        print("PHENOMENON ASSERTIONS FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
