#!/usr/bin/env python3
"""Exp A phenomenon evidence: staggered singular-value learning (Saxe et al. 2013).

Full mode(默认):template-A 设置(8 维深线性网、saxe ε=0.2、SGD lr=5e-3、8000 步),
断言 (1) top-k SV 的半高激活步严格递增(staggered);(2) 每条曲线 logit 线性化拟合
R² ≥ 0.9(sigmoid 形)。断言失败 exit 1。

--quick: 40 步小跑,只产出 schema(CI 防腐用,不断言现象)。--json: 单行 JSON 输出。
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

TOP_K = 4


def _graph(steps: int, log_freq: int) -> tuple[list[Node], list[Edge], str]:
    nodes_raw = [
        {"id": "ds", "type": "linear_dataset",
         "data": {"inputDim": 8, "outputDim": 8, "inputDistribution": "standard_normal",
                  "outputDistribution": "additive_gaussian", "noiseLevel": 0,
                  "trainSize": 2000, "testSize": 0, "seed": 42}},
        {"id": "model", "type": "mlp_model",
         "data": {"inputDim": 8, "outputDim": 8, "depth": 2, "width": 8,
                  "activation": "identity", "seed": 7}},
        {"id": "saxe", "type": "saxe_initialization", "data": {"amplitude": 0.2}},
        {"id": "opt", "type": "sgd_optimizer",
         "data": {"learningRate": 0.005, "momentum": 0, "weightDecay": 0}},
        {"id": "loss", "type": "mse_loss", "data": {}},
        {"id": "obs-sv", "type": "observable_weight_product_sv", "data": {"topK": TOP_K}},
        {"id": "trainer", "type": "trainer",
         "data": {"computeDevice": "cpu", "batchSize": 256, "trainingSteps": steps,
                  "logFrequency": log_freq}},
    ]
    edges_raw = [
        ["e-ds", "ds", "trainer", "dataset", "dataset"],
        ["e-m", "model", "trainer", "model", "model"],
        ["e-init", "saxe", "model", "initialization", "initialization"],
        ["e-o", "opt", "trainer", "optimizer", "optimizer"],
        ["e-l", "loss", "trainer", "loss", "loss"],
        ["e-obs", "obs-sv", "trainer", "observables", "observables"],
    ]
    nodes = [Node(id=n["id"], type=NodeKind(n["type"]), data=dict(n["data"])) for n in nodes_raw]
    edges = [Edge(id=e[0], source=e[1], target=e[2], sourceHandle=e[3], targetHandle=e[4]) for e in edges_raw]
    return nodes, edges, "trainer"


def _half_max_step(curve: np.ndarray, ticks: np.ndarray) -> int | None:
    lo, hi = float(curve[0]), float(curve[-1])
    if not math.isfinite(hi) or hi - lo <= 1e-9:
        return None
    thresh = lo + 0.5 * (hi - lo)
    idx = int(np.argmax(curve >= thresh))
    return int(ticks[idx])


def _logit_r2(curve: np.ndarray, ticks: np.ndarray) -> float | None:
    """Sigmoid-shape score: logistic curves are linear in logit space (R² of the linear fit)."""
    lo, hi = float(curve.min()), float(curve.max())
    if hi - lo <= 1e-9:
        return None
    y = np.clip((curve - lo) / (hi - lo), 1e-4, 1 - 1e-4)
    z = np.log(y / (1 - y))
    coef = np.polyfit(ticks, z, 1)
    pred = np.polyval(coef, ticks)
    ss_res = float(((z - pred) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="40-step smoke run; schema only, no assertions")
    ap.add_argument("--json", action="store_true", dest="as_json", help="single-line JSON output")
    args = ap.parse_args()
    steps, log_freq = (40, 2) if args.quick else (8000, 40)
    nodes, edges, tid = _graph(steps, log_freq)
    complete = [e for e in iter_trainer_events(nodes, edges, tid) if e.get("type") == "complete"]
    if not complete:
        print("no complete event", file=sys.stderr)
        return 1
    done = complete[0]
    hists = done["observable_metric_histories"]
    ticks = np.asarray(done["step_ticks"], dtype=float)
    curves = [np.asarray(hists[f"obs-sv::{i}"], dtype=float) for i in range(TOP_K)]
    half_steps = [_half_max_step(c, ticks) for c in curves]
    fits = [_logit_r2(c, ticks) for c in curves]
    known = [h for h in half_steps if h is not None]
    ordered = len(known) == TOP_K and all(a < b for a, b in zip(known, known[1:]))
    result = {
        "experiment": "staggered_sv",
        "quick": bool(args.quick),
        "top_k": TOP_K,
        "steps": steps,
        "half_max_steps": half_steps,
        "ordered": ordered,
        "logit_fit_r2": fits,
        "final_svs": [float(c[-1]) for c in curves],
    }
    print(json.dumps(result) if args.as_json else json.dumps(result, indent=2))
    if args.quick:
        return 0
    ok = ordered and all(f is not None and f >= 0.9 for f in fits)
    if not ok:
        print("PHENOMENON ASSERTIONS FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
