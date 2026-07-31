#!/usr/bin/env python3
"""Objective Slingshot reproduction detector; screenshots are not accepted as evidence.

跑 repro-thilak-fig1-slingshot 模板图(in-process,iter_trainer_events),判据:
  (a) train accuracy 达 1.0;
  (b) 其后 ≥ --min-cycles 个弹弓循环:trough→peak 的 train loss 比值 ≥ --spike-ratio,
      且尖峰前 NORM_WINDOW 步内 last-layer weight norm 正增长
      (论文签名:增长期→尖峰;尖峰瞬间可小幅回落,不做尖峰后判定);
  (c) TPT 全程 norm 净增长(Fig 1 阶梯)。
输出 JSON 摘要(device/seeds/spikes);达标退出码 0,否则 1。

用法:
  python comfy_research/tools/verify_slingshot_repro.py
  python comfy_research/tools/verify_slingshot_repro.py --steps 6000 --beta2 0.98 --width 512
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_TEMPLATE = _REPO / "data" / "graph_library" / "templates" / "repro-thilak-fig1-slingshot.json"

SPIKE_RATIO_DEFAULT = 1e4
NORM_WINDOW = 50  # pre-spike window for the norm growth check
# float32 晚期 loss 会下溢到精确 0;trough 以此为底,否则比值检测被永久关闭。
TROUGH_FLOOR = 1e-12


def _load_graph(args: argparse.Namespace):
    from comfy_research.schemas.graph import Edge, GraphDocument, Node  # noqa: F401

    entry = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    doc = GraphDocument.model_validate(entry["document"])
    nodes, edges = list(doc.nodes), list(doc.edges)

    def kind(n: Any) -> str:
        return getattr(n.type, "value", None) or str(n.type)

    for n in nodes:
        d = dict(n.data or {})
        if kind(n) == "trainer":
            if args.steps is not None:
                d["trainingSteps"] = args.steps
            d["logFrequency"] = 1  # 验证跑最密采样，防止漏掉窄尖峰。
            if args.device:
                d["computeDevice"] = args.device
            n.data = d
        elif kind(n) == "adam_optimizer" and args.beta2 is not None:
            d["beta2"] = args.beta2
            n.data = d
        elif kind(n) == "mlp_model" and args.width is not None:
            d["width"] = args.width
            n.data = d
    trainer_id = next(n.id for n in nodes if kind(n) == "trainer")
    norm_id = next(n.id for n in nodes if kind(n) == "observable_last_layer_weight_norm")
    acc_id = next(n.id for n in nodes if kind(n) == "observable_accuracy")
    trainer_node = next(n for n in nodes if n.id == trainer_id)
    device_spec = str((trainer_node.data or {}).get("computeDevice") or "auto")
    return nodes, edges, trainer_id, norm_id, acc_id, device_spec


def detect_slingshots(
    loss: list[float],
    norm: list[float],
    acc: list[float],
    spike_ratio: float,
) -> dict[str, Any]:
    """Pure detector (unit-testable): spikes after acc==1.0 with a positive norm step."""
    n = min(len(loss), len(norm), len(acc))
    first_perfect = next((i for i in range(n) if acc[i] >= 1.0), None)
    out: dict[str, Any] = {"first_perfect_acc_index": first_perfect, "cycles": [], "qualified_cycles": 0}
    if first_perfect is None:
        return out
    trough = math.inf
    trough_i = first_perfect
    i = first_perfect
    while i < n:
        li = loss[i]
        if math.isfinite(li) and li < trough:
            trough, trough_i = li, i
        if math.isfinite(li) and li / max(trough, TROUGH_FLOOR) >= spike_ratio:
            # spike at i;paper signature = rapid norm GROWTH leading into the
            # spike(增长期→尖峰→平台;尖峰瞬间可小幅回落,不做尖峰后判定)。
            lo = max(first_perfect, i - NORM_WINDOW)
            pre_growth = (
                norm[i] - norm[lo]
                if math.isfinite(norm[i]) and math.isfinite(norm[lo]) and lo < i
                else 0.0
            )
            out["cycles"].append({
                "spike_index": i,
                "trough_index": trough_i,
                "trough_loss": trough,
                "peak_loss": li,
                "ratio": li / max(trough, TROUGH_FLOOR),
                "pre_spike_norm_growth": pre_growth,
                "norm_growth_positive": pre_growth > 0,
            })
            # reset for next cycle: skip the spike region — peak-relative
            # (trough 为 0 时 trough 相对阈值会吞掉后续尖峰)。
            j = i
            while j < n and math.isfinite(loss[j]) and loss[j] > li / spike_ratio:
                j += 1
            i = j
            trough = math.inf
            trough_i = i
            continue
        i += 1
    out["qualified_cycles"] = sum(1 for c in out["cycles"] if c["norm_growth_positive"])
    # 跨周期阶梯:TPT 全程 norm 净增长(Fig 1 magenta staircase)。
    finite = [v for v in norm[first_perfect:n] if math.isfinite(v)]
    out["norm_staircase"] = bool(finite and finite[-1] > finite[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=None, help="override trainer trainingSteps")
    ap.add_argument("--beta2", type=float, default=None, help="override Adam beta2")
    ap.add_argument("--width", type=int, default=None, help="override MLP width")
    ap.add_argument("--device", type=str, default=None, help="override computeDevice (cpu/mps/cuda)")
    ap.add_argument("--spike-ratio", type=float, default=SPIKE_RATIO_DEFAULT)
    ap.add_argument("--min-cycles", type=int, default=2)
    ap.add_argument("--out", type=str, default=None, help="write full histories JSON here")
    args = ap.parse_args()

    from comfy_research.engine.runs.trainer_run import iter_trainer_events
    from comfy_research.engine.trainer.device_runtime import _resolve_trainer_compute_device

    nodes, edges, trainer_id, norm_id, acc_id, device_spec = _load_graph(args)
    complete: dict[str, Any] | None = None
    for ev in iter_trainer_events(nodes, edges, trainer_id):
        if ev.get("type") == "complete":
            complete = ev
    if complete is None:
        print(json.dumps({"ok": False, "error": "no complete event"}))
        return 1

    hist = complete["observable_metric_histories"]
    loss = [float(x) for x in complete["loss_history"]]
    norm = [float(x) for x in hist.get(norm_id, [])]
    acc = [float(x) for x in hist.get(acc_id, [])]
    det = detect_slingshots(loss, norm, acc, args.spike_ratio)

    ok = (
        det["first_perfect_acc_index"] is not None
        and det["qualified_cycles"] >= args.min_cycles
        and det.get("norm_staircase", False)
    )
    summary = {
        "ok": ok,
        # 实际生效设备(trainer computeDevice 解析结果),而非机器可用硬件。
        "device": str(_resolve_trainer_compute_device(device_spec)),
        "steps": len(loss),
        "history_lengths": {"loss": len(loss), "norm": len(norm), "acc": len(acc)},
        "max_acc": max(acc) if acc else None,
        "min_loss": min((x for x in loss if math.isfinite(x)), default=None),
        "spike_ratio_threshold": args.spike_ratio,
        "min_cycles": args.min_cycles,
        "first_perfect_acc_index": det["first_perfect_acc_index"],
        "qualified_cycles": det["qualified_cycles"],
        "norm_staircase": det.get("norm_staircase", False),
        "cycles": det["cycles"],
        "final_norm": norm[-1] if norm else None,
        "overrides": {"steps": args.steps, "beta2": args.beta2, "width": args.width, "device": args.device},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(
            json.dumps({"loss": loss, "norm": norm, "acc": acc, "summary": summary}),
            encoding="utf-8",
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
