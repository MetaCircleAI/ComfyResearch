from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from comfy_research.engine.runs.trainer_run import iter_trainer_events_from_context, prepare_trainer_run
from comfy_research.schemas.graph import Edge, Node


TEMPLATE_DIR = ROOT / "data" / "graph_library" / "templates"
OUT_PATH = ROOT / ".cache" / "validation" / "cr_trainable_runtime_validation.json"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _node_type(node: Node) -> str:
    value = node.type
    return str(getattr(value, "value", value))


def _load_template(path: Path) -> tuple[str, str, list[Node], list[Edge], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    document = raw.get("document") if isinstance(raw, dict) else None
    if not isinstance(document, dict):
        raise ValueError(f"{path.name}: missing document")
    nodes = [Node.model_validate(node) for node in document.get("nodes", [])]
    edges = [Edge.model_validate(edge) for edge in document.get("edges", [])]
    trainers = [node for node in nodes if _node_type(node) == "trainer"]
    if len(trainers) != 1:
        raise ValueError(f"{path.name}: expected exactly one trainer, got {len(trainers)}")
    template_id = str(raw.get("id") or path.stem)
    template_name = str(raw.get("name") or template_id)
    return template_id, template_name, nodes, edges, trainers[0].id


def _run_template(path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    template_id, template_name, nodes, edges, trainer_id = _load_template(path)
    ctx = prepare_trainer_run(nodes, edges, trainer_id, resume=None, hessian_oversized_policy="skip")
    progress_events = 0
    final_event: dict[str, Any] | None = None
    for event in iter_trainer_events_from_context(ctx):
        if event.get("type") == "progress":
            progress_events += 1
        if event.get("type") == "complete":
            final_event = event
            break
        if event.get("type") in {"error", "aborted", "paused"}:
            final_event = event
            break

    elapsed = time.perf_counter() - started
    if final_event is None:
        raise RuntimeError(f"{path.name}: trainer stream ended without a terminal event")
    if final_event.get("type") != "complete":
        raise RuntimeError(f"{path.name}: expected complete, got {final_event.get('type')}")

    loss_history = final_event.get("loss_history")
    test_loss_history = final_event.get("test_loss_history")
    step_ticks = final_event.get("step_ticks")
    final_train_loss = _finite_float(loss_history[-1]) if isinstance(loss_history, list) and loss_history else None
    final_test_loss = _finite_float(test_loss_history[-1]) if isinstance(test_loss_history, list) and test_loss_history else None
    if final_train_loss is None:
        raise RuntimeError(f"{path.name}: final train loss is missing or non-finite")
    if not isinstance(step_ticks, list) or not step_ticks:
        raise RuntimeError(f"{path.name}: complete event did not include step ticks")

    return {
        "id": template_id,
        "name": template_name,
        "file": path.name,
        "trainerNodeId": trainer_id,
        "status": "complete",
        "progressEventCount": progress_events,
        "stepCount": int(step_ticks[-1]),
        "lossPointCount": len(loss_history) if isinstance(loss_history, list) else 0,
        "finalTrainLoss": final_train_loss,
        "finalTestLoss": final_test_loss,
        "visualizationNodeIds": final_event.get("visualization_node_ids", []),
        "elapsedSeconds": round(elapsed, 4),
    }


def main() -> None:
    template_paths = sorted(TEMPLATE_DIR.glob("demo-trainable-code2graph-*.json"))
    if not template_paths:
        raise SystemExit("No demo-trainable-code2graph templates found.")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in template_paths:
        try:
            result = _run_template(path)
            results.append(result)
            print(
                f"OK {result['id']}: complete step={result['stepCount']} "
                f"loss={result['finalTrainLoss']:.6g}"
            )
        except Exception as exc:  # noqa: BLE001 - validation script should report every demo.
            failures.append({"file": path.name, "error": str(exc)})
            print(f"FAIL {path.name}: {exc}")

    report = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "templateCount": len(template_paths),
        "passed": len(results),
        "failed": len(failures),
        "results": results,
        "failures": failures,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
