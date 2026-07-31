"""Run the declared Information Bottleneck supplemental controls serially."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_research.engine.reproductions.information_bottleneck_controls import (  # noqa: E402
    SUPPLEMENTAL_CONTROLS,
    supplemental_control_map,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "runs" / "reproductions" / "information_bottleneck" / "supplemental_controls"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_suite_directory(root: Path) -> Path:
    stem = datetime.now().strftime("suite-%Y%m%d-%H%M%S")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _parse_args() -> argparse.Namespace:
    control_ids = tuple(control.control_id for control in SUPPLEMENTAL_CONTROLS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", nargs="+", choices=control_ids, default=list(control_ids))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Independent seeds per control (main Figure 3 remains a separate 50-repeat run).",
    )
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.benchmark:
        parser.error("--smoke and --benchmark are mutually exclusive")
    if args.repeats < 1 or args.epochs < 1:
        parser.error("--repeats and --epochs must be positive")
    return args


def main() -> int:
    args = _parse_args()
    controls = supplemental_control_map()
    suite_dir = _new_suite_directory(args.output_root.resolve())
    run_kind = "smoke" if args.smoke else "benchmark" if args.benchmark else "full"
    started = time.time()
    status: dict[str, Any] = {
        "status": "running",
        "run_kind": run_kind,
        "serial_execution": True,
        "started_at": datetime.now().astimezone().isoformat(),
        "device": args.device,
        "repeats_per_control": args.repeats,
        "epochs_per_control": args.epochs,
        "controls_requested": list(args.controls),
        "controls": {},
        "interpretation_boundary": (
            "Binned estimates are method-dependent diagnostics, not absolute mutual information "
            "of a deterministic continuous network. Main Figure 3 results remain separate."
        ),
    }
    _write_json(
        suite_dir / "matrix.json",
        {control.control_id: control.to_dict() for control in SUPPLEMENTAL_CONTROLS},
    )
    _write_json(suite_dir / "status.json", status)

    try:
        for control_id in args.controls:
            control = controls[control_id]
            output_root = suite_dir / control_id
            before = set(output_root.glob("batch-*")) if output_root.exists() else set()
            command = [
                sys.executable,
                str(ROOT / "scripts" / "reproductions" / "information_bottleneck.py"),
                "--profile",
                "saxe_2019",
                "--percent",
                "80",
                "--device",
                args.device,
                "--output-root",
                str(output_root),
                "--activation",
                control.activation,
                "--bins",
                str(control.bins),
                "--binning",
                control.binning,
                "--optimizer",
                control.optimizer,
                "--batch-size",
                str(control.batch_size),
                "--learning-rate",
                str(control.learning_rate),
                "--repeats",
                str(args.repeats),
                "--epochs",
                str(args.epochs),
            ]
            if args.smoke:
                command.append("--smoke")
            elif args.benchmark:
                command.append("--benchmark")
            control_status: dict[str, Any] = {
                "status": "running",
                "started_at": datetime.now().astimezone().isoformat(),
                "protocol": control.to_dict(),
                "command": command,
            }
            status["controls"][control_id] = control_status
            _write_json(suite_dir / "status.json", status)
            print(f"[{control_id}] starting", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            after = set(output_root.glob("batch-*")) if output_root.exists() else set()
            created = sorted(after - before, key=lambda path: path.stat().st_mtime)
            result_dir = created[-1] if created else None
            child_status: dict[str, Any] | None = None
            if result_dir is not None and (result_dir / "status.json").is_file():
                child_status = json.loads((result_dir / "status.json").read_text(encoding="utf-8"))
            control_status.update(
                {
                    "status": "completed" if completed.returncode == 0 else "failed",
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "return_code": completed.returncode,
                    "result_dir": str(result_dir) if result_dir is not None else None,
                    "child_status": child_status,
                }
            )
            _write_json(suite_dir / "status.json", status)
            if completed.returncode != 0:
                raise RuntimeError(f"supplemental control {control_id} failed")

        status.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "reload_verified": all(
                    item.get("child_status", {}).get("reload_verified")
                    for item in status["controls"].values()
                ),
            }
        )
        _write_json(suite_dir / "status.json", status)
        json.loads((suite_dir / "status.json").read_text(encoding="utf-8"))
        print(suite_dir, flush=True)
        return 0
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(suite_dir / "status.json", status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
