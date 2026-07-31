"""Run non-overwriting Zhang et al. Figure 1(a) reproductions serially.

Examples:
  python scripts/reproductions/random_labels_figure1a.py --smoke --condition true_labels --device cuda
  python scripts/reproductions/random_labels_figure1a.py --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comfy_research.engine.datasets.vision_datasets_runtime import _load_cifar10_official  # noqa: E402
from comfy_research.engine.reproductions.random_labels import (  # noqa: E402
    CONDITION_LABELS,
    CONDITION_ORDER,
    RandomLabelsProtocol,
    RandomLabelsResult,
    numpy_sha256,
    prepare_condition_arrays,
    train_random_label_condition,
)
from comfy_research.engine.reproductions.gpu_preflight import (  # noqa: E402
    resolve_reproduction_device,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "runs" / "reproductions" / "random_label_memorization" / "paper_faithful"
)


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_batch_directory(root: Path) -> Path:
    stem = datetime.now().strftime("batch-%Y%m%d-%H%M%S")
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _save_metrics(condition_dir: Path, result: RandomLabelsResult) -> None:
    path = condition_dir / "metrics.npz"
    np.savez_compressed(
        path,
        step_ticks=result.step_ticks,
        interval_loss=result.interval_loss,
        interval_accuracy=result.interval_accuracy,
        learning_rate=result.learning_rate,
        full_eval_steps=result.full_eval_steps,
        full_train_loss=result.full_train_loss,
        full_train_accuracy=result.full_train_accuracy,
        final_test_loss=np.asarray(result.final_test_loss),
        final_test_accuracy=np.asarray(result.final_test_accuracy),
    )
    _write_json(condition_dir / "aggregate.json", result.aggregate())
    with np.load(path) as saved:
        for key, expected in (
            ("step_ticks", result.step_ticks),
            ("interval_loss", result.interval_loss),
            ("interval_accuracy", result.interval_accuracy),
            ("learning_rate", result.learning_rate),
            ("full_eval_steps", result.full_eval_steps),
            ("full_train_loss", result.full_train_loss),
            ("full_train_accuracy", result.full_train_accuracy),
        ):
            np.testing.assert_array_equal(saved[key], expected)


def _load_persisted_results(batch_dir: Path, conditions: list[str]) -> dict[str, dict[str, np.ndarray]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for condition in conditions:
        with np.load(batch_dir / condition / "metrics.npz") as saved:
            loaded[condition] = {
                "step_ticks": saved["step_ticks"].copy(),
                "interval_loss": saved["interval_loss"].copy(),
                "full_eval_steps": saved["full_eval_steps"].copy(),
                "full_train_loss": saved["full_train_loss"].copy(),
            }
    return loaded


def _plot_figure1a(batch_dir: Path, conditions: list[str], *, run_kind: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    persisted = _load_persisted_results(batch_dir, conditions)
    colors = {
        "true_labels": "#355f8d",
        "random_labels": "#d1495b",
        "shuffled_pixels": "#e98b2a",
        "random_pixels": "#2a9d8f",
        "gaussian": "#7a5195",
    }
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for condition in conditions:
        values = persisted[condition]
        ticks = values["step_ticks"].astype(np.float64) / 1_000.0
        axis.plot(
            ticks,
            values["interval_loss"],
            label=CONDITION_LABELS[condition],
            color=colors[condition],
            linewidth=1.8,
        )
        # Full-dataset eval-mode diagnostics remain in metrics.npz.  Keep them
        # off the formal paper-style plot because BatchNorm eval statistics and
        # the paper's online average-loss curve are different measurement
        # conventions.  Smoke/benchmark plots retain the diagnostic dots.
        if run_kind != "full":
            axis.scatter(
                values["full_eval_steps"].astype(np.float64) / 1_000.0,
                values["full_train_loss"],
                color=colors[condition],
                s=8,
                alpha=0.45,
                edgecolors="none",
            )
    axis.set_xlabel("thousand steps")
    axis.set_ylabel("average_loss")
    axis.set_xlim(left=0)
    if run_kind == "full":
        axis.set_xlim(0, 25)
        axis.set_ylim(0, 2.5)
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, ncol=1)
    title_suffix = {
        "smoke": "smoke validation",
        "benchmark": "full-data timing benchmark",
        "full": "paper-faithful reproduction",
    }[run_kind]
    axis.set_title(f"Zhang et al. Figure 1(a) - {title_suffix}")
    figure.tight_layout()
    figure.savefig(batch_dir / "figure1a.png", dpi=220)
    figure.savefig(batch_dir / "figure1a.pdf")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", nargs="+", choices=CONDITION_ORDER, default=list(CONDITION_ORDER))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--full-eval-every", type=int, default=1_000)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1_703)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.benchmark:
        parser.error("--smoke and --benchmark are mutually exclusive")
    if args.smoke:
        args.train_size = min(args.train_size, 512)
        args.test_size = min(args.test_size, 128)
        args.steps = min(args.steps, 4)
        args.log_every = 1
        args.full_eval_every = 2
        if args.condition == list(CONDITION_ORDER):
            args.condition = ["true_labels"]
    elif args.benchmark:
        args.steps = min(args.steps, 200)
        args.log_every = min(args.log_every, 20)
        args.full_eval_every = args.steps
        if args.condition == list(CONDITION_ORDER):
            args.condition = ["true_labels"]
    return args


def main() -> int:
    args = _parse_args()
    device, gpu_preflight = resolve_reproduction_device(
        args.device, allow_busy_gpu=args.allow_busy_gpu
    )
    batch_dir = _new_batch_directory(args.output_root.resolve())
    started = time.time()
    run_kind = "smoke" if args.smoke else "benchmark" if args.benchmark else "full"
    root_status: dict[str, Any] = {
        "status": "running",
        "run_kind": run_kind,
        "started_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "gpu_preflight": gpu_preflight,
        "serial_execution": True,
        "conditions_requested": list(args.condition),
        "conditions": {},
        "paper_disclosed": {
            "dataset": "CIFAR-10 50,000 train / 10,000 validation",
            "preprocessing": "divide by 255, center crop 28x28, TensorFlow per_image_whitening",
            "model": "Small Inception, Appendix Figure 3",
            "optimizer": "SGD, momentum=0.9, learning_rate=0.1, decay factor 0.95 per epoch",
            "regularization": "none for randomized labels or pixels",
            "figure_axis": "average_loss versus thousand steps, through 25k steps",
        },
        "implementation_choices_not_disclosed_by_paper": {
            "batch_size": args.batch_size,
            "initialization": "PyTorch module defaults with a fixed shared model seed",
            "minibatches": "one fixed-seed random permutation per data epoch",
            "curve_statistic": "sample-weighted mean online cross-entropy per log interval",
            "exact_diagnostics": "full-dataset loss and accuracy every full_eval_every steps",
            "batch_norm_running_statistics": "PyTorch BatchNorm2d defaults; learned scale disabled",
        },
    }
    _write_json(batch_dir / "status.json", root_status)

    try:
        base_rng = np.random.default_rng(args.seed)
        raw_arrays = _load_cifar10_official(
            train_n=args.train_size,
            test_n=args.test_size,
            rng=base_rng,
            cache_dir=str(ROOT / "data" / "cifar10"),
        )
        raw_fingerprints = {
            "train_inputs_sha256": numpy_sha256(raw_arrays[0]),
            "train_labels_sha256": numpy_sha256(raw_arrays[1]),
            "test_inputs_sha256": numpy_sha256(raw_arrays[2]) if raw_arrays[2] is not None else None,
            "test_labels_sha256": numpy_sha256(raw_arrays[3]) if raw_arrays[3] is not None else None,
        }
        _write_json(batch_dir / "raw_dataset_fingerprints.json", raw_fingerprints)

        for condition in args.condition:
            condition_dir = batch_dir / condition
            condition_dir.mkdir()
            protocol = RandomLabelsProtocol(
                condition=condition,
                train_size=args.train_size,
                test_size=args.test_size,
                steps=args.steps,
                batch_size=args.batch_size,
                log_every=args.log_every,
                full_eval_every=args.full_eval_every,
                eval_batch_size=args.eval_batch_size,
                dataset_seed=args.seed,
                # A condition keeps the same seed whether it is run alone or
                # as part of the five-condition serial batch.
                transform_seed=args.seed + 101 + CONDITION_ORDER.index(condition) * 1_009,
                model_seed=args.seed + 3,
                minibatch_seed=args.seed + 4,
            )
            _write_json(condition_dir / "protocol.json", protocol.to_dict())
            condition_status: dict[str, Any] = {
                "status": "preparing_data",
                "started_at": datetime.now().astimezone().isoformat(),
                "step": 0,
                "steps": protocol.steps,
            }
            _write_json(condition_dir / "status.json", condition_status)
            prepared = prepare_condition_arrays(raw_arrays, protocol)
            fingerprints = {
                "train_inputs_sha256": numpy_sha256(prepared[0]),
                "train_labels_sha256": numpy_sha256(prepared[1]),
                "test_inputs_sha256": numpy_sha256(prepared[2]) if prepared[2] is not None else None,
                "test_labels_sha256": numpy_sha256(prepared[3]) if prepared[3] is not None else None,
            }
            _write_json(condition_dir / "dataset_fingerprints.json", fingerprints)
            condition_status["status"] = "running"
            _write_json(condition_dir / "status.json", condition_status)
            condition_started = time.time()

            def progress(step: int, total: int, metrics: dict[str, float]) -> None:
                condition_status.update({"step": step, "last_metrics": metrics})
                _write_json(condition_dir / "status.json", condition_status)
                print(
                    f"[{condition}] step {step}/{total} loss={metrics['loss']:.6f} "
                    f"acc={metrics['accuracy']:.4f} lr={metrics['lr']:.6g}",
                    flush=True,
                )

            result, model = train_random_label_condition(
                protocol,
                prepared,
                device=device,
                progress=progress,
            )
            _save_metrics(condition_dir, result)
            condition_status.update(
                {
                    "status": "completed",
                    "step": protocol.steps,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "elapsed_seconds": time.time() - condition_started,
                    "reload_verified": True,
                    "aggregate": result.aggregate(),
                }
            )
            _write_json(condition_dir / "status.json", condition_status)
            root_status["conditions"][condition] = condition_status
            _write_json(batch_dir / "status.json", root_status)
            del model, prepared
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _plot_figure1a(batch_dir, list(args.condition), run_kind=run_kind)
        _load_persisted_results(batch_dir, list(args.condition))
        root_status.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "reload_verified": all(
                    item.get("reload_verified") for item in root_status["conditions"].values()
                ),
                "files": {"plot_png": "figure1a.png", "plot_pdf": "figure1a.pdf"},
            }
        )
        _write_json(batch_dir / "status.json", root_status)
        print(batch_dir, flush=True)
        return 0
    except BaseException as exc:
        root_status.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_json(batch_dir / "status.json", root_status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
