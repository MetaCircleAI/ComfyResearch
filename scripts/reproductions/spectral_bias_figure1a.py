"""Run a non-overwriting Rahaman et al. Figure 1(a) reproduction.

Examples:
  python scripts/reproductions/spectral_bias_figure1a.py --smoke --device cpu
  python scripts/reproductions/spectral_bias_figure1a.py --benchmark --device cuda
  python scripts/reproductions/spectral_bias_figure1a.py --device cuda
  python scripts/reproductions/spectral_bias_figure1a.py --resume-batch runs/.../batch-...
"""

from __future__ import annotations

import argparse
from dataclasses import fields
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

from comfy_research.engine.reproductions.gpu_preflight import (  # noqa: E402
    resolve_reproduction_device,
)
from comfy_research.engine.reproductions.spectral_bias import (  # noqa: E402
    SpectralBiasProtocol,
    SpectralBiasRepeatResult,
    make_phase_target,
    train_spectral_bias_repeat,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "runs" / "reproductions" / "spectral_bias" / "paper_faithful"
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


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
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


def _save_repeat_metrics(phase_dir: Path, result: SpectralBiasRepeatResult) -> None:
    metrics_path = phase_dir / "metrics.npz"
    np.savez_compressed(
        metrics_path,
        iterations=result.iterations,
        train_loss=result.train_loss,
        normalized_amplitudes=result.normalized_amplitudes,
        author_layer_norms=result.author_layer_norms,
        singular_layer_norms=result.singular_layer_norms,
        prediction_iterations=result.prediction_iterations,
        predictions=result.predictions,
        phase_fractions=result.phase_fractions,
        final_prediction=result.final_prediction,
    )
    with np.load(metrics_path) as saved:
        for key, expected in (
            ("iterations", result.iterations),
            ("train_loss", result.train_loss),
            ("normalized_amplitudes", result.normalized_amplitudes),
            ("author_layer_norms", result.author_layer_norms),
            ("singular_layer_norms", result.singular_layer_norms),
            ("prediction_iterations", result.prediction_iterations),
            ("predictions", result.predictions),
            ("phase_fractions", result.phase_fractions),
            ("final_prediction", result.final_prediction),
        ):
            np.testing.assert_array_equal(saved[key], expected)


def _load_repeat_metrics(path: Path) -> SpectralBiasRepeatResult:
    with np.load(path) as saved:
        return SpectralBiasRepeatResult(
            iterations=saved["iterations"].copy(),
            train_loss=saved["train_loss"].copy(),
            normalized_amplitudes=saved["normalized_amplitudes"].copy(),
            author_layer_norms=saved["author_layer_norms"].copy(),
            singular_layer_norms=saved["singular_layer_norms"].copy(),
            prediction_iterations=saved["prediction_iterations"].copy(),
            predictions=saved["predictions"].copy(),
            phase_fractions=saved["phase_fractions"].copy(),
            final_prediction=saved["final_prediction"].copy(),
        )


def _stack_results(
    results: list[SpectralBiasRepeatResult],
) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("cannot aggregate an empty result list")
    for result in results[1:]:
        np.testing.assert_array_equal(result.iterations, results[0].iterations)
        np.testing.assert_array_equal(
            result.prediction_iterations, results[0].prediction_iterations
        )
    return {
        "iterations": results[0].iterations,
        "train_loss": np.stack([result.train_loss for result in results]),
        "normalized_amplitudes": np.stack(
            [result.normalized_amplitudes for result in results]
        ),
        "author_layer_norms": np.stack(
            [result.author_layer_norms for result in results]
        ),
        "singular_layer_norms": np.stack(
            [result.singular_layer_norms for result in results]
        ),
        "prediction_iterations": results[0].prediction_iterations,
        "predictions": np.stack([result.predictions for result in results]),
        "phase_fractions": np.stack([result.phase_fractions for result in results]),
        "final_prediction": np.stack([result.final_prediction for result in results]),
    }


def _onset_iterations(
    iterations: np.ndarray,
    mean_amplitudes: np.ndarray,
    *,
    threshold: float = 0.8,
) -> list[int | None]:
    output: list[int | None] = []
    for column in range(mean_amplitudes.shape[1]):
        rows = np.flatnonzero(mean_amplitudes[:, column] >= threshold)
        output.append(int(iterations[rows[0]]) if rows.size else None)
    return output


def _save_aggregate(
    batch_dir: Path,
    arrays: dict[str, np.ndarray],
    protocol: SpectralBiasProtocol,
) -> dict[str, Any]:
    metrics_path = batch_dir / "aggregate_metrics.npz"
    np.savez_compressed(metrics_path, **arrays)
    with np.load(metrics_path) as saved:
        for key, expected in arrays.items():
            np.testing.assert_array_equal(saved[key], expected)

    amplitudes = arrays["normalized_amplitudes"]
    author_norms = arrays["author_layer_norms"]
    singular_norms = arrays["singular_layer_norms"]
    mean_amplitudes = amplitudes.mean(axis=0)
    summary: dict[str, Any] = {
        "repeats": int(amplitudes.shape[0]),
        "record_count": int(amplitudes.shape[1]),
        "frequencies": list(protocol.frequencies),
        "final_recorded_iteration": int(arrays["iterations"][-1]),
        "final_mean_normalized_amplitudes": mean_amplitudes[-1].tolist(),
        "final_std_normalized_amplitudes": amplitudes[:, -1].std(axis=0).tolist(),
        "mean_amplitude_onset_iteration_at_0_8": _onset_iterations(
            arrays["iterations"], mean_amplitudes, threshold=0.8
        ),
        "final_mean_author_layer_norm_estimator": author_norms[:, -1].mean(axis=0).tolist(),
        "final_mean_conventional_singular_norm_estimator": singular_norms[:, -1].mean(
            axis=0
        ).tolist(),
        "final_mean_train_loss": float(arrays["train_loss"][:, -1].mean()),
        "author_estimator_warning": (
            "The published-code curve is not a conventional spectral norm for square layers: "
            "it iterates W rather than W.T@W and preserves the notebook's left-associative "
            "(b.T@W@b)/b.T@b expression. Conventional estimates are stored separately."
        ),
    }
    _write_json(batch_dir / "summary.json", summary)
    return summary


def _plot_main_figure(
    batch_dir: Path,
    arrays: dict[str, np.ndarray],
    protocol: SpectralBiasProtocol,
    *,
    run_kind: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        cmap = sns.cubehelix_palette(
            8, start=0.5, rot=-0.75, reverse=True, as_cmap=True
        )
    except ImportError:
        cmap = "YlGnBu_r"

    iterations = arrays["iterations"]
    mean_amplitudes = arrays["normalized_amplitudes"].mean(axis=0)
    author_norms = arrays["author_layer_norms"]
    mean_norms = author_norms.mean(axis=0)
    std_norms = author_norms.std(axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    image = axes[0].imshow(
        np.clip(mean_amplitudes, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        extent=(
            float(protocol.frequencies[0] - 2.5),
            float(protocol.frequencies[-1] + 2.5),
            float(iterations[0]),
            float(protocol.iterations),
        ),
    )
    axes[0].set_xticks(protocol.frequencies)
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("Training Iteration")
    figure.colorbar(image, ax=axes[0], fraction=0.047, pad=0.03)

    for layer in range(mean_norms.shape[1]):
        line = axes[1].plot(
            iterations,
            mean_norms[:, layer],
            linewidth=1.0,
            label=f"Layer {layer + 1}",
        )[0]
        axes[1].fill_between(
            iterations,
            mean_norms[:, layer] - std_norms[:, layer],
            mean_norms[:, layer] + std_norms[:, layer],
            color=line.get_color(),
            alpha=0.13,
            linewidth=0,
        )
    axes[1].set_xlim(0, protocol.iterations)
    axes[1].set_xlabel("Training Iteration")
    axes[1].set_ylabel("Spectral Norm of Layer Weights")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.16)
    label = (
        "Rahaman et al. Figure 1(a) - paper-faithful 10-phase aggregate"
        if run_kind == "full"
        else f"Rahaman et al. Figure 1(a) - {run_kind}"
    )
    figure.suptitle(label)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(batch_dir / "figure1a.png", dpi=240)
    figure.savefig(batch_dir / "figure1a.pdf")
    plt.close(figure)


def _plot_norm_audit(
    batch_dir: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = arrays["iterations"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True)
    for axis, key, title, ylabel in (
        (
            axes[0],
            "author_layer_norms",
            "Released-notebook estimator (Figure 1)",
            "Author estimator",
        ),
        (
            axes[1],
            "singular_layer_norms",
            "Conventional 10-step singular-norm estimate",
            "Estimated top singular value",
        ),
    ):
        values = arrays[key]
        mean = values.mean(axis=0)
        for layer in range(mean.shape[1]):
            axis.plot(iterations, mean[:, layer], linewidth=0.9, label=f"Layer {layer + 1}")
        axis.set_title(title)
        axis.set_xlabel("Training iteration")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.16)
    axes[0].legend(fontsize=8)
    figure.suptitle("Spectral-norm implementation audit")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(batch_dir / "spectral_norm_audit.png", dpi=220)
    figure.savefig(batch_dir / "spectral_norm_audit.pdf")
    plt.close(figure)


def _plot_prediction_snapshots(
    batch_dir: Path,
    arrays: dict[str, np.ndarray],
    protocol: SpectralBiasProtocol,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, target, _ = make_phase_target(protocol, phase_seed=protocol.phase_seed_base)
    steps = arrays["prediction_iterations"]
    predictions = arrays["predictions"][0]
    figure, axes = plt.subplots(1, len(steps), figsize=(3.1 * len(steps), 3.0), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for axis, step, prediction in zip(axes, steps, predictions):
        axis.plot(x, target, linewidth=0.8, color="#3569a8", label="Target")
        axis.plot(x, prediction, linewidth=0.8, color="#45a36b", label="Model")
        axis.set_title(f"Iteration {int(step)}")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.12)
    axes[0].legend(fontsize=8)
    figure.suptitle("Phase 1 prediction snapshots (Figure 2 auxiliary)")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(batch_dir / "figure2_phase01_auxiliary.png", dpi=220)
    figure.savefig(batch_dir / "figure2_phase01_auxiliary.pdf")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resume-batch", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=80_000)
    parser.add_argument("--record-every", type=int, default=100)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--linear-layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--power-iterations", type=int, default=10)
    parser.add_argument("--phase-seed-base", type=int, default=1703)
    parser.add_argument("--model-seed-base", type=int, default=2703)
    parser.add_argument("--metric-seed-base", type=int, default=3703)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.benchmark:
        parser.error("--smoke and --benchmark are mutually exclusive")
    if args.resume_batch is not None and (args.smoke or args.benchmark):
        parser.error("--resume-batch cannot be combined with --smoke/--benchmark")
    if args.smoke:
        args.repeats = 1
        args.iterations = min(args.iterations, 8)
        args.record_every = 1
        args.width = min(args.width, 16)
        args.linear_layers = min(args.linear_layers, 3)
        args.checkpoint_every = min(args.checkpoint_every, 4)
    elif args.benchmark:
        args.repeats = 1
        args.iterations = min(args.iterations, 1_000)
        args.record_every = min(args.record_every, 100)
        args.checkpoint_every = 0
    return args


def _protocol_from_args(args: argparse.Namespace) -> SpectralBiasProtocol:
    return SpectralBiasProtocol(
        repeats=args.repeats,
        iterations=args.iterations,
        record_every=args.record_every,
        width=args.width,
        linear_layers=args.linear_layers,
        learning_rate=args.learning_rate,
        power_iterations=args.power_iterations,
        phase_seed_base=args.phase_seed_base,
        model_seed_base=args.model_seed_base,
        metric_seed_base=args.metric_seed_base,
    )


def _protocol_from_json(payload: dict[str, Any]) -> SpectralBiasProtocol:
    allowed = {field.name for field in fields(SpectralBiasProtocol)}
    values = {key: value for key, value in payload.items() if key in allowed}
    for key in ("frequencies", "amplitudes", "adam_betas"):
        if key in values:
            values[key] = tuple(values[key])
    return SpectralBiasProtocol(**values)


def main() -> int:
    args = _parse_args()
    device, gpu_preflight = resolve_reproduction_device(
        args.device, allow_busy_gpu=args.allow_busy_gpu
    )
    if args.resume_batch is None:
        protocol = _protocol_from_args(args)
        protocol.validate()
        batch_dir = _new_batch_directory(args.output_root.resolve())
        run_kind = "smoke" if args.smoke else "benchmark" if args.benchmark else "full"
        started = time.time()
        root_status: dict[str, Any] = {
            "status": "running",
            "run_kind": run_kind,
            "started_at": datetime.now().astimezone().isoformat(),
            "device": str(device),
            "gpu_preflight": gpu_preflight,
            "protocol": protocol.to_dict(),
            "phases": {},
            "paper_code_discrepancies": {
                "paper_figure_axis_iterations": 80_000,
                "released_notebook_saved_default_iterations": 60_000,
                "selected_iterations": protocol.iterations,
                "released_notebook_learning_rate": 3e-4,
                "old_comfyresearch_template_learning_rate": 1e-3,
                "released_notebook_depth_semantics": "6 total Linear layers",
                "old_comfyresearch_template_depth_semantics": "depth=6 hidden layers + output = 7 Linear layers",
                "selected_linear_layers": protocol.linear_layers,
                "released_notebook_grid": "np.arange(0, 1, 1/N), endpoint excluded",
                "old_comfyresearch_template_grid": "np.linspace(0, 1, N), endpoint included",
                "released_notebook_norm_estimator": "non-standard W power iteration plus left-associative precedence quirk",
            },
        }
        _write_json(batch_dir / "status.json", root_status)
        _write_json(batch_dir / "protocol.json", protocol.to_dict())
    else:
        batch_dir = args.resume_batch.resolve()
        if not batch_dir.is_dir():
            raise FileNotFoundError(f"resume batch does not exist: {batch_dir}")
        root_status = json.loads((batch_dir / "status.json").read_text(encoding="utf-8"))
        protocol = _protocol_from_json(
            json.loads((batch_dir / "protocol.json").read_text(encoding="utf-8"))
        )
        protocol.validate()
        run_kind = str(root_status.get("run_kind") or "full")
        started = time.time() - float(root_status.get("elapsed_seconds", 0.0) or 0.0)
        root_status.update(
            {
                "status": "running",
                "resumed_at": datetime.now().astimezone().isoformat(),
                "device": str(device),
                "gpu_preflight_on_resume": gpu_preflight,
            }
        )
        _write_json(batch_dir / "status.json", root_status)

    results: list[SpectralBiasRepeatResult] = []
    try:
        for repeat_index in range(protocol.repeats):
            phase_key = f"{repeat_index:02d}"
            phase_dir = batch_dir / f"phase-{phase_key}"
            phase_dir.mkdir(exist_ok=True)
            metrics_path = phase_dir / "metrics.npz"
            prior = root_status.get("phases", {}).get(phase_key, {})
            if (
                isinstance(prior, dict)
                and prior.get("status") == "completed"
                and prior.get("reload_verified") is True
                and metrics_path.is_file()
            ):
                results.append(_load_repeat_metrics(metrics_path))
                print(f"[phase {repeat_index + 1}/{protocol.repeats}] reload completed result", flush=True)
                continue

            phase_started = time.time()
            phase_status: dict[str, Any] = {
                "status": "running",
                "repeat_index": repeat_index,
                "completed_iterations": 0,
                "iterations": protocol.iterations,
                "phase_seed": protocol.phase_seed_base + repeat_index,
                "model_seed": protocol.model_seed_base + repeat_index,
                "metric_seed": protocol.metric_seed_base + repeat_index,
                "started_at": datetime.now().astimezone().isoformat(),
            }
            checkpoint_path = phase_dir / "resume_checkpoint.pt"
            resume_state: dict[str, object] | None = None
            if checkpoint_path.is_file():
                loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if isinstance(loaded, dict):
                    resume_state = loaded
                    phase_status["resumed_from_iteration"] = int(
                        loaded.get("completed_iterations", 0)
                    )
            _write_json(phase_dir / "status.json", phase_status)
            root_status.setdefault("phases", {})[phase_key] = phase_status
            _write_json(batch_dir / "status.json", root_status)
            last_status_write = int(phase_status.get("resumed_from_iteration", 0))

            def progress(completed: int, total: int, is_record: bool) -> None:
                nonlocal last_status_write
                interval = max(protocol.record_every, total // 100 or 1)
                if completed == total or completed - last_status_write >= interval:
                    last_status_write = completed
                    phase_status.update(
                        {
                            "completed_iterations": completed,
                            "last_iteration_was_recorded": is_record,
                            "elapsed_seconds": time.time() - phase_started,
                        }
                    )
                    _write_json(phase_dir / "status.json", phase_status)
                    print(
                        f"[phase {repeat_index + 1}/{protocol.repeats}] "
                        f"iteration {completed}/{total}",
                        flush=True,
                    )

            def checkpoint(
                completed: int,
                model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                recorded: dict[str, list[np.ndarray] | list[float] | list[int]],
                author_generator: torch.Generator,
                singular_generator: torch.Generator,
            ) -> None:
                _save_checkpoint_atomic(
                    checkpoint_path,
                    {
                        "repeat_index": repeat_index,
                        "completed_iterations": completed,
                        "protocol": protocol.to_dict(),
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "recorded": recorded,
                        "author_generator_state": author_generator.get_state(),
                        "singular_generator_state": singular_generator.get_state(),
                    },
                )

            result, model = train_spectral_bias_repeat(
                protocol,
                repeat_index=repeat_index,
                device=device,
                progress=progress,
                checkpoint_every=args.checkpoint_every,
                checkpoint=checkpoint if args.checkpoint_every > 0 else None,
                resume_state=resume_state,
            )
            _save_repeat_metrics(phase_dir, result)
            _save_checkpoint_atomic(
                phase_dir / "final_checkpoint.pt",
                {
                    "repeat_index": repeat_index,
                    "protocol": protocol.to_dict(),
                    "model_state": model.state_dict(),
                },
            )
            _write_json(
                phase_dir / "phase.json",
                {
                    "repeat_index": repeat_index,
                    "phase_seed": protocol.phase_seed_base + repeat_index,
                    "model_seed": protocol.model_seed_base + repeat_index,
                    "metric_seed": protocol.metric_seed_base + repeat_index,
                    "phase_fractions": result.phase_fractions,
                    "phase_radians": 2.0 * np.pi * result.phase_fractions,
                },
            )
            phase_status.update(
                {
                    "status": "completed",
                    "completed_iterations": protocol.iterations,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "elapsed_seconds": time.time() - phase_started,
                    "reload_verified": True,
                    "files": {
                        "metrics": "metrics.npz",
                        "checkpoint": "final_checkpoint.pt",
                        "phase": "phase.json",
                    },
                }
            )
            _write_json(phase_dir / "status.json", phase_status)
            root_status["phases"][phase_key] = phase_status
            _write_json(batch_dir / "status.json", root_status)
            results.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        arrays = _stack_results(results)
        summary = _save_aggregate(batch_dir, arrays, protocol)
        _plot_main_figure(batch_dir, arrays, protocol, run_kind=run_kind)
        _plot_norm_audit(batch_dir, arrays)
        _plot_prediction_snapshots(batch_dir, arrays, protocol)
        required_artifacts = (
            "aggregate_metrics.npz",
            "summary.json",
            "figure1a.png",
            "figure1a.pdf",
            "spectral_norm_audit.png",
            "spectral_norm_audit.pdf",
            "figure2_phase01_auxiliary.png",
            "figure2_phase01_auxiliary.pdf",
        )
        missing = [name for name in required_artifacts if not (batch_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"missing aggregate artifacts: {missing}")
        root_status.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "reload_verified": all(
                    phase.get("reload_verified") is True
                    for phase in root_status["phases"].values()
                ),
                "aggregate_reload_verified": True,
                "summary": summary,
                "artifacts": list(required_artifacts),
            }
        )
        _write_json(batch_dir / "status.json", root_status)
        # Verify the final root status itself after atomic replacement.
        persisted_status = json.loads(
            (batch_dir / "status.json").read_text(encoding="utf-8")
        )
        if persisted_status.get("status") != "completed" or not persisted_status.get(
            "reload_verified"
        ):
            raise RuntimeError("root status reload verification failed")
        print(batch_dir, flush=True)
        return 0
    except BaseException as exc:
        root_status.update(
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json(batch_dir / "status.json", root_status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
