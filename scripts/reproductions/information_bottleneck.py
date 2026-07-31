"""Run non-overwriting Information Bottleneck reproductions.

Examples:
  python scripts/reproductions/information_bottleneck.py --smoke --device cpu
  python scripts/reproductions/information_bottleneck.py --device cuda
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

from comfy_research.engine.reproductions.information_bottleneck import (  # noqa: E402
    InformationBottleneckProtocol,
    InformationBottleneckResult,
    train_information_bottleneck_ensemble,
)
from comfy_research.engine.reproductions.gpu_preflight import (  # noqa: E402
    resolve_reproduction_device,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "runs"
    / "reproductions"
    / "information_bottleneck"
    / "paper_faithful"
)
SUPPLEMENTAL_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT.parent / "supplemental_controls"


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


def _save_metrics(panel_dir: Path, result: InformationBottleneckResult) -> None:
    metrics_path = panel_dir / "metrics.npz"
    np.savez_compressed(
        metrics_path,
        epochs=result.epochs,
        train_loss=result.train_loss,
        train_accuracy=result.train_accuracy,
        information_x=result.information_x,
        information_y=result.information_y,
    )
    _write_json(panel_dir / "aggregate.json", result.aggregate())

    # Reload immediately: a completed panel is not accepted unless the persisted
    # arrays survive a fresh file read.
    with np.load(metrics_path) as saved:
        for key, expected in (
            ("epochs", result.epochs),
            ("train_loss", result.train_loss),
            ("train_accuracy", result.train_accuracy),
            ("information_x", result.information_x),
            ("information_y", result.information_y),
        ):
            np.testing.assert_array_equal(saved[key], expected)


def _plot_information_panels(
    batch_dir: Path,
    results: dict[int, InformationBottleneckResult],
    *,
    run_label: str,
    output_stem: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    percents = [percent for percent in (5, 45, 85) if percent in results]
    percents.extend(sorted(percent for percent in results if percent not in {5, 45, 85}))
    if not percents:
        return
    figure, axes = plt.subplots(1, len(percents), figsize=(5.0 * len(percents), 4.4), sharey=True)
    if len(percents) == 1:
        axes = [axes]
    cmap = plt.get_cmap("gnuplot")
    norm = Normalize(vmin=0, vmax=10_000)
    for axis, percent in zip(axes, percents):
        result = results[percent]
        ix = result.information_x.mean(axis=1)
        iy = result.information_y.mean(axis=1)
        stride = max(1, len(result.epochs) // 240)
        rows = list(range(0, len(result.epochs), stride))
        if rows[-1] != len(result.epochs) - 1:
            rows.append(len(result.epochs) - 1)
        for row in rows:
            color = cmap(norm(float(result.epochs[row])))
            axis.plot(ix[row], iy[row], color=color, alpha=0.22, linewidth=0.55)
            axis.scatter(ix[row], iy[row], color=[color], s=7, alpha=0.65, edgecolors="none")
        for layer in range(ix.shape[1]):
            axis.plot(ix[:, layer], iy[:, layer], color="#303030", linewidth=0.45, alpha=0.28)
        axis.set_title(f"{percent}% training patterns")
        axis.set_xlabel("I(X; T) bits")
        axis.set_xlim(0, 12.25)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("I(T; Y) bits")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axes, label="Epochs", fraction=0.025, pad=0.02)
    figure.suptitle(run_label)
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.86, wspace=0.10)
    figure.savefig(batch_dir / f"{output_stem}.png", dpi=220)
    figure.savefig(batch_dir / f"{output_stem}.pdf")
    plt.close(figure)


def _protocol(percent: int, args: argparse.Namespace) -> InformationBottleneckProtocol:
    if args.profile == "saxe_2019":
        hidden_dims = (10, 7, 5, 4, 3)
        output_mode = "two_softmax"
        initializer = "saxe_fan_out"
        snapshot_schedule = "saxe_callback"
    else:
        hidden_dims = (10, 8, 6, 4, 2)
        output_mode = "binary_sigmoid"
        initializer = "idnns_fan_in"
        snapshot_schedule = "idnns_logspace"
    return InformationBottleneckProtocol(
        train_percent=percent,
        repeats=args.repeats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        hidden_dims=hidden_dims,
        output_mode=output_mode,
        activation=args.activation,
        initializer=initializer,
        shuffle_mode=args.shuffle_mode,
        bins=args.bins,
        binning=args.binning,
        snapshot_schedule=snapshot_schedule,
        snapshot_samples=args.snapshot_samples,
        seed=args.seed + percent,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("paper_figure3", "saxe_2019"), default="paper_figure3")
    parser.add_argument("--percent", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--activation", choices=("tanh", "relu", "softsign", "softplus"), default="tanh")
    parser.add_argument("--shuffle-mode", choices=("fixed", "affine"), default="affine")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--binning",
        choices=("idnns_equal_points", "uniform_intervals", "adaptive_minmax", "saxe_fixed_width_0_07"),
        default=None,
    )
    parser.add_argument("--snapshot-samples", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1703)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.benchmark:
        parser.error("--smoke and --benchmark are mutually exclusive")
    if args.percent is None:
        args.percent = [80] if args.profile == "saxe_2019" else [5, 45, 85]
    if args.output_root is None:
        args.output_root = SUPPLEMENTAL_OUTPUT_ROOT if args.profile == "saxe_2019" else DEFAULT_OUTPUT_ROOT
    if args.binning is None:
        args.binning = "saxe_fixed_width_0_07" if args.profile == "saxe_2019" else "idnns_equal_points"
    if args.smoke:
        args.repeats = min(args.repeats, 2)
        args.epochs = min(args.epochs, 3)
        args.snapshot_samples = min(args.snapshot_samples, 3)
        args.percent = [args.percent[0]]
    elif args.benchmark:
        args.epochs = min(args.epochs, 20)
        args.snapshot_samples = min(args.snapshot_samples, 5)
        args.percent = [args.percent[0]]
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
        "profile": args.profile,
        "started_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "gpu_preflight": gpu_preflight,
        "panels": {},
        "paper_code_discrepancies": {
            "figure3_caption_architecture": "12-10-8-6-4-2-1",
            "idnns_default_architecture": "12-10-7-5-4-3-2",
            "paper_optimizer_wording": "SGD (learning rate and batch size undisclosed)",
            "released_executable_optimizer": "Adam, learning_rate=0.0004",
            "selected_executable_choice": "caption architecture plus released-code Adam hyperparameters",
            "saxe_text_relu_binning": "100 bins over global training min/max",
            "saxe_notebook_binned_branch": "floor(activity / 0.07)",
        },
    }
    _write_json(batch_dir / "status.json", root_status)
    results: dict[int, InformationBottleneckResult] = {}

    try:
        for percent in args.percent:
            protocol = _protocol(percent, args)
            panel_dir = batch_dir / f"p{percent:02d}"
            panel_dir.mkdir()
            _write_json(panel_dir / "protocol.json", protocol.to_dict())
            panel_started = time.time()
            panel_status: dict[str, Any] = {
                "status": "running",
                "epoch": 0,
                "epochs": protocol.epochs,
                "started_at": datetime.now().astimezone().isoformat(),
            }
            _write_json(panel_dir / "status.json", panel_status)

            last_status_write = 0

            def progress(epoch: int, total: int, is_snapshot: bool) -> None:
                nonlocal last_status_write
                interval = max(1, min(100, total // 100 or 1))
                if epoch == total or epoch - last_status_write >= interval:
                    last_status_write = epoch
                    panel_status.update({"epoch": epoch, "last_epoch_was_snapshot": is_snapshot})
                    _write_json(panel_dir / "status.json", panel_status)
                    print(f"[{percent:02d}%] epoch {epoch}/{total}", flush=True)

            result, model, train_indices = train_information_bottleneck_ensemble(
                protocol,
                device=device,
                progress=progress,
            )
            _save_metrics(panel_dir, result)
            np.save(panel_dir / "training_subsets.npy", train_indices)
            panel_status.update(
                {
                    "status": "completed",
                    "epoch": protocol.epochs,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "elapsed_seconds": time.time() - panel_started,
                    "reload_verified": True,
                    "files": {
                        "metrics": "metrics.npz",
                        "aggregate": "aggregate.json",
                        "training_subsets": "training_subsets.npy",
                    },
                }
            )
            _write_json(panel_dir / "status.json", panel_status)
            root_status["panels"][str(percent)] = panel_status
            _write_json(batch_dir / "status.json", root_status)
            results[percent] = result
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        is_saxe = args.profile == "saxe_2019"
        _plot_information_panels(
            batch_dir,
            results,
            run_label=(
                "Saxe et al. information-plane smoke validation"
                if is_saxe and run_kind == "smoke"
                else "Saxe et al. information-plane timing benchmark"
                if is_saxe and run_kind == "benchmark"
                else "Saxe et al. information-plane control"
                if is_saxe
                else "Shwartz-Ziv & Tishby Figure 3 - smoke validation"
                if run_kind == "smoke"
                else "Shwartz-Ziv & Tishby Figure 3 - timing benchmark"
                if run_kind == "benchmark"
                else "Shwartz-Ziv & Tishby Figure 3 - reproduction protocol reproduction"
            ),
            output_stem="saxe_information_plane_control" if is_saxe else "figure3",
        )
        root_status.update(
            {
                "status": "completed",
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.time() - started,
                "reload_verified": all(panel.get("reload_verified") for panel in root_status["panels"].values()),
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
            }
        )
        _write_json(batch_dir / "status.json", root_status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
