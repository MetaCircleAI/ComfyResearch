"""Verify and summarize a completed Information Bottleneck control suite.

The report deliberately keeps binned information-plane estimates separate from
the reproduction protocol Figure 3 run.  It compares paired controls using the deepest
recorded representation and persists both machine-readable statistics and a
compact visual report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COMPARISONS = (
    (
        "Activation",
        "activation_tanh_fixedwidth_adam_minibatch",
        "activation_relu_fixedwidth_adam_minibatch",
        "tanh",
        "ReLU",
    ),
    (
        "Binning resolution",
        "binning_tanh_adaptive30_adam_minibatch",
        "binning_tanh_adaptive100_adam_minibatch",
        "adaptive 30 bins",
        "adaptive 100 bins",
    ),
    (
        "Optimization noise",
        "noise_tanh_adaptive30_sgd_minibatch",
        "noise_tanh_adaptive30_sgd_fullbatch",
        "SGD mini-batch",
        "SGD full-batch",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_control(
    suite: Path,
    suite_status: dict[str, Any],
    control_id: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    control_status = suite_status["controls"][control_id]
    if control_status.get("status") != "completed":
        raise RuntimeError(f"control {control_id} is not completed")
    if not control_status.get("child_status", {}).get("reload_verified"):
        raise RuntimeError(f"control {control_id} was not reload-verified")

    result_dir = Path(control_status["result_dir"])
    if not result_dir.is_absolute():
        result_dir = suite / result_dir
    panel_dir = result_dir / "p80"
    required_files = (
        panel_dir / "metrics.npz",
        panel_dir / "aggregate.json",
        panel_dir / "training_subsets.npy",
        panel_dir / "status.json",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"control {control_id} is missing files: {missing}")

    with np.load(panel_dir / "metrics.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    expected = {
        "epochs",
        "train_loss",
        "train_accuracy",
        "information_x",
        "information_y",
    }
    if set(arrays) != expected:
        raise RuntimeError(
            f"control {control_id} metrics keys differ: {sorted(arrays)}"
        )
    snapshots = int(arrays["epochs"].shape[0])
    repeats = int(arrays["train_loss"].shape[1])
    if arrays["train_loss"].shape != (snapshots, repeats):
        raise RuntimeError(f"control {control_id} has invalid train_loss shape")
    if arrays["train_accuracy"].shape != (snapshots, repeats):
        raise RuntimeError(f"control {control_id} has invalid train_accuracy shape")
    if arrays["information_x"].shape[:2] != (snapshots, repeats):
        raise RuntimeError(f"control {control_id} has invalid information_x shape")
    if arrays["information_y"].shape != arrays["information_x"].shape:
        raise RuntimeError(f"control {control_id} information arrays differ in shape")
    numeric_arrays = (value for key, value in arrays.items() if key != "epochs")
    if not all(np.isfinite(value).all() for value in numeric_arrays):
        raise RuntimeError(f"control {control_id} contains non-finite metrics")

    ix = arrays["information_x"]
    iy = arrays["information_y"]
    compression = np.max(ix, axis=0) - ix[-1]
    summary = {
        "control_id": control_id,
        "protocol": control_status["protocol"],
        "result_dir": str(result_dir),
        "snapshots": snapshots,
        "repeats": repeats,
        "last_recorded_epoch": int(arrays["epochs"][-1]),
        "elapsed_seconds": float(
            control_status["child_status"]["panels"]["80"]["elapsed_seconds"]
        ),
        "final_train_loss_mean": float(np.mean(arrays["train_loss"][-1])),
        "final_train_loss_std": float(np.std(arrays["train_loss"][-1])),
        "final_train_accuracy_mean": float(np.mean(arrays["train_accuracy"][-1])),
        "final_train_accuracy_std": float(np.std(arrays["train_accuracy"][-1])),
        "final_information_x_mean_by_layer": np.mean(ix[-1], axis=0).tolist(),
        "final_information_x_std_by_layer": np.std(ix[-1], axis=0).tolist(),
        "final_information_y_mean_by_layer": np.mean(iy[-1], axis=0).tolist(),
        "final_information_y_std_by_layer": np.std(iy[-1], axis=0).tolist(),
        "information_x_compression_mean_by_layer": np.mean(
            compression, axis=0
        ).tolist(),
        "information_x_compression_std_by_layer": np.std(
            compression, axis=0
        ).tolist(),
        "reload_verified": True,
        "all_required_files_present": True,
        "all_metrics_finite": True,
    }
    return summary, arrays


def _plot_report(
    output: Path,
    controls: dict[str, dict[str, np.ndarray]],
) -> None:
    colors = ("#0072B2", "#D55E00")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    for axis, (title, first, second, first_label, second_label) in zip(
        axes, COMPARISONS, strict=True
    ):
        for color, control_id, label in zip(
            colors, (first, second), (first_label, second_label), strict=True
        ):
            arrays = controls[control_id]
            ix = np.mean(arrays["information_x"][:, :, -1], axis=1)
            iy = np.mean(arrays["information_y"][:, :, -1], axis=1)
            axis.plot(ix, iy, color=color, linewidth=2.1, label=label)
            axis.scatter(ix[0], iy[0], color=color, s=34, marker="o", zorder=3)
            axis.scatter(ix[-1], iy[-1], color=color, s=90, marker="*", zorder=3)
        axis.set_title(title)
        axis.set_xlabel("I(X; T), binned bits")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("I(T; Y), binned bits")
    fig.suptitle(
        "Information Bottleneck supplemental controls — deepest recorded layer",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.01,
        "Circles mark the first snapshot; stars mark the final snapshot. "
        "Binned values are method-dependent diagnostics, not absolute mutual information.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    suite = args.suite.resolve()
    status_path = suite / "status.json"
    suite_status = json.loads(status_path.read_text(encoding="utf-8"))
    if suite_status.get("status") != "completed":
        raise RuntimeError("supplemental suite must be completed before reporting")
    if not suite_status.get("reload_verified"):
        raise RuntimeError("supplemental suite root was not reload-verified")

    summaries: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for control_id in suite_status["controls_requested"]:
        summaries[control_id], arrays[control_id] = _load_control(
            suite, suite_status, control_id
        )

    comparisons: dict[str, dict[str, Any]] = {}
    for title, first, second, first_label, second_label in COMPARISONS:
        first_summary = summaries[first]
        second_summary = summaries[second]
        comparisons[title] = {
            "first": first_label,
            "second": second_label,
            "deepest_layer_final_information_x_second_minus_first": (
                second_summary["final_information_x_mean_by_layer"][-1]
                - first_summary["final_information_x_mean_by_layer"][-1]
            ),
            "deepest_layer_final_information_y_second_minus_first": (
                second_summary["final_information_y_mean_by_layer"][-1]
                - first_summary["final_information_y_mean_by_layer"][-1]
            ),
            "deepest_layer_compression_second_minus_first": (
                second_summary["information_x_compression_mean_by_layer"][-1]
                - first_summary["information_x_compression_mean_by_layer"][-1]
            ),
            "final_train_accuracy_second_minus_first": (
                second_summary["final_train_accuracy_mean"]
                - first_summary["final_train_accuracy_mean"]
            ),
        }

    payload = {
        "suite": str(suite),
        "suite_status": suite_status["status"],
        "run_kind": suite_status["run_kind"],
        "repeats_per_control": suite_status["repeats_per_control"],
        "epochs_per_control": suite_status["epochs_per_control"],
        "elapsed_seconds": suite_status["elapsed_seconds"],
        "controls": summaries,
        "comparisons": comparisons,
        "interpretation_boundary": suite_status["interpretation_boundary"],
        "reload_verified": True,
    }
    summary_path = suite / "supplemental_summary.json"
    _write_json(summary_path, payload)
    _plot_report(suite / "supplemental_comparison", arrays)

    reloaded = json.loads(summary_path.read_text(encoding="utf-8"))
    if not reloaded.get("reload_verified"):
        raise RuntimeError("supplemental summary failed reload verification")
    report_files = (
        summary_path,
        suite / "supplemental_comparison.png",
        suite / "supplemental_comparison.pdf",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in report_files):
        raise RuntimeError("supplemental report files were not persisted")

    suite_status["report"] = {
        "summary": summary_path.name,
        "figure_png": "supplemental_comparison.png",
        "figure_pdf": "supplemental_comparison.pdf",
        "reload_verified": True,
    }
    _write_json(status_path, suite_status)
    json.loads(status_path.read_text(encoding="utf-8"))
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
