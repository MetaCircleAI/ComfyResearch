"""API helpers: fit curve and return JSON-serializable results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from comfy_research.lmech.lmn.finetize import (
    PiecewiseFormula,
    _eval_piecewise,
    finetize,
    finetize_at_breakpoints_forced,
    finetize_points,
)
from comfy_research.lmech.lmn.json_util import finite_float, json_safe
from comfy_research.lmech.lmn.model import LMN
from comfy_research.lmech.lmn.primitives import PrimitiveBank
from comfy_research.lmech.lmn.train import ProgressCallback, TrainConfig, TrainProgress, train_auto_k, train_lmn

MECHANISM_LABELS = {
    "exp": "指数衰减 (Exponential decay)",
    "exp_growth": "指数爆炸 (Exponential growth)",
    "const": "平台期 (Plateau)",
    "sigmoid": "Sigmoid 相变 (Phase transition)",
    "gaussian_spike": "Gaussian 尖峰 (Gaussian spike)",
    "power": "负幂律 (Negative power law)",
    "pos_power": "正幂律 (Positive power law)",
    "linear": "线性 (Linear)",
}

MECHANISM_OPTIONS = [
    {"id": key, "label": label}
    for key, label in MECHANISM_LABELS.items()
]

# Overall progress scale for web UI: train 0–70%, finetize 70–100%.
_PROGRESS_TOTAL = 1000
_TRAIN_END = 700


def _wrap_train_progress(
    callback: Optional[ProgressCallback],
) -> Optional[ProgressCallback]:
    if callback is None:
        return None

    def wrapped(p: TrainProgress) -> None:
        callback(
            TrainProgress(
                current=int(p.fraction * _TRAIN_END),
                total=_PROGRESS_TOTAL,
                phase=p.phase,
                restart=p.restart,
                num_restarts=p.num_restarts,
                epoch=p.epoch,
                num_epochs=p.num_epochs,
                k=p.k,
                k_max=p.k_max,
                mse=p.mse,
                message=p.message or "Training LMN…",
            )
        )

    return wrapped


def _formula_to_dict(formula: PiecewiseFormula) -> Dict[str, Any]:
    segments = []
    for seg in formula.segments:
        segments.append(
            {
                "phase_index": seg.phase_index,
                "t_start": seg.t_start,
                "t_end": seg.t_end,
                "mechanism": seg.mechanism,
                "mechanism_label": MECHANISM_LABELS.get(
                    seg.mechanism, seg.mechanism
                ),
                "formula": seg.formula,
                "latex": seg.latex,
                "segment_r2": finite_float(seg.segment_r2, default=0.0),
                "params": {
                    k: finite_float(float(v), default=0.0)
                    for k, v in seg.params.items()
                },
            }
        )
    mean_segment_r2 = (
        finite_float(
            float(np.mean([finite_float(seg.segment_r2, default=0.0) for seg in formula.segments])),
            default=0.0,
        )
        if formula.segments
        else 0.0
    )
    return {
        "breakpoints": [float(b) for b in formula.breakpoints],
        "segments": segments,
        "global_r2": finite_float(formula.global_r2, default=0.0),
        "mean_segment_r2": mean_segment_r2,
        "num_phases": len(formula.segments),
        "t_max": float(formula.t_max),
    }


def fit_curve_from_points(
    points: List[Dict[str, float]],
    num_phases: int = 4,
    auto_k: bool = False,
    k_max: int = 6,
    fast: bool = False,
    epochs: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    simplicity: float = 0.25,
) -> Dict[str, Any]:
    """
    Fit LMN to user-provided (t, loss) points.

    Args:
        points: list of {"t": float, "loss": float}
        num_phases: K when auto_k is False
        auto_k: search K=1..k_max by BIC
        fast: use shorter training for interactive UI
        simplicity: 0 = pure fit quality; 1 = strongly prefer simpler primitives
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points")

    t_arr = np.array([p["t"] for p in points], dtype=float)
    y_arr = np.array([p["loss"] for p in points], dtype=float)

    # Sort and dedupe by t
    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]
    _, uniq = np.unique(t_arr, return_index=True)
    t_arr, y_arr = t_arr[uniq], y_arr[uniq]

    t = torch.tensor(t_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)

    if fast:
        cfg = TrainConfig.fast(
            num_phases=num_phases,
            epochs=epochs or 800,
            restarts=1,
            beta_end=150.0,
        )
    else:
        cfg = TrainConfig(num_phases=num_phases, epochs=epochs or 2000, restarts=2)

    train_cb = _wrap_train_progress(progress_callback)

    if auto_k:
        result = train_auto_k(
            t, y, k_max=k_max, base_config=cfg, progress_callback=train_cb
        )
    else:
        result = train_lmn(t, y, cfg, progress_callback=train_cb)

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=_TRAIN_END,
                total=_PROGRESS_TOTAL,
                phase="train_done",
                k=result.model.num_phases,
                mse=result.best_mse,
                message="Training complete",
            )
        )

        def finetize_cb(step: int, total: int) -> None:
            span = _PROGRESS_TOTAL - _TRAIN_END
            current = _TRAIN_END + int(span * step / max(total, 1))
            progress_callback(
                TrainProgress(
                    current=current,
                    total=_PROGRESS_TOTAL,
                    phase="finetize",
                    k=result.model.num_phases,
                    message=f"Finetizing formula ({step}/{total})",
                )
            )
    else:
        finetize_cb = None

    formula = finetize(
        result.model,
        t,
        y,
        simplicity=simplicity,
        progress_callback=finetize_cb,
    )

    # Dense curves for overlay
    t_dense = torch.linspace(float(t.min()), float(t.max()), 200)
    bank = PrimitiveBank()
    y_finetized = _eval_piecewise(bank, formula.segments, t_dense.numpy())
    with torch.no_grad():
        y_soft = result.model.predict(t_dense, beta=150.0).numpy()

    def _series(t_vals, y_vals):
        return [
            {
                "t": finite_float(float(a), default=0.0),
                "loss": finite_float(float(b), default=0.0),
            }
            for a, b in zip(t_vals, y_vals)
        ]

    out = _formula_to_dict(formula)
    out.update(
        {
            "data": _series(t_arr, y_arr),
            "fitted": _series(t_dense.numpy(), y_finetized),
            "soft": _series(t_dense.numpy(), y_soft),
            "train_mse": finite_float(float(result.best_mse)),
            "bic": finite_float(float(result.bic)) if result.bic is not None else None,
        }
    )
    return json_safe(out)


def finetize_curve_from_points(
    points: List[Dict[str, float]],
    num_phases: int = 4,
    breakpoints: Optional[List[float]] = None,
    lock_breakpoints: bool = False,
    simplicity: float = 0.25,
    progress_callback: Optional[ProgressCallback] = None,
    soft_curve: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Refit piecewise formulas; optionally lock boundary positions."""
    if len(points) < 5:
        raise ValueError("Need at least 5 points")

    t_arr = np.array([p["t"] for p in points], dtype=float)
    y_arr = np.array([p["loss"] for p in points], dtype=float)

    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]
    _, uniq = np.unique(t_arr, return_index=True)
    t_arr, y_arr = t_arr[uniq], y_arr[uniq]

    t = torch.tensor(t_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)

    def finetize_cb(step: int, total: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            TrainProgress(
                current=int(_PROGRESS_TOTAL * step / max(total, 1)),
                total=_PROGRESS_TOTAL,
                phase="finetize",
                k=num_phases,
                message=f"Finetizing formula ({step}/{total})",
            )
        )

    formula = finetize_points(
        t,
        y,
        num_phases=num_phases,
        breakpoints=breakpoints,
        lock_breakpoints=lock_breakpoints,
        simplicity=simplicity,
        progress_callback=finetize_cb if progress_callback else None,
    )

    t_dense = torch.linspace(float(t.min()), float(t.max()), 200)
    bank = PrimitiveBank()
    y_finetized = _eval_piecewise(bank, formula.segments, t_dense.numpy())

    def _series(t_vals, y_vals):
        return [
            {
                "t": finite_float(float(a), default=0.0),
                "loss": finite_float(float(b), default=0.0),
            }
            for a, b in zip(t_vals, y_vals)
        ]

    out = _formula_to_dict(formula)
    out.update(
        {
            "data": _series(t_arr, y_arr),
            "fitted": _series(t_dense.numpy(), y_finetized),
            "soft": soft_curve or [],
        }
    )
    return json_safe(out)


def refit_segments_with_mechanisms(
    points: List[Dict[str, float]],
    breakpoints: List[float],
    segment_mechanisms: List[str],
    soft_curve: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Refit piecewise formulas with user-selected mechanisms per segment."""
    if len(points) < 5:
        raise ValueError("Need at least 5 points")
    if not breakpoints and len(segment_mechanisms) != 1:
        raise ValueError("breakpoints required when num_phases > 1")
    if len(breakpoints) != len(segment_mechanisms) - 1:
        raise ValueError("breakpoints length must be num_phases - 1")

    t_arr = np.array([p["t"] for p in points], dtype=float)
    y_arr = np.array([p["loss"] for p in points], dtype=float)

    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]
    _, uniq = np.unique(t_arr, return_index=True)
    t_arr, y_arr = t_arr[uniq], y_arr[uniq]

    t = torch.tensor(t_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)

    formula = finetize_at_breakpoints_forced(
        t,
        y,
        breakpoints,
        segment_mechanisms,
    )

    t_dense = torch.linspace(float(t.min()), float(t.max()), 200)
    bank = PrimitiveBank()
    y_finetized = _eval_piecewise(bank, formula.segments, t_dense.numpy())

    def _series(t_vals, y_vals):
        return [
            {
                "t": finite_float(float(a), default=0.0),
                "loss": finite_float(float(b), default=0.0),
            }
            for a, b in zip(t_vals, y_vals)
        ]

    out = _formula_to_dict(formula)
    out.update(
        {
            "data": _series(t_arr, y_arr),
            "fitted": _series(t_dense.numpy(), y_finetized),
            "soft": soft_curve or [],
        }
    )
    return json_safe(out)


def template_points(name: str, n: int = 80) -> List[Dict[str, float]]:
    """Return built-in or user-saved template curve points."""
    from comfy_research.lmech.lmn.templates import get_template_points

    return get_template_points(name, n=n)
