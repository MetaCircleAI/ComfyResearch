"""Auto-select partition count K for LPD inference."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from comfy_research.lmech.lmn.serve import finetize_curve_from_points
from comfy_research.lmech.lmn.train import ProgressCallback, TrainProgress
from comfy_research.lmech.lpd.model import CurveDETR
from comfy_research.lmech.lpd.predict import PredictConfig, predict_segments

_PROGRESS_TOTAL = 1000
AUTO_K_R2_TARGET = 0.99


def _mean_segment_r2(result: Dict[str, Any]) -> float:
    segments = result.get("segments") or []
    if not segments:
        return float(result.get("global_r2", 0.0))
    return float(np.mean([float(seg.get("segment_r2", 0.0)) for seg in segments]))


def _global_r2(result: Dict[str, Any]) -> float:
    return float(result.get("global_r2", 0.0))


def _segment_and_finetize(
    model: CurveDETR,
    points: List[Dict[str, float]],
    config: PredictConfig,
    num_phases: int,
    k_max: int,
    simplicity: float,
    progress_callback: Optional[ProgressCallback],
    progress_base: int,
    progress_span: int,
) -> Dict[str, Any]:
    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=progress_base,
                total=_PROGRESS_TOTAL,
                phase="auto_k",
                k=num_phases,
                k_max=k_max,
                message=f"LPD Auto-K: segmenting K={num_phases}…",
            )
        )

    partition = predict_segments(model, points, config, num_phases=num_phases)
    segment_end = progress_base + progress_span // 3

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=segment_end,
                total=_PROGRESS_TOTAL,
                phase="auto_k",
                k=num_phases,
                k_max=k_max,
                message=f"LPD Auto-K: finetizing K={num_phases}…",
            )
        )

    def finetize_cb(progress: TrainProgress) -> None:
        if progress_callback is None:
            return
        fin_span = progress_span - (segment_end - progress_base)
        current = segment_end + int(fin_span * progress.fraction)
        progress_callback(
            TrainProgress(
                current=current,
                total=_PROGRESS_TOTAL,
                phase="auto_k",
                k=num_phases,
                k_max=k_max,
                message=progress.message or f"Finetizing K={num_phases}…",
            )
        )

    bps = partition["breakpoints"] or None
    result = finetize_curve_from_points(
        points,
        num_phases=partition["num_phases"],
        breakpoints=bps,
        lock_breakpoints=bool(bps),
        simplicity=simplicity,
        progress_callback=finetize_cb if progress_callback else None,
    )
    return result


def predict_auto_k(
    model: CurveDETR,
    points: List[Dict[str, float]],
    config: PredictConfig,
    k_max: int = 6,
    simplicity: float = 0.25,
    r2_target: float = AUTO_K_R2_TARGET,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Try K=1,2,… and stop at the first K whose mean segment R² ≥ target.

    If no K up to k_max reaches the target, keep the K with the highest mean segment R².
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points")

    k_hi = min(max(int(k_max), 1), config.num_queries)
    per_k_span = _PROGRESS_TOTAL // max(k_hi, 1)

    best_k = 1
    best_mean_r2 = float("-inf")
    best_result: Optional[Dict[str, Any]] = None
    trials: List[Dict[str, float]] = []
    selected_result: Optional[Dict[str, Any]] = None
    selected_k: Optional[int] = None

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=0,
                total=_PROGRESS_TOTAL,
                phase="auto_k",
                k=1,
                k_max=k_hi,
                message=f"LPD Auto-K: mean seg R² ≥ {r2_target:.2f}, K=1..{k_hi}",
            )
        )

    for k in range(1, k_hi + 1):
        result = _segment_and_finetize(
            model,
            points,
            config,
            num_phases=k,
            k_max=k_hi,
            simplicity=simplicity,
            progress_callback=progress_callback,
            progress_base=(k - 1) * per_k_span,
            progress_span=per_k_span,
        )
        mean_r2 = _mean_segment_r2(result)
        global_r2 = _global_r2(result)
        result["mean_segment_r2"] = mean_r2
        trials.append({"k": float(k), "mean_r2": mean_r2, "global_r2": global_r2})
        print(
            f"[LPD Auto-K] K={k} global R²={global_r2:.4f} mean R²={mean_r2:.4f}",
            flush=True,
        )

        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    current=min(k * per_k_span, _PROGRESS_TOTAL),
                    total=_PROGRESS_TOTAL,
                    phase="auto_k",
                    k=k,
                    k_max=k_hi,
                    message=(
                        f"LPD Auto-K: K={k} mean seg R²={mean_r2:.3f}"
                        + (" ✓" if mean_r2 >= r2_target else "")
                    ),
                )
            )

        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_k = k
            best_result = result

        if mean_r2 >= r2_target:
            selected_k = k
            selected_result = result
            print(
                f"[LPD Auto-K] early stop at K={k} "
                f"(mean seg R²={mean_r2:.4f} ≥ {r2_target:.2f}), "
                f"skipped K={k + 1}..{k_hi}",
                flush=True,
            )
            if progress_callback is not None:
                progress_callback(
                    TrainProgress(
                        current=_PROGRESS_TOTAL,
                        total=_PROGRESS_TOTAL,
                        phase="auto_k",
                        k=k,
                        k_max=k_hi,
                        message=(
                            f"LPD Auto-K: stopping early at K={k} "
                            f"(mean seg R²={mean_r2:.3f} ≥ {r2_target:.2f})"
                        ),
                    )
                )
            break

    if selected_result is None:
        assert best_result is not None
        selected_result = best_result
        selected_k = best_k

    selected_result["auto_k"] = True
    selected_result["selected_k"] = selected_k
    selected_result["mean_segment_r2"] = _mean_segment_r2(selected_result)
    selected_result["auto_k_target"] = r2_target
    selected_result["auto_k_trials"] = trials
    selected_result["auto_k_num_trials"] = len(trials)
    selected_result["auto_k_stopped_early"] = (
        _mean_segment_r2(selected_result) >= r2_target
    )

    if not selected_result["auto_k_stopped_early"]:
        print(
            f"[LPD Auto-K] no early stop — tried K=1..{k_hi}, "
            f"best K={selected_k} (mean seg R²={_mean_segment_r2(selected_result):.4f})",
            flush=True,
        )

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=_PROGRESS_TOTAL,
                total=_PROGRESS_TOTAL,
                phase="segment_done",
                k=selected_k,
                k_max=k_hi,
                message=(
                    f"Auto-K selected K={selected_k} "
                    f"(mean seg R²={_mean_segment_r2(selected_result):.3f})"
                ),
            )
        )

    return selected_result
