"""Inference and decode for LPD breakpoint predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from comfy_research.lmech.lpd.checkpoint import is_legacy_checkpoint, load_state_dict_compat
from comfy_research.lmech.lpd.dataset import curve_feature_channels
from comfy_research.lmech.lpd.features import merge_breakpoints_hybrid, sanitize_breakpoints
from comfy_research.lmech.lpd.model import CurveDETR


@dataclass
class PredictConfig:
    seq_len: int = 256
    num_queries: int = 8
    d_model: int = 128
    num_phases: int = 3
    device: str = "cpu"
    use_derivative_snap: bool = True
    snap_radius_frac: float = 0.15
    min_score_frac: float = 0.15


def _sort_dedupe_points(
    points: List[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    t_arr = np.array([p["t"] for p in points], dtype=float)
    y_arr = np.array([p["loss"] for p in points], dtype=float)
    order = np.argsort(t_arr)
    t_arr, y_arr = t_arr[order], y_arr[order]
    _, uniq = np.unique(t_arr, return_index=True)
    return t_arr[uniq], y_arr[uniq]


def resample_curve(
    t: torch.Tensor,
    y: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    t_new = torch.linspace(t[0], t[-1], seq_len)
    y_new = torch.zeros(seq_len)
    for i in range(seq_len):
        ti = t_new[i]
        j = torch.searchsorted(t, ti).clamp(1, len(t) - 1)
        t0, t1 = t[j - 1], t[j]
        y0, y1 = y[j - 1], y[j]
        frac = (ti - t0) / (t1 - t0) if t1 > t0 else 0.0
        y_new[i] = y0 + frac * (y1 - y0)
    return y_new


def normalize_y(y: torch.Tensor) -> torch.Tensor:
    y_min, y_max = y.min(), y.max()
    span = y_max - y_min
    if span < 1e-6 or not torch.isfinite(span):
        return torch.zeros_like(y)
    return (y - y_min) / span


def preprocess_points(
    points: List[Dict[str, float]],
    seq_len: int = 256,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float, float]:
    t_arr, y_arr = _sort_dedupe_points(points)
    t = torch.tensor(t_arr, dtype=torch.float32)
    y = torch.tensor(y_arr, dtype=torch.float32)
    t_min = float(t[0].item())
    t_max = float(t[-1].item())
    y_resampled = resample_curve(t, y, seq_len)
    y_norm = normalize_y(y_resampled)
    curve = curve_feature_channels(y_norm).unsqueeze(0)
    return curve, t_arr, y_arr, t_min, t_max


def decode_breakpoint_predictions(
    pred_breakpoints: torch.Tensor,
    pred_scores: torch.Tensor,
    t_min: float,
    t_max: float,
    num_phases: int,
) -> Tuple[List[float], List[float]]:
    """
    Select ``num_phases - 1`` breakpoints from query predictions.

    Returns absolute breakpoint times and their query scores.
    """
    if num_phases < 1:
        raise ValueError("num_phases must be >= 1")

    span = max(t_max - t_min, 1e-6)
    need = num_phases - 1
    if need <= 0:
        return [], []

    k = min(need, pred_breakpoints.shape[0])
    top_q = pred_scores.topk(k).indices
    norm_bps = pred_breakpoints[top_q]
    scores = pred_scores[top_q]
    order = norm_bps.argsort()
    norm_sorted = norm_bps[order]
    scores_sorted = scores[order]

    abs_bps = sanitize_breakpoints(
        [float(t_min + float(bp) * span) for bp in norm_sorted],
        t_min,
        t_max,
        count=need,
    )
    return abs_bps, [float(s) for s in scores_sorted]


def breakpoints_to_segments(
    breakpoints: List[float],
    t_min: float,
    t_max: float,
    scores: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    bounds = [t_min, *breakpoints, t_max]
    segments: List[Dict[str, Any]] = []
    for i in range(len(bounds) - 1):
        seg = {
            "start": float(bounds[i]),
            "end": float(bounds[i + 1]),
        }
        if scores is not None and i < len(scores):
            seg["score"] = scores[i]
        segments.append(seg)
    return segments


def segments_to_analysis(
    segments: List[Dict[str, Any]],
    t_min: float,
    t_max: float,
) -> Dict[str, Any]:
    out_segments: List[Dict[str, Any]] = []
    breakpoints: List[float] = []

    for i, seg in enumerate(segments):
        out_segments.append(
            {
                "phase_index": i,
                "t_start": seg["start"],
                "t_end": seg["end"],
                "score": seg.get("score"),
            }
        )
        if i < len(segments) - 1:
            breakpoints.append(float(seg["end"]))

    return {
        "segments": out_segments,
        "breakpoints": breakpoints,
        "num_phases": len(out_segments),
        "t_min": t_min,
        "t_max": t_max,
    }


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> Tuple[CurveDETR, PredictConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    config = PredictConfig(
        seq_len=int(cfg_dict.get("seq_len", 256)),
        num_queries=int(cfg_dict.get("num_queries", 8)),
        d_model=int(cfg_dict.get("d_model", 128)),
        device=device,
        use_derivative_snap=bool(cfg_dict.get("use_derivative_snap", True)),
        snap_radius_frac=float(cfg_dict.get("snap_radius_frac", 0.12)),
        min_score_frac=float(cfg_dict.get("min_score_frac", 0.2)),
    )
    in_channels = int(cfg_dict.get("in_channels", 3))
    model = CurveDETR(
        num_queries=config.num_queries,
        d_model=config.d_model,
        in_channels=in_channels,
    )
    skipped, loaded_bp = load_state_dict_compat(model, state_dict)
    if is_legacy_checkpoint(state_dict) or not loaded_bp:
        import warnings

        warnings.warn(
            "LPD checkpoint is outdated (box-head / 1-channel). "
            "Loaded compatible backbone weights only; retrain for best results:\n"
            "  python scripts/train_lpd.py --train-steps 3000 --save-checkpoint",
            stacklevel=2,
        )
        if skipped and __debug__:
            for msg in skipped[:5]:
                print(f"  checkpoint: {msg}", flush=True)
    model.to(device)
    model.eval()
    return model, config


def save_checkpoint(
    model: CurveDETR,
    path: str | Path,
    config: PredictConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "seq_len": config.seq_len,
                "num_queries": config.num_queries,
                "d_model": config.d_model,
                "in_channels": model.in_channels,
                "use_derivative_snap": config.use_derivative_snap,
                "snap_radius_frac": config.snap_radius_frac,
                "min_score_frac": config.min_score_frac,
            },
        },
        path,
    )


@torch.no_grad()
def predict_segments(
    model: CurveDETR,
    points: List[Dict[str, float]],
    config: Optional[PredictConfig] = None,
    num_phases: Optional[int] = None,
) -> Dict[str, Any]:
    if config is None:
        config = PredictConfig()
    if len(points) < 5:
        raise ValueError("Need at least 5 points")

    k = num_phases if num_phases is not None else config.num_phases
    if k < 1:
        raise ValueError("num_phases must be >= 1")
    if k > config.num_queries:
        raise ValueError(
            f"num_phases={k} exceeds model num_queries={config.num_queries}"
        )

    device = torch.device(config.device)
    model = model.to(device)

    curve, t_arr, y_arr, t_min, t_max = preprocess_points(
        points, seq_len=config.seq_len
    )
    curve = curve.to(device)

    outputs = model(curve)
    neural_bps, neural_scores = decode_breakpoint_predictions(
        outputs["pred_breakpoints"][0],
        outputs["pred_scores"][0],
        t_min,
        t_max,
        num_phases=k,
    )

    if config.use_derivative_snap:
        breakpoints = merge_breakpoints_hybrid(
            neural_bps,
            neural_scores,
            t_arr,
            y_arr,
            t_min,
            t_max,
            num_phases=k,
            snap_radius_frac=config.snap_radius_frac,
            min_score_frac=config.min_score_frac,
        )
    else:
        breakpoints = neural_bps

    segments = breakpoints_to_segments(breakpoints, t_min, t_max, neural_scores)
    return segments_to_analysis(segments, t_min, t_max)
