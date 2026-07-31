"""Web API helpers for LPD curve segment prediction."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from comfy_research.lmech.lmn.json_util import json_safe
from comfy_research.lmech.lmn.serve import finetize_curve_from_points
from comfy_research.lmech.lmn.train import ProgressCallback, TrainProgress
from comfy_research.lmech.lpd.checkpoint import is_legacy_checkpoint_file
from comfy_research.lmech.lpd.auto_k import predict_auto_k
from comfy_research.lmech.lpd.model import CurveDETR
from comfy_research.lmech.lpd.predict import PredictConfig, load_checkpoint, predict_segments, save_checkpoint
from comfy_research.lmech.lpd.synthetic import generate_fake_dataset
from comfy_research.lmech.lpd.train import TrainLPDConfig, train_lpd

DEFAULT_CHECKPOINT = Path("data/lpd/checkpoints/curve_detr.pt")

_model_cache: Optional[CurveDETR] = None
_config_cache: Optional[PredictConfig] = None

_PROGRESS_TOTAL = 1000
_SEGMENT_END = 200


def train_and_save_checkpoint(
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    data_dir: str = "data/lpd/fake",
    train_steps: int = 800,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """Train CurveDETR and persist weights for inference."""
    data_path = Path(data_dir)
    if not data_path.is_dir() or not (data_path / "manifest.json").is_file():
        generate_fake_dataset(data_path, num_samples=200, seed=42, noise_std=0.002)
    train_config = TrainLPDConfig(
        data_dir=str(data_path),
        train_steps=train_steps,
        device=device,
    )
    model, history = train_lpd(train_config)
    pred_config = PredictConfig(
        seq_len=train_config.seq_len,
        num_queries=train_config.num_queries,
        d_model=train_config.d_model,
        device=device,
    )
    save_checkpoint(model, checkpoint, pred_config)
    return history


def get_lpd_model(
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    auto_train: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> tuple[CurveDETR, PredictConfig]:
    global _model_cache, _config_cache

    checkpoint = Path(checkpoint)
    if _model_cache is not None and _config_cache is not None:
        return _model_cache, _config_cache

    if auto_train and not checkpoint.is_file():
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    current=10,
                    total=_PROGRESS_TOTAL,
                    phase="bootstrap",
                    message="Preparing LPD checkpoint (first run)…",
                )
            )
        train_and_save_checkpoint(checkpoint)
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    current=180,
                    total=_PROGRESS_TOTAL,
                    phase="bootstrap",
                    message="LPD checkpoint ready",
                )
            )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"LPD checkpoint not found at {checkpoint}. "
            "Copy an existing curve_detr.pt to data/lpd/checkpoints/curve_detr.pt "
            "or run: python scripts/train_lpd.py --save-checkpoint"
        )

    if auto_train and is_legacy_checkpoint_file(checkpoint):
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    current=10,
                    total=_PROGRESS_TOTAL,
                    phase="bootstrap",
                    message="Updating legacy LPD checkpoint…",
                )
            )
        train_and_save_checkpoint(checkpoint)
        _model_cache = None
        _config_cache = None
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    current=180,
                    total=_PROGRESS_TOTAL,
                    phase="bootstrap",
                    message="LPD checkpoint ready",
                )
            )

    model, config = load_checkpoint(checkpoint, device="cpu")
    _model_cache = model
    _config_cache = config
    return model, config


def predict_curve_from_points(
    points: List[Dict[str, float]],
    num_phases: int = 3,
    auto_k: bool = False,
    k_max: int = 6,
    auto_k_target: float = 0.99,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    auto_train: bool = False,
    simplicity: float = 0.25,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    LPD segments the curve; per-segment formulas come from locked finetize.

    When ``auto_k`` is True, tries K=1..k_max and stops at the first global R² ≥ 0.99.
    """
    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=1,
                total=_PROGRESS_TOTAL,
                phase="bootstrap",
                k=num_phases,
                message="Loading LPD model…",
            )
        )
    model, config = get_lpd_model(
        checkpoint,
        auto_train=auto_train,
        progress_callback=progress_callback,
    )

    if auto_k:
        return predict_auto_k(
            model,
            points,
            config,
            k_max=k_max,
            simplicity=simplicity,
            r2_target=auto_k_target,
            progress_callback=progress_callback,
        )

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=0,
                total=_PROGRESS_TOTAL,
                phase="segment",
                k=num_phases,
                message="LPD segmenting curve…",
            )
        )

    partition = predict_segments(model, points, config, num_phases=num_phases)

    if progress_callback is not None:
        progress_callback(
            TrainProgress(
                current=_SEGMENT_END,
                total=_PROGRESS_TOTAL,
                phase="segment_done",
                k=num_phases,
                message="Segmentation complete",
            )
        )

        def finetize_cb(progress: TrainProgress) -> None:
            span = _PROGRESS_TOTAL - _SEGMENT_END
            current = _SEGMENT_END + int(span * progress.fraction)
            progress_callback(
                TrainProgress(
                    current=current,
                    total=_PROGRESS_TOTAL,
                    phase="finetize",
                    k=num_phases,
                    message=progress.message or "Finetizing segments…",
                )
            )
    else:
        finetize_cb = None

    bps = partition["breakpoints"] or None
    return finetize_curve_from_points(
        points,
        num_phases=partition["num_phases"],
        breakpoints=bps,
        lock_breakpoints=bool(bps),
        simplicity=simplicity,
        progress_callback=finetize_cb,
    )


def predict_curves_batch_from_points(
    curves_points: List[List[Dict[str, float]]],
    num_phases: int = 3,
    auto_k: bool = False,
    k_max: int = 6,
    auto_k_target: float = 0.99,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    auto_train: bool = False,
    simplicity: float = 0.25,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run LPD predict on multiple curves in parallel on CPU (thread pool).

    The shared CurveDETR checkpoint is loaded once; workers reuse the cached model.
    """
    n = len(curves_points)
    if n == 0:
        return []

    predict_kwargs = dict(
        num_phases=num_phases,
        auto_k=auto_k,
        k_max=k_max,
        auto_k_target=auto_k_target,
        checkpoint=checkpoint,
        auto_train=auto_train,
        simplicity=simplicity,
    )

    if n == 1:
        try:
            result = predict_curve_from_points(curves_points[0], **predict_kwargs)
            return [{"ok": True, "result": result}]
        except Exception as e:
            return [{"ok": False, "error": str(e)}]

    get_lpd_model(checkpoint, auto_train=auto_train)

    workers = max_workers or min(n, os.cpu_count() or 4)

    def _predict_indexed(
        item: tuple[int, List[Dict[str, float]]],
    ) -> tuple[int, Dict[str, Any]]:
        idx, pts = item
        try:
            result = predict_curve_from_points(pts, **predict_kwargs)
            return idx, {"ok": True, "result": result}
        except Exception as e:
            return idx, {"ok": False, "error": str(e)}

    out: List[Optional[Dict[str, Any]]] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, item in pool.map(_predict_indexed, enumerate(curves_points)):
            out[idx] = item
    return out  # type: ignore[return-value]
