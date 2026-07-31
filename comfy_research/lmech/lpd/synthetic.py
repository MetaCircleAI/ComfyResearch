"""Synthetic curve + segment generator (placeholder before LMN-assisted labeling)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from comfy_research.lmech.lpd.schema import CurveSample, SegmentAnnotation, SEGMENT_TYPES

_EXP_CLIP = 20.0
_Y_CLIP = 50.0
_MAX_SHAPE_AMP = 2.0


def _exp_clip(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -_EXP_CLIP, _EXP_CLIP))


def _eval_segment_shape(
    t_local: np.ndarray,
    seg_type: str,
    rng: np.random.Generator,
    t_span: float,
) -> np.ndarray:
    """
    Segment shape anchored at the first sample (value 0 at t_local[0]).

    Caller stitches with: y_level + shape - shape[0].
    """
    t_local = np.asarray(t_local, dtype=float)
    t_span = max(float(t_span), 1.0)

    if seg_type == "const":
        return np.zeros_like(t_local)
    if seg_type == "linear":
        a = rng.uniform(-0.0002, 0.0002)
        return a * (t_local - t_local[0])
    if seg_type == "exp":
        a = rng.uniform(0.2, 1.0)
        r = rng.uniform(0.001, 0.05)
        tau = rng.uniform(-0.2 * t_span, 0.2 * t_span)
        expo = -r * (t_local + tau)
        ref = -r * (t_local[0] + tau)
        return a * (_exp_clip(expo) - _exp_clip(ref))
    if seg_type == "exp_growth":
        a = rng.uniform(0.01, 0.2)
        r_max = min(0.005, 4.0 / t_span)
        r = rng.uniform(0.0001, max(r_max, 0.0001))
        tau = rng.uniform(-0.2 * t_span, 0.2 * t_span)
        expo = r * (t_local + tau)
        ref = r * (t_local[0] + tau)
        return a * (_exp_clip(expo) - _exp_clip(ref))
    if seg_type == "sigmoid":
        a = rng.uniform(0.05, 0.3)
        k = rng.uniform(0.005, 0.03)
        tau = rng.uniform(0.0, t_span)
        logistic = 1.0 / (
            1.0 + np.exp(-np.clip(k * (t_local - tau), -500, 500))
        )
        ref = 1.0 / (1.0 + np.exp(-np.clip(k * (t_local[0] - tau), -500, 500)))
        return a * (logistic - ref)
    if seg_type == "gaussian_spike":
        a = rng.uniform(0.05, 0.35)
        k = rng.uniform(1e-6, 5e-5)
        peak = rng.uniform(0.2 * t_span, 0.8 * t_span)
        tau = -peak
        expo = -k * (t_local + tau) ** 2
        ref = np.exp(np.clip(-k * (t_local[0] + tau) ** 2, -500, 500))
        return a * (np.exp(np.clip(expo, -500, 500)) - ref)
    if seg_type == "power":
        a = rng.uniform(0.05, 0.3)
        alpha = rng.uniform(0.1, 0.6)
        tau = rng.uniform(1.0, 100.0)
        ref = np.power(t_local[0] + tau, -alpha)
        return a * (np.power(t_local + tau, -alpha) - ref)
    if seg_type == "pos_power":
        a = rng.uniform(0.01, 0.2)
        beta = rng.uniform(0.05, 0.3)
        tau = rng.uniform(1.0, 50.0)
        ref = np.power(max(t_local[0] + tau, 1.0), beta)
        return a * (
            np.power(np.clip(t_local + tau, 1.0, None), beta) - ref
        )
    return np.zeros_like(t_local)


def _segment_mask(
    t: np.ndarray,
    start: float,
    end: float,
    is_last: bool,
) -> np.ndarray:
    if is_last:
        return (t >= start) & (t <= end)
    return (t >= start) & (t < end)


def generate_random_curve(
    t_max: float = 10000.0,
    num_points: int = 200,
    num_segments: Optional[int] = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> CurveSample:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, num_points)
    k = num_segments or rng.integers(2, 5)
    breakpoints = np.sort(
        rng.choice(np.linspace(0.2 * t_max, 0.8 * t_max, 20), size=k - 1, replace=False)
    )
    bounds = np.concatenate([[0.0], breakpoints, [t_max]])

    y = np.zeros_like(t)
    segments: List[SegmentAnnotation] = []
    y_level = rng.uniform(0.2, 0.8)

    for i in range(k):
        start, end = bounds[i], bounds[i + 1]
        mask = _segment_mask(t, start, end, is_last=(i == k - 1))
        if not np.any(mask):
            continue

        seg_type = rng.choice(SEGMENT_TYPES)
        t_seg = t[mask]
        t_local = t_seg - start
        t_span = end - start
        shape = _eval_segment_shape(t_local, seg_type, rng, t_span)
        seg_y = y_level + shape - shape[0]
        seg_y = np.clip(seg_y, 0.0, _Y_CLIP)
        y[mask] = seg_y
        y_level = float(seg_y[-1])
        segments.append(
            SegmentAnnotation(type=seg_type, start=float(start), end=float(end))
        )

    if noise_std > 0:
        y = y + rng.normal(0, noise_std, size=y.shape)
    y = np.clip(y, 0.0, _Y_CLIP)

    return CurveSample(
        id=f"sample_{seed:04d}",
        t=t.tolist(),
        y=y.tolist(),
        segments=segments,
    )


def generate_fake_dataset(
    out_dir: str | Path,
    num_samples: int = 10,
    seed: int = 42,
    **curve_kwargs,
) -> List[CurveSample]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples: List[CurveSample] = []
    for i in range(num_samples):
        sample = generate_random_curve(seed=seed + i, **curve_kwargs)
        sample.id = f"sample_{i:04d}"
        samples.append(sample)
        with open(out / f"{sample.id}.json", "w") as f:
            json.dump(sample.to_dict(), f, indent=2)

    manifest = {"samples": [s.id for s in samples], "num_samples": len(samples)}
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return samples


def interval_to_box(start: float, end: float, t_min: float, t_max: float) -> Tuple[float, float]:
    """Convert [start, end] to normalized DETR box (center, width)."""
    span = max(t_max - t_min, 1e-6)
    cx = (start + end) / 2.0
    w = end - start
    return (cx - t_min) / span, w / span
