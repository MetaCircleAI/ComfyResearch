"""Derivative-based changepoint features for LPD boundary refinement."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _smooth_y(y: np.ndarray, n: int) -> np.ndarray:
    window = max(3, min(n // 10 | 1, 9))
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _merge_candidate_scores(
    scored: List[Tuple[float, float]],
    span: float,
    merge_frac: float = 0.05,
) -> List[Tuple[float, float]]:
    if not scored:
        return []
    scored = sorted(scored, key=lambda x: -x[0])
    merge_dist = merge_frac * span
    merged: List[Tuple[float, float]] = []
    for score, tp in scored:
        replaced = False
        for j, (ms, mt) in enumerate(merged):
            if abs(tp - mt) < merge_dist:
                if score > ms:
                    merged[j] = (score, tp)
                replaced = True
                break
        if not replaced:
            merged.append((score, tp))
    merged.sort(key=lambda x: -x[0])
    return merged


def _ranked_local_extrema(
    t: np.ndarray,
    y: np.ndarray,
    margin_frac: float = 0.05,
    min_prominence_frac: float = 0.04,
) -> List[Tuple[float, float]]:
    """Local minima / maxima as breakpoint candidates (score = prominence)."""
    from scipy.signal import find_peaks

    n = len(t)
    if n < 5:
        return []

    t_min, t_max = float(t[0]), float(t[-1])
    span = max(t_max - t_min, 1.0)
    margin = margin_frac * span

    y_smooth = _smooth_y(y, n)
    y_range = float(y_smooth.max() - y_smooth.min())
    if y_range < 1e-9:
        return []

    prominence = y_range * min_prominence_frac
    distance = max(3, n // 20)

    scored: List[Tuple[float, float]] = []
    for signal in (-y_smooth, y_smooth):
        peaks, peak_props = find_peaks(
            signal, prominence=prominence, distance=distance
        )
        for i, prom in zip(peaks, peak_props["prominences"]):
            tp = float(t[i])
            if t_min + margin < tp < t_max - margin:
                scored.append((float(prom), tp))
    return scored


def ranked_derivative_changepoints(
    t: np.ndarray,
    y: np.ndarray,
    margin_frac: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Rank candidate breakpoints by structural change in the curve.

    Combines |Δ(dy/dt)| (slope kinks) and |d²y/dt²| peaks.
    Returns ``(score, t)`` sorted by score descending.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    if n < 5:
        return []

    t_min, t_max = float(t[0]), float(t[-1])
    span = max(t_max - t_min, 1.0)
    margin = margin_frac * span

    y_smooth = _smooth_y(y, n)
    dt = np.diff(t)
    dy = np.diff(y_smooth) / np.maximum(dt, 1e-12)

    scored: List[Tuple[float, float]] = []

    if len(dy) >= 2:
        ds = np.abs(np.diff(dy))
        for i, ds_val in enumerate(ds):
            tp = 0.5 * (float(t[i + 1]) + float(t[i + 2]))
            if t_min + margin < tp < t_max - margin:
                scored.append((float(ds_val) * span, tp))

    if len(dy) >= 2 and len(dt) >= 2:
        d2y = np.diff(dy) / np.maximum(dt[1:], 1e-12)
        for i, d2_val in enumerate(d2y):
            tp = float(t[i + 1])
            if t_min + margin < tp < t_max - margin:
                scored.append((abs(float(d2y[i])) * span * span, tp))

    if not scored:
        return []
    return _merge_candidate_scores(scored, span)


def ranked_breakpoint_candidates(
    t: np.ndarray,
    y: np.ndarray,
    margin_frac: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Unified ranked candidates: local extrema + slope kinks + curvature peaks.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    span = max(float(t[-1] - t[0]), 1.0)

    extrema = _ranked_local_extrema(t, y, margin_frac=margin_frac)
    deriv = ranked_derivative_changepoints(t, y, margin_frac=margin_frac)

    combined = extrema + deriv
    if not combined:
        return []

    max_score = max(s for s, _ in combined)
    if max_score <= 0:
        return _merge_candidate_scores(combined, span)

    # Scale extrema (prominence) and derivatives to comparable range.
    normalized: List[Tuple[float, float]] = []
    extrema_max = max((s for s, _ in extrema), default=0.0)
    deriv_max = max((s for s, _ in deriv), default=0.0)
    for score, tp in extrema:
        norm = score / extrema_max if extrema_max > 0 else 0.0
        normalized.append((norm * max_score, tp))
    for score, tp in deriv:
        norm = score / deriv_max if deriv_max > 0 else 0.0
        normalized.append((norm * max_score * 0.85, tp))

    return _merge_candidate_scores(normalized, span)


def sanitize_breakpoints(
    breakpoints: Sequence[float],
    t_min: float,
    t_max: float,
    count: Optional[int] = None,
    min_gap_frac: float = 1 / 32,
) -> List[float]:
    """Sort breakpoints and enforce minimum spacing on ``(t_min, t_max)``."""
    span = max(t_max - t_min, 1e-6)
    need = count if count is not None else len(breakpoints)
    if need <= 0:
        return []

    min_gap = span * min_gap_frac
    raw = sorted(float(b) for b in breakpoints)[:need]

    while len(raw) < need:
        raw.append(t_min + span * (len(raw) + 1) / (need + 1))

    out: List[float] = []
    prev = t_min
    for i, bp in enumerate(raw):
        remaining = need - i - 1
        max_bp = t_max - remaining * min_gap
        clamped = max(prev + min_gap, min(max_bp, bp))
        out.append(clamped)
        prev = clamped
    return out


def _select_spread_breakpoints(
    candidates: List[Tuple[float, float]],
    need: int,
    t_min: float,
    t_max: float,
    min_gap_frac: float = 1 / 32,
) -> List[float]:
    """Greedy pick ``need`` breakpoints with highest score and min spacing."""
    span = max(t_max - t_min, 1e-6)
    min_gap = span * min_gap_frac
    pool = sorted(candidates, key=lambda x: -x[0])
    chosen: List[float] = []

    for _, tp in pool:
        if any(abs(tp - c) < min_gap for c in chosen):
            continue
        chosen.append(float(tp))
        if len(chosen) == need:
            break

    if len(chosen) < need:
        fallback = [t_min + span * (i + 1) / (need + 1) for i in range(need)]
        for tp in fallback:
            if len(chosen) >= need:
                break
            if not any(abs(tp - c) < min_gap for c in chosen):
                chosen.append(tp)

    return sanitize_breakpoints(sorted(chosen), t_min, t_max, need, min_gap_frac)


def merge_breakpoints_hybrid(
    neural_bps: Sequence[float],
    neural_scores: Optional[Sequence[float]],
    t: np.ndarray,
    y: np.ndarray,
    t_min: float,
    t_max: float,
    num_phases: int,
    neural_weight: float = 0.35,
    snap_radius_frac: float = 0.15,
    min_score_frac: float = 0.15,
) -> List[float]:
    """
    Derivative-first hybrid: pick K-1 breakpoints from extrema + derivatives,
    boosted by (not anchored to) neural predictions.
    """
    need = num_phases - 1
    if need <= 0:
        return []

    span = max(t_max - t_min, 1e-6)
    candidates = ranked_breakpoint_candidates(t, y)
    max_struct = candidates[0][0] if candidates else span * 0.01

    pool: List[Tuple[float, float]] = list(candidates)

    scores_arr = (
        list(neural_scores)
        if neural_scores is not None
        else [1.0] * len(neural_bps)
    )
    snap_radius = snap_radius_frac * span
    for bp, ns in zip(neural_bps, scores_arr):
        boost = float(ns) * max_struct * neural_weight
        pool.append((boost, float(bp)))
        # Local snap: reinforce nearest structural candidate
        for j, (c_score, c_tp) in enumerate(pool[:-1]):
            if abs(c_tp - bp) <= snap_radius:
                pool[j] = (c_score + boost, c_tp)

    if not pool:
        return sanitize_breakpoints(
            [t_min + span * (i + 1) / (need + 1) for i in range(need)],
            t_min,
            t_max,
            need,
        )

    score_floor = max_struct * min_score_frac
    pool = [(s, tp) for s, tp in pool if s >= score_floor] or pool
    return _select_spread_breakpoints(pool, need, t_min, t_max)


def snap_breakpoints_hybrid(
    neural_bps: Sequence[float],
    neural_scores: Optional[Sequence[float]],
    t: np.ndarray,
    y: np.ndarray,
    t_min: float,
    t_max: float,
    num_phases: int,
    snap_radius_frac: float = 0.15,
    min_score_frac: float = 0.15,
) -> List[float]:
    """Backward-compatible alias for merge_breakpoints_hybrid."""
    return merge_breakpoints_hybrid(
        neural_bps,
        neural_scores,
        t,
        y,
        t_min,
        t_max,
        num_phases,
        snap_radius_frac=snap_radius_frac,
        min_score_frac=min_score_frac,
    )
