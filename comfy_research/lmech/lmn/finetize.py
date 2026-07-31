"""Finetize soft LMN into hard piecewise symbolic formula."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from comfy_research.lmech.lmn.model import LMN
from comfy_research.lmech.lmn.primitives import PRIMITIVE_NAMES, PrimitiveBank

# Lower = simpler. Used when simplicity knob > 0.
MECHANISM_COMPLEXITY: Dict[str, int] = {
    "const": 0,
    "linear": 1,
    "exp": 2,
    "exp_growth": 2,
    "power": 2,
    "pos_power": 2,
    "sigmoid": 3,
    "gaussian_spike": 3,
}

FinetizeProgressCallback = Callable[[int, int], None]

# Cap breakpoint search for interactive UI (K=4 was ~234 candidates → minutes).
MAX_BREAKPOINT_CANDIDATES = 36
EARLY_STOP_R2 = 0.9999


@dataclass
class SegmentFormula:
    phase_index: int
    t_start: float
    t_end: float
    mechanism: str
    params: Dict[str, float]
    formula: str
    latex: str
    segment_r2: float
    raw_params: Optional[np.ndarray] = None


@dataclass
class PiecewiseFormula:
    breakpoints: List[float]
    segments: List[SegmentFormula]
    global_r2: float
    t_max: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _round_sig(x: float, sig: int = 2) -> float:
    if x == 0:
        return 0.0
    return float(f"{x:.{sig}g}")


def _cleanup_params(
    name: str,
    phys: Dict[str, float],
    t_start: float = 0.0,
    t_end: float = 0.0,
) -> Tuple[str, Dict[str, float]]:
    cleaned = dict(phys)
    t_span = max(t_end - t_start, 1.0)

    if name == "linear":
        slope = cleaned.get("a", 0.0)
        intercept = cleaned.get("c", 0.0)
        if abs(slope) * t_span < max(1e-4, 1e-3 * max(abs(intercept), 1e-3)):
            return "const", {"c": intercept}

    if name == "exp":
        a = cleaned.get("a", 0.0)
        r = cleaned.get("r", 0.0)
        c = cleaned.get("c", 0.0)
        if r < 1e-4:
            return "const", {"c": c}
        amp_lo = a * np.exp(-r * max(t_start, 0.0))
        amp_hi = a * np.exp(-r * max(t_end, 0.0))
        if abs(amp_lo - amp_hi) < max(1e-4, 1e-3 * max(abs(c), 1e-3)):
            return "const", {"c": c}

    if name == "exp_growth":
        a = cleaned.get("a", 0.0)
        r = cleaned.get("r", 0.0)
        c = cleaned.get("c", 0.0)
        if r < 1e-4:
            return "const", {"c": c}
        amp_lo = a * np.exp(r * max(t_start, 0.0))
        amp_hi = a * np.exp(r * max(t_end, 0.0))
        if abs(amp_lo - amp_hi) < max(1e-4, 1e-3 * max(abs(c), 1e-3)):
            return "const", {"c": c}

    if name == "power":
        alpha = cleaned.get("alpha", 0.0)
        c = cleaned.get("c", 0.0)
        if alpha < 1e-4:
            return "const", {"c": c}

    if name == "pos_power":
        beta = cleaned.get("beta", 0.0)
        c = cleaned.get("c", 0.0)
        if beta < 1e-4:
            return "const", {"c": c}

    if name == "sigmoid":
        a = cleaned.get("a", 0.0)
        k = cleaned.get("k", 0.0)
        c0 = cleaned.get("c0", 0.0)
        if abs(a) < 1e-4 or k < 1e-6:
            return "const", {"c": c0 + 0.5 * a}

    if name == "gaussian_spike":
        a = cleaned.get("a", 0.0)
        k = cleaned.get("k", 0.0)
        c0 = cleaned.get("c0", 0.0)
        if abs(a) < 1e-4 or k < 1e-6:
            return "const", {"c": c0}

    return name, cleaned


def _pos_power_beta2_should_compete(
    phys: Dict[str, float], t_start: float, t_end: float, margin_frac: float = 0.05
) -> bool:
    """Drop beta≈2 pos_power only when the vertex t=-τ lies clearly outside the segment."""
    beta = float(phys.get("beta", 0.0))
    if abs(beta - 2.0) > 0.35:
        return True
    tau = float(phys.get("tau", 0.0))
    vertex = -tau
    span = max(t_end - t_start, 1.0)
    margin = margin_frac * span
    if vertex < t_start - margin or vertex > t_end + margin:
        return False
    return True


def _pick_segment_fit(
    candidates: List[Tuple[str, Dict[str, float], np.ndarray, float]],
    simplicity: float,
    t_start: float = 0.0,
    t_end: float = 0.0,
) -> Tuple[str, Dict[str, float], np.ndarray, float]:
    """Pick best primitive; among near-equal MSE prefer simpler mechanisms."""
    finite = [c for c in candidates if np.isfinite(c[3])]
    if not finite:
        return min(candidates, key=lambda c: c[3])
    if t_end > t_start:
        finite = [
            c
            for c in finite
            if c[0] != "pos_power"
            or _pos_power_beta2_should_compete(c[1], t_start, t_end)
        ] or finite
    best_mse = min(c[3] for c in finite)
    close = [c for c in finite if c[3] <= best_mse * 1.05 + 1e-12]
    if not close:
        close = finite
    if simplicity > 0:
        return min(
            close,
            key=lambda c: (
                MECHANISM_COMPLEXITY.get(c[0].split("+")[0], 2),
                c[3],
            ),
        )
    return min(close, key=lambda c: c[3])


def _formula_score(
    formula: PiecewiseFormula,
    t_np: np.ndarray,
    y_np: np.ndarray,
    bank: PrimitiveBank,
    simplicity: float,
) -> float:
    y_pred = _eval_piecewise(bank, formula.segments, t_np)
    mse = float(np.mean((y_np - y_pred) ** 2))
    complexity = sum(
        MECHANISM_COMPLEXITY.get(seg.mechanism.split("+")[0], 2)
        for seg in formula.segments
    )
    return mse * (1.0 + simplicity * complexity / max(len(formula.segments), 1))


def _segment_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def _segment_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mechanism: str,
    t_true: Optional[np.ndarray] = None,
) -> float:
    """
    Segment fit score shown in UI.

    - Non-plateau mechanisms use classic segment R².
    - Plateau (const) uses a more forgiving score that is robust when segment
      variance is tiny (where plain R² can collapse to ~0 despite visually good fit).
    """
    mech = mechanism.split("+")[0]
    if mech != "const":
        return _segment_r2(y_true, y_pred)

    # Plateau-aware scoring: combine residual fit and trend flatness.
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    y_level = float(max(abs(float(np.mean(y_true))), 1e-6))
    y_span = float(max(np.max(y_true) - np.min(y_true), 0.0))

    # Tolerance scale for near-plateau curves (relative to level/span).
    tol = max(0.02 * y_level, 0.15 * y_span, 1e-3)
    fit_score = 1.0 / (1.0 + mse / (tol * tol))

    trend_score = 1.0
    if t_true is not None and len(t_true) >= 3:
        t0 = float(np.min(t_true))
        t_span = float(max(np.max(t_true) - t0, 1e-9))
        t_norm = (t_true - t0) / t_span
        slope = float(np.polyfit(t_norm, y_true, deg=1)[0])  # delta-y over segment
        trend_tol = max(0.05 * y_level, 0.25 * y_span, 1e-3)
        trend_score = 1.0 / (1.0 + (abs(slope) / trend_tol) ** 2)

    score = 0.7 * fit_score + 0.3 * trend_score
    return float(np.clip(score, 0.0, 1.0))


def _sanitize_breakpoints(breakpoints: List[float], t_max: float) -> List[float]:
    bps = sorted(set(float(round(b)) for b in breakpoints if 0 < b < t_max))
    return bps


def _assign_by_breakpoints(t: np.ndarray, breakpoints: List[float]) -> List[np.ndarray]:
    t_max = float(t.max())
    bounds = [0.0] + list(breakpoints) + [t_max]
    segments = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            segments.append(np.array([], dtype=int))
            continue
        if i < len(bounds) - 2:
            mask = (t >= lo) & (t < hi)
        else:
            mask = (t >= lo) & (t <= hi)
        segments.append(np.where(mask)[0])
    return segments


def _fit_segment_forced(
    bank: PrimitiveBank,
    t: np.ndarray,
    y: np.ndarray,
    mechanism: str,
    t_start: float = 0.0,
    t_end: float = 0.0,
) -> Tuple[str, Dict[str, float], np.ndarray, float]:
    """Fit a single primitive chosen by the user (no auto mechanism selection)."""
    name = mechanism.split("+")[0]
    if name not in PRIMITIVE_NAMES:
        raise ValueError(f"Unknown mechanism: {mechanism}")

    raw, _ = bank.fit_single(t, y, name)
    phys = bank.decode_params(name, raw)
    phys = bank.profile_baseline(name, t, y, phys)
    pred = bank.eval_np(name, t, phys)
    mse = float(np.mean((y - pred) ** 2))
    return name, phys, raw, mse


def _fit_segment(
    bank: PrimitiveBank,
    t: np.ndarray,
    y: np.ndarray,
    allow_combo: bool = False,
    simplicity: float = 0.0,
    t_start: float = 0.0,
    t_end: float = 0.0,
) -> Tuple[str, Dict[str, float], np.ndarray, float]:
    """Fit best single primitive; optionally try 2-primitive combo."""
    if t_end <= t_start and len(t) > 0:
        t_start = float(np.min(t))
        t_end = float(np.max(t))

    best_name = "const"
    best_raw = np.zeros(4)
    best_mse = float("inf")
    best_phys: Dict[str, float] = {"c": float(y.mean())}

    candidates: List[Tuple[str, Dict[str, float], np.ndarray, float]] = []
    for name in PRIMITIVE_NAMES:
        raw, mse = bank.fit_single(t, y, name)
        phys = bank.decode_params(name, raw)
        clean_name, clean_phys = _cleanup_params(name, phys, t_start, t_end)
        clean_phys = bank.profile_baseline(clean_name, t, y, clean_phys)
        pred = bank.eval_np(clean_name, t, clean_phys)
        clean_mse = float(np.mean((y - pred) ** 2))
        candidates.append((clean_name, clean_phys, raw, clean_mse))

    best_name, best_phys, best_raw, best_mse = _pick_segment_fit(
        candidates, simplicity, t_start, t_end
    )

    if allow_combo and len(t) >= 10 and best_mse > 1e-6:
        pred = bank.eval_np(best_name, t, best_phys)
        resid = y - pred
        for name2 in PRIMITIVE_NAMES:
            if name2 == best_name:
                continue
            raw2, _ = bank.fit_single(t, resid, name2)
            phys2 = bank.decode_params(name2, raw2)
            combo_pred = pred + bank.eval_np(name2, t, phys2)
            mse_combo = float(np.mean((combo_pred - y) ** 2))
            if mse_combo < best_mse * 0.5:
                best_name = f"{best_name}+{name2}"
                best_mse = mse_combo
                best_phys = {
                    **{f"p1_{k}": v for k, v in best_phys.items()},
                    **{f"p2_{k}": v for k, v in phys2.items()},
                }

    return best_name, best_phys, best_raw, best_mse


def _eval_piecewise(
    bank: PrimitiveBank,
    segments: List[SegmentFormula],
    t: np.ndarray,
) -> np.ndarray:
    y = np.zeros_like(t, dtype=float)
    for seg in segments:
        if seg.phase_index < len(segments) - 1:
            mask = (t >= seg.t_start) & (t < seg.t_end)
        else:
            mask = (t >= seg.t_start) & (t <= seg.t_end)

        if mask.sum() == 0:
            continue

        if "+" in seg.mechanism:
            parts = seg.mechanism.split("+")
            pred = np.zeros(mask.sum())
            t_loc = t[mask] - seg.t_start
            for i, part in enumerate(parts):
                prefix = f"p{i+1}_"
                phys = {
                    k[len(prefix):]: v
                    for k, v in seg.params.items()
                    if k.startswith(prefix)
                }
                if phys:
                    pred += bank.eval_np(part, t_loc, phys)
            y[mask] = pred
        else:
            y[mask] = bank.eval_np(seg.mechanism, t[mask] - seg.t_start, seg.params)
    return y


def _finetize_at_breakpoints(
    bank: PrimitiveBank,
    t_np: np.ndarray,
    y_np: np.ndarray,
    breakpoints: List[float],
    min_samples: int = 2,
    simplicity: float = 0.0,
) -> Optional[PiecewiseFormula]:
    """Build piecewise formula for a fixed breakpoint set."""
    t_max = float(t_np.max())
    breakpoints = _sanitize_breakpoints(breakpoints, t_max)
    segment_indices = _assign_by_breakpoints(t_np, breakpoints)

    if any(len(idx) < min_samples for idx in segment_indices):
        return None

    bounds = [0.0] + breakpoints + [t_max]
    segments: List[SegmentFormula] = []

    for i, idx in enumerate(segment_indices):
        if len(idx) == 0:
            return None
        t_lo, t_hi = bounds[i], bounds[i + 1]
        t_loc = t_np[idx] - t_lo
        name, phys, raw, _ = _fit_segment(
            bank,
            t_loc,
            y_np[idx],
            allow_combo=False,
            simplicity=simplicity,
            t_start=0.0,
            t_end=t_hi - t_lo,
        )
        if "+" not in name:
            formula = bank.to_formula(name, phys)
            latex = bank.to_latex(name, phys)
        else:
            formula = name
            latex = name

        pred_seg = _eval_piecewise(
            bank,
            [
                SegmentFormula(
                    phase_index=i,
                    t_start=t_lo,
                    t_end=t_hi,
                    mechanism=name,
                    params=phys,
                    formula=formula,
                    latex=latex,
                    segment_r2=0.0,
                )
            ],
            t_np[idx],
        )
        segments.append(
            SegmentFormula(
                phase_index=i,
                t_start=t_lo,
                t_end=t_hi,
                mechanism=name,
                params=phys,
                formula=formula,
                latex=latex,
                segment_r2=_segment_score(
                    y_np[idx],
                    pred_seg,
                    mechanism=name,
                    t_true=t_np[idx],
                ),
                raw_params=raw,
            )
        )

    y_pred = _eval_piecewise(bank, segments, t_np)
    global_r2 = _segment_r2(y_np, y_pred)
    return PiecewiseFormula(
        breakpoints=breakpoints,
        segments=segments,
        global_r2=global_r2,
        t_max=t_max,
    )


def finetize_at_breakpoints_forced(
    t: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    breakpoints: List[float],
    segment_mechanisms: List[str],
    min_samples: int = 2,
) -> PiecewiseFormula:
    """Refit each segment with user-selected mechanisms at fixed breakpoints."""
    bank = PrimitiveBank()
    t_np = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)
    t_max = float(t_np.max())
    breakpoints = _sanitize_breakpoints(breakpoints, t_max)
    segment_indices = _assign_by_breakpoints(t_np, breakpoints)

    if len(segment_mechanisms) != len(segment_indices):
        raise ValueError(
            "segment_mechanisms length must match number of segments "
            f"({len(segment_indices)})"
        )
    if any(len(idx) < min_samples for idx in segment_indices):
        raise ValueError("Cannot refit — one or more segments have too few points")

    bounds = [0.0] + breakpoints + [t_max]
    segments: List[SegmentFormula] = []

    for i, idx in enumerate(segment_indices):
        if len(idx) == 0:
            raise ValueError("Cannot refit — empty segment")
        t_lo, t_hi = bounds[i], bounds[i + 1]
        t_loc = t_np[idx] - t_lo
        name, phys, raw, _ = _fit_segment_forced(
            bank,
            t_loc,
            y_np[idx],
            segment_mechanisms[i],
            t_start=0.0,
            t_end=t_hi - t_lo,
        )
        formula = bank.to_formula(name, phys)
        latex = bank.to_latex(name, phys)
        pred_seg = _eval_piecewise(
            bank,
            [
                SegmentFormula(
                    phase_index=i,
                    t_start=t_lo,
                    t_end=t_hi,
                    mechanism=name,
                    params=phys,
                    formula=formula,
                    latex=latex,
                    segment_r2=0.0,
                )
            ],
            t_np[idx],
        )
        segments.append(
            SegmentFormula(
                phase_index=i,
                t_start=t_lo,
                t_end=t_hi,
                mechanism=name,
                params=phys,
                formula=formula,
                latex=latex,
                segment_r2=_segment_score(
                    y_np[idx],
                    pred_seg,
                    mechanism=name,
                    t_true=t_np[idx],
                ),
                raw_params=raw,
            )
        )

    y_pred = _eval_piecewise(bank, segments, t_np)
    return PiecewiseFormula(
        breakpoints=breakpoints,
        segments=segments,
        global_r2=_segment_r2(y_np, y_pred),
        t_max=t_max,
    )


def _is_non_monotonic(t_np: np.ndarray, y_np: np.ndarray) -> bool:
    """
    True when the curve has major trend reversals (down→up→down), not noise ripples.
    """
    n = len(t_np)
    if n < 8:
        return False

    window = max(3, min(n // 10 | 1, 9))
    y_smooth = np.convolve(y_np, np.ones(window) / window, mode="same")
    y_range = float(y_smooth.max() - y_smooth.min())
    if y_range < 1e-9:
        return False

    t_min, t_max = float(t_np[0]), float(t_np[-1])
    margin = 0.08 * (t_max - t_min)

    slopes = np.diff(y_smooth) / np.maximum(np.diff(t_np), 1e-12)
    reversal_ts: List[float] = []
    for i in range(1, len(slopes)):
        if slopes[i - 1] * slopes[i] < 0:
            tp = 0.5 * (float(t_np[i]) + float(t_np[i + 1]))
            if t_min + margin < tp < t_max - margin:
                reversal_ts.append(tp)

    if len(reversal_ts) < 2:
        return False

    reversal_ts.sort()
    for i in range(len(reversal_ts) - 1):
        lo, hi = reversal_ts[i], reversal_ts[i + 1]
        mask = (t_np >= lo) & (t_np <= hi)
        if mask.sum() < 2:
            continue
        swing = float(y_smooth[mask].max() - y_smooth[mask].min())
        if swing < 0.15 * y_range:
            return False
    return True


def _find_ranked_turning_points(
    t_np: np.ndarray,
    y_np: np.ndarray,
    min_prominence_frac: float = 0.03,
) -> List[Tuple[float, float]]:
    """
    Trend-reversal points for non-monotonic curves.

    Combines slope sign changes (dy/dt zero crossings) with local extrema.
    Returns list of (score, t) sorted by t.
    """
    from scipy.signal import find_peaks

    n = len(t_np)
    if n < 5:
        return []

    t_min, t_max = float(t_np[0]), float(t_np[-1])
    t_span = max(t_max - t_min, 1.0)
    margin = 0.05 * t_span

    window = max(3, min(n // 10 | 1, 9))
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_np, kernel, mode="same")

    y_range = float(y_smooth.max() - y_smooth.min())
    if y_range < 1e-9:
        return []

    scored: List[Tuple[float, float]] = []

    slopes = np.diff(y_smooth) / np.maximum(np.diff(t_np), 1e-12)
    for i in range(1, len(slopes)):
        if slopes[i - 1] * slopes[i] < 0:
            tp = 0.5 * (float(t_np[i]) + float(t_np[i + 1]))
            if t_min + margin < tp < t_max - margin:
                score = abs(float(slopes[i - 1] - slopes[i])) * t_span
                scored.append((score, tp))

    prominence = y_range * min_prominence_frac
    distance = max(3, n // 15)
    peak_idx, peak_props = find_peaks(
        y_smooth, prominence=prominence, distance=distance
    )
    valley_idx, valley_props = find_peaks(
        -y_smooth, prominence=prominence, distance=distance
    )
    for idx, prop in zip(peak_idx, peak_props["prominences"]):
        tp = float(t_np[idx])
        if t_min + margin < tp < t_max - margin:
            scored.append((float(prop), tp))
    for idx, prop in zip(valley_idx, valley_props["prominences"]):
        tp = float(t_np[idx])
        if t_min + margin < tp < t_max - margin:
            scored.append((float(prop), tp))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1])
    merge_dist = 0.06 * t_span
    merged: List[Tuple[float, float]] = []
    for score, tp in scored:
        if merged and tp - merged[-1][1] < merge_dist:
            if score > merged[-1][0]:
                merged[-1] = (score, tp)
        else:
            merged.append((score, tp))
    return merged


def find_turning_points(
    t: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    min_prominence_frac: float = 0.03,
) -> List[float]:
    """Significant local extrema as t-coordinates (for non-monotonic curves)."""
    t_np = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)
    return [t_val for _, t_val in _find_ranked_turning_points(t_np, y_np, min_prominence_frac)]


def turning_point_breakpoints(
    t: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    num_phases: int,
    min_prominence_frac: float = 0.03,
) -> Optional[List[float]]:
    """Pick num_phases-1 breakpoints from the most prominent turning points."""
    if num_phases <= 1:
        return []
    t_np = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)
    if not _is_non_monotonic(t_np, y_np):
        return None
    ranked = _find_ranked_turning_points(t_np, y_np, min_prominence_frac)
    need = num_phases - 1
    if len(ranked) < need:
        return None
    top = sorted(ranked, key=lambda x: -x[0])[:need]
    t_max = float(t_np.max())
    return _sanitize_breakpoints([t_val for _, t_val in top], t_max)


def _turning_point_breakpoint_candidates(
    t_np: np.ndarray, y_np: np.ndarray, k: int
) -> List[List[float]]:
    """Breakpoint sets anchored at local extrema (valleys/peaks)."""
    if k <= 1 or not _is_non_monotonic(t_np, y_np):
        return []

    ranked = _find_ranked_turning_points(t_np, y_np)
    need = k - 1
    if len(ranked) < need:
        return []

    t_max = float(t_np.max())
    out: List[List[float]] = []
    seen: set = set()

    def _add(bps: List[float]) -> None:
        clean = _sanitize_breakpoints(bps, t_max)
        if len(clean) == need:
            key = tuple(clean)
            if key not in seen:
                seen.add(key)
                out.append(clean)

    by_prom = sorted(ranked, key=lambda x: -x[0])
    _add([t for _, t in by_prom[:need]])

    by_time = [t for _, t in ranked]
    if len(by_time) >= need:
        _add(by_time[:need])
        _add(by_time[-need:])
        for start in range(0, len(by_time) - need + 1):
            _add(by_time[start : start + need])
            if len(out) >= 8:
                break

    return out


def _changepoint_breakpoint_candidates(
    t_np: np.ndarray, y_np: np.ndarray, k: int, top_n: int = 15
) -> List[List[float]]:
    """Candidate breakpoints from largest slope changes (good for piecewise linear)."""
    if k <= 1 or len(t_np) < 2 * k + 1:
        return []

    dt = np.diff(t_np)
    slopes = np.diff(y_np) / np.maximum(dt, 1e-12)
    if len(slopes) < 2:
        return []

    ds = np.abs(np.diff(slopes))
    interior = np.arange(1, len(ds) - 1)
    if len(interior) == 0:
        return []

    ranked = interior[np.argsort(ds[interior])[::-1]]
    t_max = float(t_np.max())
    bps: List[float] = []
    for idx in ranked:
        bp = 0.5 * (float(t_np[idx]) + float(t_np[idx + 1]))
        if 0 < bp < t_max and bp not in bps:
            bps.append(bp)
        if len(bps) >= top_n:
            break

    if not bps:
        return []

    if k == 2:
        return [[b] for b in bps]

    # k >= 3: avoid combinatorial explosion — sliding windows over top changepoints
    out: List[List[float]] = []
    need = k - 1
    if len(bps) >= need:
        out.append(_sanitize_breakpoints(bps[:need], t_max))
    for start in range(1, min(6, max(1, len(bps) - need + 1))):
        out.append(_sanitize_breakpoints(bps[start : start + need], t_max))
    return out


def _dedupe_breakpoint_candidates(
    candidates: List[List[float]], k: int
) -> List[List[float]]:
    seen = set()
    unique: List[List[float]] = []
    for c in candidates:
        if len(c) != k - 1:
            continue
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:MAX_BREAKPOINT_CANDIDATES]


def _data_breakpoint_candidates(
    t_np: np.ndarray,
    y_np: np.ndarray,
    k: int,
    max_refine: int = 2,
    extra_candidates: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """Breakpoint candidates from data geometry (no trained model)."""
    t_max = float(t_np.max())
    if k <= 1:
        return [[]]

    candidates: List[List[float]] = []
    if extra_candidates:
        for c in extra_candidates:
            candidates.append(_sanitize_breakpoints(c, t_max))

    # Turning points first — critical for non-monotonic (U-shaped) curves.
    candidates.extend(_turning_point_breakpoint_candidates(t_np, y_np, k))

    n = len(t_np)
    idx = np.linspace(0, n - 1, k + 1)[1:-1].astype(int)
    uniform_bps = _sanitize_breakpoints([float(t_np[i]) for i in idx], t_max)
    candidates.append(uniform_bps)

    qs = np.linspace(0, 1, k + 1)[1:-1]
    candidates.append(
        _sanitize_breakpoints([float(np.quantile(t_np, q)) for q in qs], t_max)
    )

    n_t = len(t_np)
    for bi, bp in enumerate(uniform_bps):
        pos = int(np.searchsorted(t_np, bp))
        for offset in range(-max_refine, max_refine + 1):
            if offset == 0:
                continue
            j = pos + offset
            if 0 < j < n_t:
                refined = list(uniform_bps)
                refined[bi] = float(t_np[j])
                candidates.append(_sanitize_breakpoints(refined, t_max))

    candidates.extend(_changepoint_breakpoint_candidates(t_np, y_np, k))
    return _dedupe_breakpoint_candidates(candidates, k)


def _breakpoint_candidates(
    model: LMN,
    t_np: np.ndarray,
    y_np: np.ndarray,
    max_refine: int = 2,
    extra_candidates: Optional[List[List[float]]] = None,
) -> List[List[float]]:
    """Generate candidate breakpoint sets."""
    t_max = float(t_np.max())
    k = model.num_phases
    if k <= 1:
        return [[]]

    candidates = _data_breakpoint_candidates(
        t_np, y_np, k, max_refine=max_refine, extra_candidates=extra_candidates
    )

    # Trained router breakpoints
    with torch.no_grad():
        b_train = model.router.breakpoints().numpy()
    candidates.append(_sanitize_breakpoints([float(b) for b in b_train], t_max))

    t_tensor = torch.tensor(t_np, dtype=torch.float32)
    with torch.no_grad():
        phase_ids = model.router.hard_assign(t_tensor, beta=500.0).numpy()
    changepoints = []
    for i in range(len(phase_ids) - 1):
        if phase_ids[i] != phase_ids[i + 1]:
            changepoints.append(0.5 * (t_np[i] + t_np[i + 1]))
    if len(changepoints) >= k - 1:
        candidates.append(_sanitize_breakpoints(changepoints[: k - 1], t_max))

    return _dedupe_breakpoint_candidates(candidates, k)


def finetize(
    model: LMN,
    t: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    beta_hard: float = 500.0,
    r2_min: float = 0.90,
    min_samples: int = 3,
    simplicity: float = 0.0,
    progress_callback: Optional[FinetizeProgressCallback] = None,
) -> PiecewiseFormula:
    """
    Convert trained soft LMN to hard piecewise symbolic formula.

    Finetize uses per-segment primitive refit (independent of training pi).
    simplicity in [0, 1]: higher values prefer const > linear > exp/power > sigmoid.
    """
    bank = PrimitiveBank()
    t_np = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)

    candidates = _breakpoint_candidates(model, t_np, y_np)
    return _search_finetize_candidates(
        bank,
        t_np,
        y_np,
        candidates,
        min_samples=min_samples,
        simplicity=simplicity,
        progress_callback=progress_callback,
    )


def _search_finetize_candidates(
    bank: PrimitiveBank,
    t_np: np.ndarray,
    y_np: np.ndarray,
    candidates: List[List[float]],
    min_samples: int,
    simplicity: float,
    progress_callback: Optional[FinetizeProgressCallback] = None,
) -> PiecewiseFormula:
    best: Optional[PiecewiseFormula] = None
    best_score = float("inf")
    n_cands = len(candidates)

    for i, bps in enumerate(candidates):
        if progress_callback is not None:
            progress_callback(i + 1, n_cands)
        formula = _finetize_at_breakpoints(
            bank,
            t_np,
            y_np,
            bps,
            min_samples=min_samples,
            simplicity=simplicity,
        )
        if formula is None:
            continue
        score = _formula_score(formula, t_np, y_np, bank, simplicity)
        if score < best_score:
            best_score = score
            best = formula
            if formula.global_r2 >= EARLY_STOP_R2 and score < 1e-10:
                break

    if best is None:
        name, phys, raw, _ = _fit_segment(
            bank,
            t_np,
            y_np,
            simplicity=simplicity,
            t_start=0.0,
            t_end=float(t_np.max()),
        )
        seg = SegmentFormula(
            phase_index=0,
            t_start=0.0,
            t_end=float(t_np.max()),
            mechanism=name,
            params=phys,
            formula=bank.to_formula(name, phys),
            latex=bank.to_latex(name, phys),
            segment_r2=1.0,
            raw_params=raw,
        )
        y_pred = _eval_piecewise(bank, [seg], t_np)
        best = PiecewiseFormula(
            breakpoints=[],
            segments=[seg],
            global_r2=_segment_r2(y_np, y_pred),
            t_max=float(t_np.max()),
        )

    return best


def finetize_points(
    t: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    num_phases: int = 4,
    breakpoints: Optional[List[float]] = None,
    lock_breakpoints: bool = False,
    min_samples: int = 3,
    simplicity: float = 0.0,
    progress_callback: Optional[FinetizeProgressCallback] = None,
    model: Optional[LMN] = None,
) -> PiecewiseFormula:
    """
    Finetize curve data into a piecewise symbolic formula.

    When lock_breakpoints is True, breakpoints must be provided and only
    per-segment primitives are refit (boundary positions stay fixed).
    """
    bank = PrimitiveBank()
    t_np = t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y, dtype=float)
    k = max(1, num_phases)

    if lock_breakpoints:
        if breakpoints is None:
            raise ValueError("lock_breakpoints requires breakpoints")
        if progress_callback is not None:
            progress_callback(1, 1)
        formula = _finetize_at_breakpoints(
            bank,
            t_np,
            y_np,
            breakpoints,
            min_samples=min(min_samples, 2),
            simplicity=simplicity,
        )
        if formula is None:
            raise ValueError(
                "Cannot finetize with locked breakpoints — segments too small"
            )
        return formula

    extra = (
        [breakpoints]
        if breakpoints is not None and len(breakpoints) == k - 1
        else None
    )
    if model is not None:
        candidates = _breakpoint_candidates(
            model, t_np, y_np, extra_candidates=extra
        )
    else:
        candidates = _data_breakpoint_candidates(
            t_np, y_np, k, extra_candidates=extra
        )

    return _search_finetize_candidates(
        bank,
        t_np,
        y_np,
        candidates,
        min_samples=min_samples,
        simplicity=simplicity,
        progress_callback=progress_callback,
    )
