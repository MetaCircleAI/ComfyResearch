"""Mechanistic primitive functions for training dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


PRIMITIVE_NAMES = [
    "exp",
    "const",
    "sigmoid",
    "gaussian_spike",
    "power",
    "pos_power",
    "linear",
    "exp_growth",
]
NUM_PRIMITIVES = len(PRIMITIVE_NAMES)
NAME_TO_ID = {n: i for i, n in enumerate(PRIMITIVE_NAMES)}
_POWER_LIKE = frozenset({"power", "pos_power"})
_EXP_LIKE = frozenset({"exp", "exp_growth"})
# Horizontal shift τ in (t+τ); peak/anchor at t = −τ. const has no shift.
_TAU_SHIFT = frozenset(
    {"exp", "exp_growth", "linear", "power", "pos_power", "sigmoid", "gaussian_spike"}
)

# Max raw (unconstrained) params per primitive slot
MAX_PARAMS = 4
EPS = 1e-6


@dataclass
class PrimitiveSpec:
    name: str
    num_params: int  # number of meaningful params (excluding padding)


PRIMITIVE_SPECS: Dict[str, PrimitiveSpec] = {
    "exp": PrimitiveSpec("exp", 4),       # a, r, tau, c  →  a·exp(-r·(t+τ))+c
    "exp_growth": PrimitiveSpec("exp_growth", 4),  # a, r, tau, c
    "const": PrimitiveSpec("const", 1),   # c
    "sigmoid": PrimitiveSpec("sigmoid", 4),  # c0, a, k, tau
    "gaussian_spike": PrimitiveSpec("gaussian_spike", 4),  # c0, a, k, tau
    "power": PrimitiveSpec("power", 4),   # a, alpha, tau, c
    "pos_power": PrimitiveSpec("pos_power", 4),  # a, beta, tau, c
    "linear": PrimitiveSpec("linear", 3),  # a, tau, c  →  a·(t+τ)+c
}


EPS = 1e-6
# Keep exp primitives finite at large training steps (e.g. t ~ 1e4).
_EXP_CLIP = 20.0


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _exp_clip_torch(expo: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(expo, -_EXP_CLIP, _EXP_CLIP))


def _exp_clip_np(expo: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(expo, -_EXP_CLIP, _EXP_CLIP))


def _gaussian_spike_np(
    t: np.ndarray, c0: float, a: float, k: float, tau: float
) -> np.ndarray:
    z = np.asarray(t, dtype=float) + float(tau)
    expo = np.clip(-float(k) * z * z, -_EXP_CLIP, _EXP_CLIP)
    return float(c0) + float(a) * np.exp(expo)


def _gaussian_spike_torch(
    t: torch.Tensor, c0: torch.Tensor, a: torch.Tensor, k: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    z = t + tau
    expo = torch.clamp(-k * z * z, -_EXP_CLIP, _EXP_CLIP)
    return c0 + a * torch.exp(expo)


def _gaussian_spike_seed(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """Init c0 + a·exp(-k·(t+τ)²) from the dominant peak/trough in y."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 3:
        return None
    c0 = float(np.median(y))
    resid = y - c0
    idx = int(np.argmax(np.abs(resid)))
    a = float(resid[idx])
    if abs(a) < 1e-12:
        return None
    tau = -float(t[idx])
    half = 0.5 * abs(a)
    w = max(float(t.max() - t.min()) * 0.15, 1.0)
    for j in range(len(t)):
        if np.abs(resid[j]) <= half:
            w = min(w, abs(float(t[j]) + tau))
    k = float(np.log(2.0) / max(w * w, EPS))
    return {"c0": c0, "a": a, "k": k, "tau": tau}


def _softplus_decode(raw: float) -> float:
    return float(np.log1p(np.exp(np.clip(float(raw), -20.0, 20.0))) + EPS)


def _softplus_encode(value: float) -> float:
    x = max(float(value), EPS)
    if x > 20.0:
        return x
    return float(np.log(np.expm1(x)))


def _exp_amplitude_decode(raw: float) -> float:
    return float(np.log1p(np.exp(np.clip(float(raw), -20.0, 20.0))))


def _exp_decay_c_bounds(y: np.ndarray) -> Tuple[float, float]:
    """Physical c bounds for a·exp(-r·t)+c: asymptote at/below segment minimum."""
    y = np.asarray(y, dtype=float)
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max - y_min, 1e-6)
    c_hi = y_min + 0.2 * y_range
    c_lo = y_min - 0.5 * y_range
    if c_lo >= c_hi:
        mid = 0.5 * (y_min + y_max)
        pad = max(0.05 * y_range, 1e-4)
        c_lo, c_hi = mid - pad, mid + pad
    return c_lo, c_hi


def _exp_growth_c_bounds(y: np.ndarray) -> Tuple[float, float]:
    """Physical c bounds for a·exp(+r·t)+c: lower asymptote below rising segment."""
    y = np.asarray(y, dtype=float)
    y_min, y_max = float(y.min()), float(y.max())
    y_range = max(y_max - y_min, 1e-6)
    c_lo = y_min - 0.5 * y_range
    c_hi = y_max + 0.5 * y_range
    if c_lo >= c_hi:
        mid = 0.5 * (y_min + y_max)
        pad = max(0.05 * y_range, 1e-4)
        c_lo, c_hi = mid - pad, mid + pad
    return c_lo, c_hi


def _sanitize_lbfgsb_bounds(
    bounds: List[Tuple[Optional[float], Optional[float]]],
    n: int,
) -> List[Tuple[float, float]]:
    """Ensure every L-BFGS-B bound pair has lo < hi (scipy rejects lo >= hi)."""
    out: List[Tuple[float, float]] = []
    for i in range(n):
        lo, hi = bounds[i]
        if lo is None:
            lo = -1e6
        if hi is None:
            hi = 1e6
        lo_f, hi_f = float(lo), float(hi)
        if lo_f >= hi_f:
            mid = 0.5 * (lo_f + hi_f)
            pad = max(abs(mid) * 1e-6, 1e-6)
            lo_f, hi_f = mid - pad, mid + pad
        out.append((lo_f, hi_f))
    return out


def _pos_power_quad_seed(t: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Map quadratic least-squares fit to pos_power with beta=2.

    a·(t+τ)^2 + c expands to a·t^2 + 2aτ·t + (aτ^2+c), so τ shifts the vertex.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 3:
        return None
    p0, p1, p2 = np.polyfit(t, y, 2)
    if not np.isfinite(p0) or abs(p0) < 1e-20:
        return None
    a = float(p0)
    tau = float(p1 / (2.0 * a))
    c = float(p2 - a * tau * tau)
    if not all(np.isfinite(v) for v in (a, tau, c)):
        return None
    return {"a": a, "beta": 2.0, "tau": tau, "c": c}


def _pos_power_allows_signed_shift(beta: float) -> bool:
    """Even integer beta (e.g. 2) allows t+τ<0 so horizontal shift creates a linear term."""
    beta_r = round(float(beta))
    return abs(float(beta) - beta_r) < 1e-3 and beta_r > 0 and beta_r % 2 == 0


def _pos_power_effective_beta(beta: float) -> float:
    if _pos_power_allows_signed_shift(beta):
        return float(round(beta))
    return float(beta)


def _pos_power_base_np(t: np.ndarray, tau: float, beta: float) -> np.ndarray:
    shifted = np.asarray(t, dtype=float) + float(tau)
    if _pos_power_allows_signed_shift(beta):
        return shifted
    return np.maximum(shifted, EPS)


class PrimitiveBank(nn.Module):
    """Library of interpretable 1D mechanistic primitives."""

    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        t: torch.Tensor,
        primitive_id: int | torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate primitive(s).

        Args:
            t: (N,) time steps
            primitive_id: int or (N,) long tensor
            params: (..., MAX_PARAMS) unconstrained raw parameters

        Returns:
            (N,) or broadcast shape values
        """
        if isinstance(primitive_id, int):
            return self._eval_single(t, primitive_id, params)
        # Vectorized over samples with per-sample primitive (used rarely)
        out = torch.zeros_like(t, dtype=params.dtype)
        for pid in range(NUM_PRIMITIVES):
            mask = primitive_id == pid
            if mask.any():
                out[mask] = self._eval_single(t[mask], pid, params[mask])
        return out

    def _eval_single(
        self, t: torch.Tensor, pid: int, params: torch.Tensor
    ) -> torch.Tensor:
        p = params[..., :MAX_PARAMS]
        if pid == NAME_TO_ID["exp"]:
            a = torch.nn.functional.softplus(p[..., 0])
            r = torch.nn.functional.softplus(p[..., 1]) + self.eps
            tau = p[..., 2]
            c = p[..., 3]
            return a * _exp_clip_torch(-r * (t + tau)) + c
        if pid == NAME_TO_ID["exp_growth"]:
            a = p[..., 0]
            r = torch.nn.functional.softplus(p[..., 1]) + self.eps
            tau = p[..., 2]
            c = p[..., 3]
            return a * _exp_clip_torch(r * (t + tau)) + c
        if pid == NAME_TO_ID["const"]:
            return p[..., 0].expand_as(t) if p.ndim > 0 and p.shape[-1] == MAX_PARAMS else p[..., 0]
        if pid == NAME_TO_ID["sigmoid"]:
            c0 = p[..., 0]
            a = p[..., 1]
            k = torch.nn.functional.softplus(p[..., 2]) + self.eps
            tau = p[..., 3]
            return c0 + a * _sigmoid(k * (t + tau))
        if pid == NAME_TO_ID["gaussian_spike"]:
            c0 = p[..., 0]
            a = p[..., 1]
            k = torch.nn.functional.softplus(p[..., 2]) + self.eps
            tau = p[..., 3]
            return _gaussian_spike_torch(t, c0, a, k, tau)
        if pid == NAME_TO_ID["power"]:
            a = torch.exp(p[..., 0])
            alpha = torch.nn.functional.softplus(p[..., 1]) + self.eps
            tau = p[..., 2]
            c = p[..., 3]
            base = torch.clamp(t + tau, min=self.eps)
            return a * base ** (-alpha) + c
        if pid == NAME_TO_ID["pos_power"]:
            a = p[..., 0]
            beta = torch.nn.functional.softplus(p[..., 1]) + self.eps
            tau = p[..., 2]
            c = p[..., 3]
            shifted = t + tau
            beta_int = torch.round(beta)
            even_int = (beta - beta_int).abs() < 1e-3
            even_int = even_int & (beta_int > 0) & (torch.remainder(beta_int, 2) == 0)
            exp = torch.where(even_int, beta_int, beta)
            base = torch.where(even_int, shifted, torch.clamp(shifted, min=self.eps))
            return a * base ** exp + c
        if pid == NAME_TO_ID["linear"]:
            a = p[..., 0]
            tau = p[..., 1]
            c = p[..., 2]
            return a * (t + tau) + c
        raise ValueError(f"Unknown primitive id {pid}")

    def eval_by_name(
        self, t: torch.Tensor, name: str, params: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(t, NAME_TO_ID[name], params)

    def eval_all(
        self, t: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate all primitives for each phase.

        Args:
            t: (N,)
            params: (K, M, MAX_PARAMS)

        Returns:
            (N, K, M) tensor
        """
        n = t.shape[0]
        k, m, _ = params.shape
        out = torch.zeros(n, k, m, device=t.device, dtype=params.dtype)
        for pid in range(NUM_PRIMITIVES):
            out[:, :, pid] = self._eval_single(
                t.unsqueeze(1).expand(n, m),
                pid,
                params[:, pid, :].unsqueeze(0).expand(n, m, MAX_PARAMS),
            )
        return out

    def eval_all_phases(
        self, t: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate all K phases x M primitives.

        Args:
            t: (N,)
            theta: (K, M, MAX_PARAMS)

        Returns:
            (N, K, M)
        """
        n = t.shape[0]
        k, m, _ = theta.shape
        results = []
        for ki in range(k):
            row = []
            for mi in range(m):
                row.append(self.forward(t, mi, theta[ki, mi]))
            results.append(torch.stack(row, dim=-1))
        return torch.stack(results, dim=1)

    # ------------------------------------------------------------------ #
    # NumPy evaluation for finetize (decoded physical params)
    # ------------------------------------------------------------------ #

    @staticmethod
    def decode_params(name: str, raw: np.ndarray) -> Dict[str, float]:
        """Convert unconstrained raw vector to physical parameters."""
        raw = np.asarray(raw, dtype=float).ravel()
        if name == "exp":
            return {
                "a": _exp_amplitude_decode(raw[0]),
                "r": _softplus_decode(raw[1]),
                "tau": float(raw[2]),
                "c": float(raw[3]),
            }
        if name == "exp_growth":
            return {
                "a": float(raw[0]),
                "r": _softplus_decode(raw[1]),
                "tau": float(raw[2]),
                "c": float(raw[3]),
            }
        if name == "const":
            return {"c": float(raw[0])}
        if name == "sigmoid":
            return {
                "c0": float(raw[0]),
                "a": float(raw[1]),
                "k": _softplus_decode(raw[2]),
                "tau": float(raw[3]),
            }
        if name == "gaussian_spike":
            return {
                "c0": float(raw[0]),
                "a": float(raw[1]),
                "k": _softplus_decode(raw[2]),
                "tau": float(raw[3]),
            }
        if name == "power":
            return {
                "a": float(np.exp(np.clip(raw[0], -30, 30))),
                "alpha": _softplus_decode(raw[1]),
                "tau": float(raw[2]),
                "c": float(raw[3]),
            }
        if name == "pos_power":
            return {
                "a": float(raw[0]),
                "beta": _softplus_decode(raw[1]),
                "tau": float(raw[2]),
                "c": float(raw[3]),
            }
        if name == "linear":
            return {"a": float(raw[0]), "tau": float(raw[1]), "c": float(raw[2])}
        raise ValueError(name)

    @staticmethod
    def encode_params(name: str, physical: Dict[str, float]) -> np.ndarray:
        """Encode physical parameters to unconstrained raw vector."""
        raw = np.zeros(MAX_PARAMS, dtype=float)
        if name == "exp":
            a, r, tau, c = (
                physical["a"],
                physical["r"],
                physical.get("tau", 0.0),
                physical["c"],
            )
            raw[0] = _softplus_encode(a) if a > 0.01 else 0.0
            raw[1] = _softplus_encode(r)
            raw[2] = tau
            raw[3] = c
        elif name == "exp_growth":
            a, r, tau, c = (
                physical["a"],
                physical["r"],
                physical.get("tau", 0.0),
                physical["c"],
            )
            raw[0] = a
            raw[1] = _softplus_encode(r)
            raw[2] = tau
            raw[3] = c
        elif name == "const":
            raw[0] = physical["c"]
        elif name == "sigmoid":
            raw[0] = physical["c0"]
            raw[1] = physical["a"]
            raw[2] = _softplus_encode(physical["k"])
            raw[3] = physical["tau"]
        elif name == "gaussian_spike":
            raw[0] = physical["c0"]
            raw[1] = physical["a"]
            raw[2] = _softplus_encode(physical["k"])
            raw[3] = physical["tau"]
        elif name == "power":
            a, alpha, tau, c = (
                physical["a"],
                physical["alpha"],
                physical["tau"],
                physical["c"],
            )
            raw[0] = np.log(max(a, EPS))
            raw[1] = _softplus_encode(alpha)
            raw[2] = tau
            raw[3] = c
        elif name == "pos_power":
            a, beta, tau, c = (
                physical["a"],
                physical["beta"],
                physical["tau"],
                physical["c"],
            )
            raw[0] = a
            raw[1] = _softplus_encode(beta)
            raw[2] = tau
            raw[3] = c
        elif name == "linear":
            raw[0] = physical["a"]
            raw[1] = physical.get("tau", 0.0)
            raw[2] = physical["c"]
        return raw

    @staticmethod
    def safe_flat_raw(name: str, y_value: float) -> np.ndarray:
        """Raw params that evaluate to ~constant y_value (inactive mixture slot)."""
        raw = np.zeros(MAX_PARAMS, dtype=float)
        y = float(y_value)
        if name == "const":
            raw[0] = y
        elif name in _EXP_LIKE:
            raw[0] = 0.0 if name == "exp_growth" else -30.0
            raw[1] = -30.0 if name == "exp_growth" else 30.0
            raw[2] = 0.0
            raw[3] = y
        elif name == "sigmoid":
            raw[0] = y
            raw[1] = 0.0
            raw[2] = -30.0
            raw[3] = 0.0
        elif name == "gaussian_spike":
            raw[0] = y
            raw[1] = 0.0
            raw[2] = -30.0
            raw[3] = 0.0
        elif name in _POWER_LIKE:
            raw[0] = -30.0
            raw[1] = -30.0
            raw[2] = 0.0
            raw[3] = y
        elif name == "linear":
            raw[0] = 0.0
            raw[1] = 0.0
            raw[2] = y
        return raw

    @staticmethod
    def eval_np(name: str, t: np.ndarray, physical: Dict[str, float]) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if name == "exp":
            tau = float(physical.get("tau", 0.0))
            return (
                physical["a"] * _exp_clip_np(-physical["r"] * (t + tau))
                + physical["c"]
            )
        if name == "exp_growth":
            tau = float(physical.get("tau", 0.0))
            return (
                physical["a"] * _exp_clip_np(physical["r"] * (t + tau))
                + physical["c"]
            )
        if name == "const":
            return np.full_like(t, physical["c"], dtype=float)
        if name == "sigmoid":
            return physical["c0"] + physical["a"] * _sigmoid_np(
                physical["k"] * (t + physical["tau"])
            )
        if name == "gaussian_spike":
            return _gaussian_spike_np(
                t,
                physical["c0"],
                physical["a"],
                physical["k"],
                physical["tau"],
            )
        if name == "power":
            base = np.maximum(t + physical["tau"], EPS)
            return physical["a"] * base ** (-physical["alpha"]) + physical["c"]
        if name == "pos_power":
            beta = _pos_power_effective_beta(physical["beta"])
            base = _pos_power_base_np(t, physical["tau"], physical["beta"])
            return physical["a"] * base ** beta + physical["c"]
        if name == "linear":
            tau = float(physical.get("tau", 0.0))
            return physical["a"] * (t + tau) + physical["c"]
        raise ValueError(name)

    @staticmethod
    def _time_scale(t: np.ndarray) -> float:
        """Scale time to O(1) for numerically stable fitting."""
        t = np.asarray(t, dtype=float)
        if len(t) == 0:
            return 1.0
        return max(float(t.max() - t.min()), float(np.max(np.abs(t))), 1.0)

    @staticmethod
    def _y_scale(y: np.ndarray) -> float:
        """Scale response to O(1) for numerically stable fitting."""
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            return 1.0
        return max(float(y.max() - y.min()), float(np.max(np.abs(y))), 1e-6)

    @staticmethod
    def _scale_amplitude(
        name: str, phys: Dict[str, float], y_scale: float, to_normalized: bool
    ) -> Dict[str, float]:
        """Map amplitude / baseline params between physical and normalized y."""
        if abs(y_scale - 1.0) < 1e-12:
            return dict(phys)
        out = dict(phys)
        factor = (1.0 / y_scale) if to_normalized else y_scale
        if name == "const":
            out["c"] *= factor
        elif name in _EXP_LIKE:
            out["a"] *= factor
            out["c"] *= factor
        elif name == "linear":
            out["a"] *= factor
            out["c"] *= factor
        elif name == "power":
            out["a"] *= factor
            out["c"] *= factor
        elif name == "pos_power":
            out["a"] *= factor
            out["c"] *= factor
        elif name == "sigmoid":
            out["c0"] *= factor
            out["a"] *= factor
        elif name == "gaussian_spike":
            out["c0"] *= factor
            out["a"] *= factor
        return out

    @staticmethod
    def _shift_params_to_input_time(
        name: str, phys: Dict[str, float], t_origin: float
    ) -> Dict[str, float]:
        """Convert params fitted on t_local = t - t_origin back to input t coords."""
        if abs(t_origin) < 1e-12:
            return dict(phys)
        out = dict(phys)
        if name in _TAU_SHIFT:
            out["tau"] = float(out["tau"] - t_origin)
        return out

    @staticmethod
    def _shift_params_to_local_time(
        name: str, phys: Dict[str, float], t_origin: float
    ) -> Dict[str, float]:
        """Convert input-time params to local coordinates t_local = t - t_origin."""
        if abs(t_origin) < 1e-12:
            return dict(phys)
        out = dict(phys)
        if name in _TAU_SHIFT:
            out["tau"] = float(out["tau"] + t_origin)
        return out

    @staticmethod
    def _params_from_normalized(
        name: str, phys_norm: Dict[str, float], scale: float
    ) -> Dict[str, float]:
        """Map parameters fitted on t/scale back to original time coordinates."""
        if name == "const" or abs(scale - 1.0) < 1e-12:
            return dict(phys_norm)
        s = scale
        if name in _EXP_LIKE:
            return {
                "a": phys_norm["a"],
                "r": phys_norm["r"] / s,
                "tau": float(phys_norm["tau"] * s),
                "c": phys_norm["c"],
            }
        if name == "linear":
            return {
                "a": phys_norm["a"] / s,
                "tau": float(phys_norm["tau"] * s),
                "c": phys_norm["c"],
            }
        if name == "power":
            alpha = min(float(phys_norm["alpha"]), 10.0)
            a_norm = float(phys_norm["a"])
            if not np.isfinite(a_norm):
                a_norm = 1.0
            log_a = np.log(max(abs(a_norm), EPS)) + alpha * np.log(s)
            a_phys = float(np.exp(np.clip(log_a, -50, 50)))
            if a_norm < 0:
                a_phys = -a_phys
            return {
                "a": a_phys,
                "alpha": alpha,
                "tau": float(phys_norm["tau"] * s),
                "c": float(phys_norm["c"]),
            }
        if name == "pos_power":
            beta = min(float(phys_norm["beta"]), 10.0)
            a_norm = float(phys_norm["a"])
            if not np.isfinite(a_norm):
                a_norm = 1.0
            log_a = np.log(max(abs(a_norm), EPS)) - beta * np.log(s)
            a_phys = float(np.exp(np.clip(log_a, -50, 50)))
            if a_norm < 0:
                a_phys = -a_phys
            return {
                "a": a_phys,
                "beta": beta,
                "tau": float(phys_norm["tau"] * s),
                "c": float(phys_norm["c"]),
            }
        if name == "sigmoid":
            return {
                "c0": phys_norm["c0"],
                "a": phys_norm["a"],
                "k": phys_norm["k"] / s,
                "tau": phys_norm["tau"] * s,
            }
        if name == "gaussian_spike":
            return {
                "c0": phys_norm["c0"],
                "a": phys_norm["a"],
                "k": phys_norm["k"] / (s * s),
                "tau": phys_norm["tau"] * s,
            }
        raise ValueError(name)

    @staticmethod
    def _params_to_normalized(
        name: str, phys: Dict[str, float], scale: float
    ) -> Dict[str, float]:
        """Map original-time parameters to normalized fitting coordinates."""
        if name == "const" or abs(scale - 1.0) < 1e-12:
            return dict(phys)
        s = scale
        if name in _EXP_LIKE:
            return {
                "a": phys["a"],
                "r": phys["r"] * s,
                "tau": phys["tau"] / s,
                "c": phys["c"],
            }
        if name == "linear":
            return {
                "a": phys["a"] * s,
                "tau": phys["tau"] / s,
                "c": phys["c"],
            }
        if name == "power":
            alpha = phys["alpha"]
            log_a = np.log(max(abs(phys["a"]), EPS)) - alpha * np.log(s)
            a_norm = float(np.exp(np.clip(log_a, -50, 50)))
            if phys["a"] < 0:
                a_norm = -a_norm
            return {
                "a": a_norm,
                "alpha": alpha,
                "tau": phys["tau"] / s,
                "c": phys["c"],
            }
        if name == "pos_power":
            beta = phys["beta"]
            log_a = np.log(max(abs(phys["a"]), EPS)) + beta * np.log(s)
            a_norm = float(np.exp(np.clip(log_a, -50, 50)))
            if phys["a"] < 0:
                a_norm = -a_norm
            return {
                "a": a_norm,
                "beta": beta,
                "tau": phys["tau"] / s,
                "c": phys["c"],
            }
        if name == "sigmoid":
            return {
                "c0": phys["c0"],
                "a": phys["a"],
                "k": phys["k"] * s,
                "tau": phys["tau"] / s,
            }
        if name == "gaussian_spike":
            return {
                "c0": phys["c0"],
                "a": phys["a"],
                "k": phys["k"] * s * s,
                "tau": phys["tau"] / s,
            }
        raise ValueError(name)

    @staticmethod
    def profile_baseline(
        name: str, t: np.ndarray, y: np.ndarray, phys: Dict[str, float]
    ) -> Dict[str, float]:
        """Re-estimate additive baseline given fixed shape parameters."""
        out = dict(phys)
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if name == "const":
            out["c"] = float(np.mean(y))
        elif name == "exp":
            tau = float(out.get("tau", 0.0))
            out["c"] = float(
                np.mean(y - out["a"] * _exp_clip_np(-out["r"] * (t + tau)))
            )
            c_lo, c_hi = _exp_decay_c_bounds(y)
            out["c"] = float(np.clip(out["c"], c_lo, c_hi))
        elif name == "exp_growth":
            tau = float(out.get("tau", 0.0))
            out["c"] = float(
                np.mean(y - out["a"] * _exp_clip_np(out["r"] * (t + tau)))
            )
            c_lo, c_hi = _exp_growth_c_bounds(y)
            out["c"] = float(np.clip(out["c"], c_lo, c_hi))
        elif name == "linear":
            tau = float(out.get("tau", 0.0))
            out["c"] = float(np.mean(y - out["a"] * (t + tau)))
        elif name == "power":
            base = np.maximum(t + out["tau"], EPS)
            shape = out["a"] * base ** (-out["alpha"])
            out["c"] = float(np.mean(y - shape))
        elif name == "pos_power":
            beta = _pos_power_effective_beta(out["beta"])
            base = _pos_power_base_np(t, out["tau"], out["beta"])
            shape = out["a"] * base ** beta
            out["c"] = float(np.mean(y - shape))
        elif name == "sigmoid":
            out["c0"] = float(
                np.mean(y - out["a"] * _sigmoid_np(out["k"] * (t + out["tau"])))
            )
        elif name == "gaussian_spike":
            out["c0"] = float(
                np.mean(
                    y
                    - out["a"]
                    * np.exp(
                        np.clip(
                            -out["k"] * (t + out["tau"]) ** 2, -_EXP_CLIP, _EXP_CLIP
                        )
                    )
                )
            )
        return out

    def _canonicalize_power_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> Dict[str, float]:
        """Prefer small local tau; refit amplitude/exponent when tau is ill-conditioned."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        t_span = max(float(t.max() - t.min()), 1.0)
        tau = float(phys.get("tau", 0.0))
        a = float(phys.get("a", 0.0))
        if (
            np.isfinite(a)
            and abs(a) < 1e4
            and tau <= max(t_span * 0.5, 100.0)
        ):
            return dict(phys)

        c_est = float(phys.get("c", y.min()))
        y_shift = np.maximum(y - c_est, EPS)
        best = dict(phys)
        best_mse = float("inf")

        for tau_cand in (
            0.0,
            EPS,
            1e-3,
            0.01,
            0.05,
            0.1,
            0.25 * t_span,
            0.5 * t_span,
        ):
            base = np.maximum(t + tau_cand, EPS)
            log_t = np.log(base)
            log_y = np.log(y_shift)
            if len(t) < 3:
                break
            slope, intercept = np.polyfit(log_t, log_y, 1)
            if slope >= -EPS:
                continue
            a_cand = float(np.exp(np.clip(intercept, -30, 30)))
            alpha_cand = float(-slope)
            cand = {
                "a": a_cand,
                "alpha": alpha_cand,
                "tau": tau_cand,
                "c": c_est,
            }
            cand = self.profile_baseline("power", t, y, cand)
            pred = self.eval_np("power", t, cand)
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((pred - y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = cand

        return best

    def _canonicalize_pos_power_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> Dict[str, float]:
        """Refit pos_power; use beta=2 + horizontal tau shift for parabolic segments."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        t_min = float(t.min())
        t_max = float(t.max())
        t_span = max(t_max - t_min, 1.0)
        best = dict(phys)
        best_mse = float("inf")
        pred = self.eval_np("pos_power", t, phys)
        if np.all(np.isfinite(pred)):
            best_mse = float(np.mean((pred - y) ** 2))

        def _try(cand: Dict[str, float]) -> None:
            nonlocal best, best_mse
            cand = self.profile_baseline("pos_power", t, y, cand)
            pred = self.eval_np("pos_power", t, cand)
            if not np.all(np.isfinite(pred)):
                return
            mse = float(np.mean((pred - y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = cand

        quad = _pos_power_quad_seed(t, y)
        if quad is not None:
            _try(quad)

        c_est = float(phys.get("c", y[0]))
        y_shift = np.maximum(c_est - y, EPS)
        tau_fracs = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
        tau_candidates = sorted(
            {frac * t_span for frac in tau_fracs}
            | {0.0, EPS, 1e-3, 0.01, 0.05, 0.1, t_min, t_max - t_span}
        )

        for tau_cand in tau_candidates:
            shifted = t + tau_cand
            x2 = shifted ** 2
            if len(t) >= 3:
                slope, intercept = np.polyfit(x2, y, 1)
                _try(
                    {
                        "a": float(slope),
                        "beta": 2.0,
                        "tau": float(tau_cand),
                        "c": float(intercept),
                    }
                )

            base = np.maximum(shifted, EPS)
            log_t = np.log(base)
            log_y = np.log(y_shift)
            if len(t) >= 3:
                slope, intercept = np.polyfit(log_t, log_y, 1)
                if slope > EPS:
                    _try(
                        {
                            "a": -float(np.exp(np.clip(intercept, -30, 30))),
                            "beta": max(float(slope), EPS),
                            "tau": float(tau_cand),
                            "c": c_est,
                        }
                    )

        return best

    @staticmethod
    def _exp_params_well_conditioned(
        name: str,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> bool:
        """True when exp-like params are finite and baseline/amplitude stay near the data."""
        if name not in _EXP_LIKE:
            return False
        if not all(np.isfinite(v) for v in phys.values()):
            return False
        a = float(phys.get("a", 0.0))
        r = float(phys.get("r", 0.0))
        c = float(phys.get("c", 0.0))
        if r <= 0.0:
            return False
        if name == "exp" and a <= 0.0:
            return False
        if name == "exp_growth" and abs(a) < EPS:
            return False
        y_min, y_max = float(y.min()), float(y.max())
        y_range = max(y_max - y_min, 1e-6)
        if c < max(0.0, y_min - 0.5 * y_range) or c > y_max + 0.5 * y_range:
            return False
        if name == "exp":
            c_lo, c_hi = _exp_decay_c_bounds(y)
            if c < c_lo - 1e-9 or c > c_hi + 1e-9:
                return False
        if abs(a) > 20.0 * y_range:
            return False
        pred = PrimitiveBank.eval_np(name, t, phys)
        return bool(np.all(np.isfinite(pred)))

    def _canonicalize_exp_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
        *,
        normalized: bool = False,
    ) -> Dict[str, float]:
        """Re-estimate (a, r, c) via log-linear fits when optimization drifts ill-conditioned."""
        return self._canonicalize_exp_like_params(
            "exp", t, y, phys, decay=True, normalized=normalized
        )

    def _canonicalize_exp_growth_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
        *,
        normalized: bool = False,
    ) -> Dict[str, float]:
        """Re-estimate growth exp params via log-linear fits."""
        return self._canonicalize_exp_like_params(
            "exp_growth", t, y, phys, decay=False, normalized=normalized
        )

    def _canonicalize_exp_like_params(
        self,
        name: str,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
        decay: bool,
        *,
        normalized: bool = False,
    ) -> Dict[str, float]:
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        y_min, y_max = float(y.min()), float(y.max())
        y_range = max(y_max - y_min, 1e-6)
        c_candidates = set(np.linspace(0.0, float(y[0]), 9))
        c_candidates.update(
            {
                y_min,
                float(y[-1]),
                float(np.percentile(y, 5)),
                float(np.percentile(y, 10)),
            }
        )
        if not decay:
            c_candidates.update(
                np.linspace(float(y_max), float(y_max) + 0.2 * y_range, 5)
            )
            c_candidates.add(max(float(y[0]) - 0.15 * y_range, 0.0))
        c0 = float(phys.get("c", y_min))
        if y_min - 0.5 * y_range <= c0 <= y_max + 0.5 * y_range:
            c_candidates.add(c0)

        best = dict(phys)
        if "tau" not in best:
            best["tau"] = 0.0
        best_mse = float("inf")
        if self._exp_params_well_conditioned(name, t, y, phys):
            pred = self.eval_np(name, t, phys)
            if np.all(np.isfinite(pred)):
                best_mse = float(np.mean((pred - y) ** 2))

        t_min = float(t.min())
        t_max = float(t.max())
        t_span = max(t_max - t_min, 1.0)
        tau_candidates = {
            0.0,
            float(phys.get("tau", 0.0)),
            -t_min,
            -0.5 * t_span,
            0.5 * t_span,
        }

        for c_est in c_candidates:
            for tau_cand in tau_candidates:
                if decay:
                    y_shift = np.maximum(y - c_est, EPS)
                    pos = y_shift > max(y_range * 0.01, 1e-3)
                    if pos.sum() < 3:
                        continue
                    t_eff = t[pos] + tau_cand
                    slope, intercept = np.polyfit(t_eff, np.log(y_shift[pos]), 1)
                    if slope >= -EPS:
                        continue
                    r_cand = max(float(-slope), EPS)
                    a_cand = max(
                        float(np.exp(np.clip(intercept, -30.0, 30.0))), EPS
                    )
                else:
                    decreasing = float(y[-1]) < float(y[0]) - 0.01 * y_range
                    if decreasing:
                        y_shift = np.maximum(c_est - y, EPS)
                        pos = y_shift > max(y_range * 0.01, 1e-3)
                        if pos.sum() < 3:
                            continue
                        t_eff = t[pos] + tau_cand
                        slope, intercept = np.polyfit(t_eff, np.log(y_shift[pos]), 1)
                        if slope <= EPS:
                            continue
                        r_cand = max(float(slope), EPS)
                        a_cand = -max(
                            float(np.exp(np.clip(intercept, -30.0, 30.0))), EPS
                        )
                    else:
                        y_shift = np.maximum(y - c_est, EPS)
                        pos = y_shift > max(y_range * 0.01, 1e-3)
                        if pos.sum() < 3:
                            continue
                        t_eff = t[pos] + tau_cand
                        slope, intercept = np.polyfit(t_eff, np.log(y_shift[pos]), 1)
                        if slope <= EPS:
                            continue
                        r_cand = max(float(slope), EPS)
                        a_cand = max(
                            float(np.exp(np.clip(intercept, -30.0, 30.0))), EPS
                        )
                cand = {
                    "a": a_cand,
                    "r": r_cand,
                    "tau": float(tau_cand),
                    "c": c_est,
                }
                cand = self.profile_baseline(name, t, y, cand)
                if not self._exp_params_well_conditioned(name, t, y, cand):
                    continue
                pred = self.eval_np(name, t, cand)
                mse = float(np.mean((pred - y) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best = cand

        if np.isfinite(best_mse):
            return best
        return dict(phys)

    def _refine_exp_like_phys(
        self,
        name: str,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> Dict[str, float]:
        """Polish (a, r, tau, c) in physical coordinates to minimize segment MSE."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        y_min, y_max = float(y.min()), float(y.max())
        y_range = max(y_max - y_min, 1e-6)
        t_span = max(float(t.max() - t.min()), 1.0)

        if name == "exp":
            c_lo, c_hi = _exp_decay_c_bounds(y)
        elif name == "exp_growth":
            c_lo, c_hi = _exp_growth_c_bounds(y)
        else:
            return dict(phys)

        a0 = float(phys.get("a", EPS))
        if name == "exp" and a0 <= 0.0:
            a0 = EPS
        r0 = max(float(phys.get("r", EPS)), EPS)
        tau0 = float(phys.get("tau", 0.0))
        c0 = float(np.clip(float(phys.get("c", 0.0)), c_lo, c_hi))

        def mse_vec(x: np.ndarray) -> float:
            a, r, tau, c = float(x[0]), float(x[1]), float(x[2]), float(x[3])
            if r <= 0.0 or c < c_lo or c > c_hi:
                return 1e9
            if name == "exp" and a <= 0.0:
                return 1e9
            if name == "exp_growth" and abs(a) < EPS:
                return 1e9
            if abs(a) > 25.0 * y_range:
                return 1e9
            pred = self.eval_np(
                name, t, {"a": a, "r": r, "tau": tau, "c": c}
            )
            if not np.all(np.isfinite(pred)):
                return 1e9
            return float(np.mean((pred - y) ** 2))

        x0 = np.array([a0, r0, tau0, c0], dtype=float)
        best_mse = mse_vec(x0)
        best_x = x0.copy()
        r_hi = max(r0 * 20.0, 10.0 / t_span, 1e-3)
        tau_lo = min(-t_span - 0.5, -1.5)
        tau_hi = max(t_span + 0.5, 1.5)
        if name == "exp":
            a_bounds = (EPS, 25.0 * y_range)
        else:
            a_bounds = (-25.0 * y_range, 25.0 * y_range)
        bounds = [
            a_bounds,
            (EPS / max(t_span, 1.0), r_hi),
            (tau_lo, tau_hi),
            (c_lo, c_hi),
        ]
        bounds = _sanitize_lbfgsb_bounds(bounds, 4)

        res = minimize(mse_vec, x0, method="L-BFGS-B", bounds=bounds)
        if np.isfinite(res.fun) and res.fun < best_mse:
            best_x = res.x
            best_mse = res.fun

        decreasing = float(y[-1]) < float(y[0]) - 0.01 * y_range
        for c_seed in np.linspace(c_lo, c_hi, 9):
            for tau_seed in (0.0, tau0, -0.5 * t_span):
                seeds: List[Tuple[float, float]] = []
                if name == "exp":
                    y_shift = np.maximum(y - c_seed, EPS)
                    pos = y_shift > max(y_range * 0.01, 1e-3)
                    if pos.sum() >= 3:
                        t_eff = t[pos] + tau_seed
                        slope, intercept = np.polyfit(
                            t_eff, np.log(y_shift[pos]), 1
                        )
                        if slope < -EPS:
                            seeds.append(
                                (
                                    max(
                                        float(
                                            np.exp(np.clip(intercept, -30.0, 30.0))
                                        ),
                                        EPS,
                                    ),
                                    max(float(-slope), EPS),
                                )
                            )
                elif not decreasing:
                    y_shift = np.maximum(y - c_seed, EPS)
                    pos = y_shift > max(y_range * 0.01, 1e-3)
                    if pos.sum() >= 3:
                        t_eff = t[pos] + tau_seed
                        slope, intercept = np.polyfit(
                            t_eff, np.log(y_shift[pos]), 1
                        )
                        if slope > EPS:
                            seeds.append(
                                (
                                    max(
                                        float(
                                            np.exp(np.clip(intercept, -30.0, 30.0))
                                        ),
                                        EPS,
                                    ),
                                    max(float(slope), EPS),
                                )
                            )
                if name == "exp_growth" and decreasing:
                    y_shift = np.maximum(c_seed - y, EPS)
                    pos = y_shift > max(y_range * 0.01, 1e-3)
                    if pos.sum() >= 3:
                        t_eff = t[pos] + tau_seed
                        slope, intercept = np.polyfit(
                            t_eff, np.log(y_shift[pos]), 1
                        )
                        if slope > EPS:
                            seeds.append(
                                (
                                    -max(
                                        float(
                                            np.exp(np.clip(intercept, -30.0, 30.0))
                                        ),
                                        EPS,
                                    ),
                                    max(float(slope), EPS),
                                )
                            )
                for a_seed, r_seed in seeds:
                    x_seed = np.array(
                        [a_seed, r_seed, float(tau_seed), float(c_seed)]
                    )
                    mse_seed = mse_vec(x_seed)
                    if mse_seed < best_mse:
                        res_seed = minimize(
                            mse_vec, x_seed, method="L-BFGS-B", bounds=bounds
                        )
                        if np.isfinite(res_seed.fun) and res_seed.fun < best_mse:
                            best_x = res_seed.x
                            best_mse = res_seed.fun

        return {
            "a": float(best_x[0]),
            "r": float(best_x[1]),
            "tau": float(best_x[2]),
            "c": float(best_x[3]),
        }

    def _canonicalize_sigmoid_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> Dict[str, float]:
        """Re-estimate tau when it drifts outside the segment (avoids saturated fits)."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        t_min = float(t.min())
        t_max = float(t.max())
        t_span = max(t_max - t_min, 1.0)
        tau = float(phys.get("tau", 0.0))
        a = float(phys.get("a", 0.0))
        k = float(phys.get("k", EPS))

        if (
            np.isfinite(a)
            and abs(a) > EPS
            and np.isfinite(k)
            and k > EPS
            and -t_span <= tau <= t_max + t_span
        ):
            return dict(phys)

        decreasing = float(y[-1]) < float(y[0]) - 0.01 * max(float(y.max() - y.min()), EPS)
        if decreasing:
            c0 = float(y.max())
            amp = float(y.min() - y.max())
        else:
            c0 = float(y.min())
            amp = max(float(y.max() - y.min()), EPS)

        if abs(amp) < EPS:
            return dict(phys)

        y_norm = np.clip((y - c0) / amp, 0.02, 0.98)
        logit = np.log(y_norm / (1.0 - y_norm))

        tau_candidates = {
            0.0,
            -t_min,
            t_min - t_max,
            -0.5 * t_span,
            0.5 * t_span,
            t_max - 0.5 * t_span,
        }
        if len(t) >= 3:
            slope, intercept = np.polyfit(t, logit, 1)
            if abs(slope) > 1e-8:
                tau_candidates.add(float(intercept / slope))
                for offset in (-0.25 * t_span, 0.0, 0.25 * t_span):
                    tau_candidates.add(float(intercept / slope + offset))

        best = dict(phys)
        best_mse = float("inf")
        for tau_cand in tau_candidates:
            if len(t) < 3:
                break
            slope, intercept = np.polyfit(t, logit, 1)
            if abs(slope) <= 1e-8:
                break
            k_est = abs(float(slope))
            pred_logit = k_est * (t + tau_cand)
            logit_err = float(np.mean((pred_logit - logit) ** 2))
            cand = {
                "c0": c0,
                "a": amp,
                "k": k_est,
                "tau": tau_cand,
            }
            cand = self.profile_baseline("sigmoid", t, y, cand)
            pred = self.eval_np("sigmoid", t, cand)
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((pred - y) ** 2))
            if mse < best_mse or (mse <= best_mse * 1.01 and logit_err < 1e-6):
                best_mse = mse
                best = cand

        return best

    def _canonicalize_gaussian_spike_params(
        self,
        t: np.ndarray,
        y: np.ndarray,
        phys: Dict[str, float],
    ) -> Dict[str, float]:
        """Re-estimate tau/k when the spike drifts outside the segment."""
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        t_min = float(t.min())
        t_max = float(t.max())
        t_span = max(t_max - t_min, 1.0)
        tau = float(phys.get("tau", 0.0))
        a = float(phys.get("a", 0.0))
        k = float(phys.get("k", EPS))

        if (
            np.isfinite(a)
            and abs(a) > EPS
            and np.isfinite(k)
            and k > EPS
            and -t_span <= tau <= t_max + t_span
        ):
            return dict(phys)

        seed = _gaussian_spike_seed(t, y)
        if seed is None:
            return dict(phys)

        tau_candidates = {
            seed["tau"],
            0.0,
            -t_min,
            t_min - t_max,
            -0.5 * t_span,
            0.5 * t_span,
        }
        best = dict(phys)
        best_mse = float("inf")
        for tau_cand in tau_candidates:
            cand = dict(seed)
            cand["tau"] = float(tau_cand)
            cand = self.profile_baseline("gaussian_spike", t, y, cand)
            pred = self.eval_np("gaussian_spike", t, cand)
            if not np.all(np.isfinite(pred)):
                continue
            mse = float(np.mean((pred - y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best = cand
        return best

    def _init_raw_normalized(
        self,
        name: str,
        t_local: np.ndarray,
        y: np.ndarray,
        t_scale: float,
        y_scale: float,
    ) -> np.ndarray:
        """Default physical init → y-scaled → t-normalized raw vector for t_fit/y_fit."""
        init_raw = self._default_init(name, t_local, y)
        init_phys = self.decode_params(name, init_raw)
        init_phys = self._scale_amplitude(name, init_phys, y_scale, True)
        init_phys_norm = self._params_to_normalized(name, init_phys, t_scale)
        return self.encode_params(name, init_phys_norm)

    def fit_single(
        self,
        t: np.ndarray,
        y: np.ndarray,
        primitive_id: int | str,
        init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Fit one primitive to segment data via L-BFGS-B on raw params.

        All primitives share one pipeline:
          t_fit = (t - t_min) / t_scale,  y_fit = y / y_scale
          init from (t_local, y) → y-scale → t-normalize → optimize on (t_fit, y_fit)
          denormalize back to input-time physical params for output.

        Returns:
            raw_params (MAX_PARAMS,), mse
        """
        if isinstance(primitive_id, str):
            primitive_id = NAME_TO_ID[primitive_id]
        name = PRIMITIVE_NAMES[primitive_id]
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        n_params = PRIMITIVE_SPECS[name].num_params

        t_origin = float(np.min(t)) if len(t) else 0.0
        t_local = t - t_origin
        t_scale = self._time_scale(t_local)
        t_fit = t_local / t_scale
        y_scale = self._y_scale(y)
        y_fit = y / y_scale

        if init is None:
            init = self._init_raw_normalized(name, t_local, y, t_scale, y_scale)
        else:
            phys = self.decode_params(name, init)
            phys = self._shift_params_to_local_time(name, phys, t_origin)
            phys = self._scale_amplitude(name, phys, y_scale, True)
            phys_norm = self._params_to_normalized(name, phys, t_scale)
            init = self.encode_params(name, phys_norm)

        bounds = self._fit_bounds(name, t_fit, y_fit)
        bounds = _sanitize_lbfgsb_bounds(bounds, n_params)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        def loss(raw: np.ndarray) -> float:
            raw_full = np.zeros(MAX_PARAMS)
            raw_full[:n_params] = raw[:n_params]
            phys_norm = self.decode_params(name, raw_full)
            pred = self.eval_np(name, t_fit, phys_norm)
            if not np.all(np.isfinite(pred)):
                return 1e9
            return float(np.mean((pred - y_fit) ** 2))

        def _denorm_phys(phys_norm: Dict[str, float]) -> Dict[str, float]:
            phys = self._params_from_normalized(name, phys_norm, t_scale)
            phys = self._scale_amplitude(name, phys, y_scale, False)
            if name == "power":
                phys = self._canonicalize_power_params(t_local, y, phys)
            elif name == "pos_power":
                phys = self._canonicalize_pos_power_params(t_local, y, phys)
            elif name == "sigmoid":
                phys = self._canonicalize_sigmoid_params(t_local, y, phys)
            elif name == "gaussian_spike":
                phys = self._canonicalize_gaussian_spike_params(t_local, y, phys)
            phys = self._shift_params_to_input_time(name, phys, t_origin)
            return phys

        def _candidate_mse(raw_vec: np.ndarray) -> float:
            raw_full = np.zeros(MAX_PARAMS)
            raw_full[:n_params] = raw_vec[:n_params]
            phys_norm = self.decode_params(name, raw_full)
            if name == "power":
                phys_norm["alpha"] = min(float(phys_norm["alpha"]), 10.0)
            elif name == "pos_power":
                phys_norm["beta"] = min(float(phys_norm["beta"]), 10.0)
            phys = _denorm_phys(phys_norm)
            if not all(np.isfinite(v) for v in phys.values()):
                return float("inf")
            pred = self.eval_np(name, t, phys)
            if not np.all(np.isfinite(pred)):
                return float("inf")
            return float(np.mean((pred - y) ** 2))

        x0 = init[:n_params].copy()
        best_x, best_loss = x0, _candidate_mse(x0)

        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds[:n_params])
        if res.fun < loss(x0):
            cand_loss = _candidate_mse(res.x)
            if cand_loss < best_loss:
                best_x, best_loss = res.x, cand_loss

        # Nelder-Mead fallback for non-convex fits (L-BFGS-B can miss minima).
        if name in ("sigmoid", "gaussian_spike", "power", "pos_power", "exp", "exp_growth") and best_loss > 1e-8:
            seen: set = set()
            for start in (x0, best_x):
                key = tuple(np.round(start, 6))
                if key in seen:
                    continue
                seen.add(key)
                nm = minimize(
                    loss,
                    start,
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-10},
                )
                if nm.fun < loss(start):
                    nm_x = np.clip(nm.x, lo, hi)
                    cand_loss = _candidate_mse(nm_x)
                    if cand_loss < best_loss:
                        best_x, best_loss = nm_x, cand_loss

        raw_norm = np.zeros(MAX_PARAMS)
        raw_norm[:n_params] = best_x
        phys_norm = self.decode_params(name, raw_norm)
        if name == "power":
            phys_norm["alpha"] = min(float(phys_norm["alpha"]), 10.0)
        elif name == "pos_power":
            phys_norm["beta"] = min(float(phys_norm["beta"]), 10.0)
        phys = _denorm_phys(phys_norm)
        if name in _EXP_LIKE:
            phys = self._refine_exp_like_phys(name, t, y, phys)
        else:
            phys = self.profile_baseline(name, t, y, phys)
        pred = self.eval_np(name, t, phys)
        if not np.all(np.isfinite(pred)):
            phys_norm_fb = self.decode_params(
                name, self._init_raw_normalized(name, t_local, y, t_scale, y_scale)
            )
            phys = _denorm_phys(phys_norm_fb)
            if name in _EXP_LIKE:
                phys = self._refine_exp_like_phys(name, t, y, phys)
            else:
                phys = self.profile_baseline(name, t, y, phys)
            pred = self.eval_np(name, t, phys)
        best_loss = float(np.mean((pred - y) ** 2))
        raw_full = self.encode_params(name, phys)
        return raw_full, best_loss

    def _default_init(self, name: str, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        raw = np.zeros(MAX_PARAMS)
        y_mean, y_min, y_max = float(y.mean()), float(y.min()), float(y.max())
        t_mid = float(np.median(t))
        if name == "exp":
            # Log-linear init: ln(y - c) ≈ ln(a) - r·t (avoids softplus r≈0.69 trap)
            c_est = y_min
            y_shift = np.maximum(y - c_est, EPS)
            y_range = max(y_max - y_min, EPS)
            pos = y_shift > max(y_range * 0.01, 1e-3)
            if pos.sum() >= 3:
                slope, intercept = np.polyfit(t[pos], np.log(y_shift[pos]), 1)
                if slope < -EPS:
                    return self.encode_params(
                        "exp",
                        {
                            "a": max(float(np.exp(intercept)), EPS),
                            "r": max(float(-slope), EPS),
                            "tau": 0.0,
                            "c": c_est,
                        },
                    )
            raw[0] = np.log(max(y_max - y_min, 0.1))
            raw[1] = 0.0
            raw[2] = 0.0
            raw[3] = y_min
        elif name == "exp_growth":
            y_range = max(y_max - y_min, EPS)
            decreasing = float(y[-1]) < float(y[0]) - 0.01 * y_range
            if decreasing:
                c_est = float(y_max) + 0.05 * y_range
                y_shift = np.maximum(c_est - y, EPS)
            else:
                c_est = max(float(y[0]) - 0.15 * y_range, 0.0)
                y_shift = np.maximum(y - c_est, EPS)
            pos = y_shift > max(y_range * 0.01, 1e-3)
            if pos.sum() >= 3:
                slope, intercept = np.polyfit(t[pos], np.log(y_shift[pos]), 1)
                if decreasing and slope > EPS:
                    return self.encode_params(
                        "exp_growth",
                        {
                            "a": -max(float(np.exp(intercept)), EPS),
                            "r": max(float(slope), EPS),
                            "tau": 0.0,
                            "c": c_est,
                        },
                    )
                if not decreasing and slope > EPS:
                    return self.encode_params(
                        "exp_growth",
                        {
                            "a": max(float(np.exp(intercept)), EPS),
                            "r": max(float(slope), EPS),
                            "tau": 0.0,
                            "c": c_est,
                        },
                    )
            raw[0] = -max(y_range * 0.5, 0.1) if decreasing else max(y_range, 0.1)
            raw[1] = 0.0
            raw[2] = 0.0
            raw[3] = c_est
        elif name == "const":
            raw[0] = y_mean
        elif name == "sigmoid":
            decreasing = float(y[-1]) < float(y[0]) - 0.01 * max(y_max - y_min, EPS)
            if decreasing:
                c0 = y_max
                amp = y_min - y_max
            else:
                c0 = y_min
                amp = max(y_max - y_min, EPS)
            y_norm = np.clip((y - c0) / amp, 0.02, 0.98)
            logit = np.log(y_norm / (1.0 - y_norm))
            t_min = float(t.min())
            t_max = float(t.max())
            t_span = max(t_max - t_min, 1.0)

            tau_candidates = {
                0.0,
                -t_min,
                t_min - t_max,
                -0.5 * t_span,
                0.5 * t_span,
                t_max - 0.5 * t_span,
            }
            best_init = None
            best_err = float("inf")
            if len(t) >= 3:
                slope, intercept = np.polyfit(t, logit, 1)
                if abs(slope) > 1e-8:
                    tau_candidates.add(float(intercept / slope))
                    for offset in (-0.25 * t_span, 0.0, 0.25 * t_span):
                        tau_candidates.add(float(intercept / slope + offset))

            for tau_cand in tau_candidates:
                if len(t) < 3:
                    break
                slope, intercept = np.polyfit(t, logit, 1)
                if abs(slope) <= 1e-8:
                    break
                k_est = abs(float(slope))
                pred_logit = k_est * (t + tau_cand)
                err = float(np.mean((pred_logit - logit) ** 2))
                if err < best_err:
                    best_err = err
                    best_init = self.encode_params(
                        "sigmoid",
                        {
                            "c0": c0,
                            "a": amp,
                            "k": k_est,
                            "tau": tau_cand,
                        },
                    )
            if best_init is not None:
                return best_init
            raw[0] = y_max if decreasing else y_min
            raw[1] = amp
            raw[2] = 0.0
            raw[3] = -t_mid
        elif name == "gaussian_spike":
            seed = _gaussian_spike_seed(t, y)
            if seed is not None:
                return self.encode_params("gaussian_spike", seed)
            raw[0] = y_mean
            raw[1] = max(y_max - y_min, EPS)
            raw[2] = 0.0
            raw[3] = -t_mid
        elif name == "power":
            c_est = y_min
            y_shift = np.maximum(y - c_est, EPS)
            t_min = float(t.min())
            t_max = float(t.max())
            t_span = max(t_max - t_min, 1.0)
            tau_lo = max(EPS, 1e-3)

            tau_candidates = sorted(
                {
                    0.0,
                    1.0,
                    10.0,
                    t_span * 0.01,
                    t_span * 0.05,
                    t_span * 0.1,
                    t_span * 0.25,
                    t_span * 0.5,
                }
            )

            best_init = None
            best_err = float("inf")
            best_tau = None
            for tau_cand in tau_candidates:
                if tau_cand < tau_lo:
                    continue
                base = np.maximum(t + tau_cand, EPS)
                log_t = np.log(base)
                log_y = np.log(y_shift)
                if len(t) >= 3:
                    slope, intercept = np.polyfit(log_t, log_y, 1)
                    if slope < -EPS:
                        pred_log = intercept + slope * log_t
                        err = float(np.mean((pred_log - log_y) ** 2))
                        if err < best_err * 1.01 and (
                            err < best_err - 1e-12
                            or best_tau is None
                            or tau_cand < best_tau
                        ):
                            best_err = err
                            best_tau = tau_cand
                            best_init = self.encode_params(
                                "power",
                                {
                                    "a": min(
                                        max(
                                            float(
                                                np.exp(np.clip(intercept, -30, 30))
                                            ),
                                            EPS,
                                        ),
                                        1e3,
                                    ),
                                    "alpha": max(float(-slope), EPS),
                                    "tau": tau_cand,
                                    "c": c_est,
                                },
                            )
            if best_init is not None:
                return best_init
            raw[0] = np.log(max(y_max, 0.1))
            raw[1] = 0.0
            raw[2] = 1.0
            raw[3] = y_min
        elif name == "pos_power":
            quad = _pos_power_quad_seed(t, y)
            if quad is not None:
                pred = self.eval_np("pos_power", t, quad)
                if np.all(np.isfinite(pred)):
                    return self.encode_params("pos_power", quad)

            c_est = float(y[0])
            y_shift = np.maximum(c_est - y, EPS)
            t_min = float(t.min())
            t_max = float(t.max())
            t_span = max(t_max - t_min, 1.0)

            tau_candidates = sorted(
                {frac * t_span for frac in (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)}
                | {0.0, 1.0, 10.0, t_span * 0.01, t_span * 0.05, t_span * 0.1, t_span * 0.25, t_span * 0.5}
            )

            best_init = None
            best_err = float("inf")
            for tau_cand in tau_candidates:
                shifted = t + tau_cand
                x2 = shifted ** 2
                if len(t) >= 3:
                    slope, intercept = np.polyfit(x2, y, 1)
                    pred = slope * x2 + intercept
                    err = float(np.mean((pred - y) ** 2))
                    if err < best_err:
                        best_err = err
                        best_init = self.encode_params(
                            "pos_power",
                            {
                                "a": float(slope),
                                "beta": 2.0,
                                "tau": float(tau_cand),
                                "c": float(intercept),
                            },
                        )

                base = np.maximum(shifted, EPS)
                log_t = np.log(base)
                log_y = np.log(y_shift)
                if len(t) >= 3:
                    slope, intercept = np.polyfit(log_t, log_y, 1)
                    if slope > EPS:
                        pred_log = intercept + slope * log_t
                        err = float(np.mean((pred_log - log_y) ** 2))
                        if err < best_err:
                            best_err = err
                            best_init = self.encode_params(
                                "pos_power",
                                {
                                    "a": -min(
                                        max(
                                            float(
                                                np.exp(np.clip(intercept, -30, 30))
                                            ),
                                            EPS,
                                        ),
                                        1e3,
                                    ),
                                    "beta": max(float(slope), EPS),
                                    "tau": float(tau_cand),
                                    "c": c_est,
                                },
                            )
            if best_init is not None:
                return best_init
            raw[0] = -(y_max - y_min) / max(t_max**2, 1.0)
            raw[1] = np.log(np.expm1(2.0))
            raw[2] = -0.5 * t_span
            raw[3] = c_est
        elif name == "linear":
            slope = (y[-1] - y[0]) / max(t[-1] - t[0], 1.0)
            return self.encode_params(
                "linear",
                {
                    "a": float(slope),
                    "tau": 0.0,
                    "c": float(y[0] - slope * t[0]),
                },
            )
        return raw

    def _fit_bounds(
        self, name: str, t: np.ndarray, y: np.ndarray
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        t_min = float(t.min()) if len(t) else 0.0
        t_max = float(t.max()) if len(t) else 1.0
        y_range = max(float(y.max() - y.min()), 1e-3)
        y_max = float(y.max()) if len(y) else 1.0
        y_lo = -y_range * 5
        y_hi = y_range * 5 + y_max + 1

        if name == "exp":
            c_lo, c_hi = _exp_decay_c_bounds(y)
            t_span = max(t_max - t_min, 0.1)
            tau_lo = min(-t_span - 0.5, -1.5)
            tau_hi = max(t_span + 0.5, 1.5)
            bounds = [(-10.0, 10.0), (-10.0, 10.0), (tau_lo, tau_hi), (c_lo, c_hi)]
        elif name == "exp_growth":
            c_lo, c_hi = _exp_growth_c_bounds(y)
            a_hi = max(25.0 * y_range, 1.0)
            t_span = max(t_max - t_min, 0.1)
            tau_lo = min(-t_span - 0.5, -1.5)
            tau_hi = max(t_span + 0.5, 1.5)
            bounds = [
                (-a_hi, a_hi),
                (-10.0, 10.0),
                (tau_lo, tau_hi),
                (c_lo, c_hi),
            ]
        elif name == "const":
            bounds = [(y_lo, y_hi)]
        elif name == "sigmoid":
            bounds = [(y_lo, y_hi), (y_lo, y_hi), (-10.0, 10.0), (-3.0, 3.0)]
        elif name == "gaussian_spike":
            t_span = max(t_max - t_min, 0.1)
            tau_lo = min(-t_span - 0.5, -1.5)
            tau_hi = max(t_span + 0.5, 1.5)
            bounds = [
                (y_lo, y_hi),
                (y_lo, y_hi),
                (-10.0, 10.0),
                (tau_lo, tau_hi),
            ]
        elif name == "power":
            bounds = [
                (-5.0, 5.0),
                (-5.0, 5.0),
                (max(-t_min + EPS, -0.5), min(max(t_max, 0.1) + 0.5, 2.0)),
                (y_lo, y_hi),
            ]
        elif name == "pos_power":
            t_span = max(t_max - t_min, 0.1)
            tau_lo = min(-t_span - 0.5, -1.5)
            tau_hi = max(t_span + 0.5, 1.5)
            bounds = [
                (-10.0, 10.0),
                (-5.0, 5.0),
                (tau_lo, tau_hi),
                (y_lo, y_hi),
            ]
        elif name == "linear":
            t_span = max(t_max - t_min, 0.1)
            tau_lo = min(-t_span - 0.5, -1.5)
            tau_hi = max(t_span + 0.5, 1.5)
            bounds = [(-10.0, 10.0), (tau_lo, tau_hi), (y_lo, y_hi)]
        else:
            bounds = []

        n = PRIMITIVE_SPECS[name].num_params
        return bounds[:n] + [(None, None)] * (MAX_PARAMS - n)

    @staticmethod
    def param_count(name: str) -> int:
        return PRIMITIVE_SPECS[name].num_params

    @staticmethod
    def _shift_t_term(tau: float) -> str:
        if abs(tau) < 1e-4:
            return "t"
        if tau > 0:
            return f"(t+{tau:.4g})"
        return f"(t-{abs(tau):.4g})"

    @staticmethod
    def to_formula(name: str, physical: Dict[str, float]) -> str:
        """Human-readable formula string."""
        if name == "exp":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            return (
                f"{physical['a']:.4g}·exp(-{physical['r']:.4g}·{t_term})"
                f" + {physical['c']:.4g}"
            )
        if name == "exp_growth":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c']:.4g} {sign} {a_abs:.4g}·exp(+{physical['r']:.4g}·{t_term})"
            )
        if name == "const":
            return f"{physical['c']:.4g}"
        if name == "sigmoid":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c0']:.4g} {sign} {a_abs:.4g}·σ("
                f"{physical['k']:.4g}·{t_term})"
            )
        if name == "gaussian_spike":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c0']:.4g} {sign} {a_abs:.4g}·exp(-"
                f"{physical['k']:.4g}·{t_term}²)"
            )
        if name == "power":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            return (
                f"{physical['a']:.4g}·{t_term}^(-{physical['alpha']:.4g})"
                f" + {physical['c']:.4g}"
            )
        if name == "pos_power":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            return (
                f"{physical['a']:.4g}·{t_term}^({physical['beta']:.4g})"
                f" + {physical['c']:.4g}"
            )
        if name == "linear":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            return f"{physical['a']:.4g}·{t_term} + {physical['c']:.4g}"
        raise ValueError(name)

    @staticmethod
    def to_latex(name: str, physical: Dict[str, float]) -> str:
        if name == "exp":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            return (
                f"{physical['a']:.4g} e^{{-{physical['r']:.4g}{t_term}}}"
                f" + {physical['c']:.4g}"
            )
        if name == "exp_growth":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c']:.4g} {sign} {a_abs:.4g}\\,"
                f"e^{{{physical['r']:.4g}{t_term}}}"
            )
        if name == "const":
            return f"{physical['c']:.4g}"
        if name == "sigmoid":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c0']:.4g} {sign} {a_abs:.4g}\\,"
                f"\\sigma({physical['k']:.4g}{t_term})"
            )
        if name == "gaussian_spike":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            a = physical["a"]
            a_abs = abs(a)
            sign = "+" if a >= 0 else "-"
            return (
                f"{physical['c0']:.4g} {sign} {a_abs:.4g}\\,"
                f"e^{{-{physical['k']:.4g}{t_term}^2}}"
            )
        if name == "power":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            return (
                f"{physical['a']:.4g}{t_term}^{{-{physical['alpha']:.4g}}}"
                f" + {physical['c']:.4g}"
            )
        if name == "pos_power":
            t_term = PrimitiveBank._shift_t_term(physical["tau"])
            return (
                f"{physical['a']:.4g}{t_term}^{{{physical['beta']:.4g}}}"
                f" + {physical['c']:.4g}"
            )
        if name == "linear":
            t_term = PrimitiveBank._shift_t_term(physical.get("tau", 0.0))
            return f"{physical['a']:.4g}{t_term} + {physical['c']:.4g}"
        raise ValueError(name)
