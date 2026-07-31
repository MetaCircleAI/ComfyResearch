"""Compile ``symbolic_func_dataset`` LaTeX → vectorized NumPy y(x)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp
from fastapi import HTTPException
from sympy.parsing.latex import parse_latex


def _scalar_int(x: Any, default: int = 0) -> int:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return int(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return float(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _expr_size(expr: sp.Expr, limit: int = 400) -> int:
    n = 0

    def walk(e: sp.Basic) -> None:
        nonlocal n
        n += 1
        if n > limit:
            raise HTTPException(
                status_code=400,
                detail="Symbolic expression is too large; simplify the LaTeX.",
            )
        for a in e.args:
            walk(a)

    walk(expr)
    return n


def _normalize_latex(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s.strip())
    return s.strip()


def _expand_sum_over_dummy(
    expr: sp.Expr,
    input_dim: int,
) -> sp.Expr:
    """Turn Sum(..., (i, lo, hi)) into explicit Add when dummy i indexes x_{{i}}."""
    if not isinstance(expr, sp.Sum):
        return expr
    if len(expr.limits) != 1:
        raise HTTPException(status_code=400, detail="Only single-index sums are supported in LaTeX.")
    v, lo, hi = expr.limits[0]
    lo_i = int(lo) if lo.is_Integer else int(sp.N(lo))
    hi_sym = hi
    if hi_sym in (sp.Symbol("d"), sp.Symbol("D")):
        hi_i = input_dim
    elif hi_sym.is_Integer:
        hi_i = int(hi_sym)
    else:
        try:
            hi_i = int(sp.N(hi_sym))
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Sum upper bound must be an integer or symbol `d` matching input dimension.",
            ) from None
    if lo_i != 1:
        raise HTTPException(status_code=400, detail="Sum lower bound must be 1 for this dataset node.")
    if hi_i != input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Sum upper bound ({hi_i}) must equal inputDim ({input_dim}).",
        )
    xi_sym = sp.Symbol("x_{i}")
    parts: list[sp.Expr] = []
    for k in range(lo_i, hi_i + 1):
        xk = sp.Symbol(f"x_{k}")
        part = expr.function.subs({v: sp.Integer(k), xi_sym: xk})
        parts.append(sp.simplify(part))
    return sp.Add(*parts)


def _expand_supported_sums(expr: sp.Expr, input_dim: int) -> sp.Expr:
    """Expand any supported Sum nodes anywhere in the expression tree."""
    sums = list(expr.atoms(sp.Sum))
    if not sums:
        return expr
    out = expr
    for s in sums:
        out = out.xreplace({s: _expand_sum_over_dummy(s, input_dim)})
    return out


def _canonicalize_feature_symbols(expr: sp.Expr) -> sp.Expr:
    """Normalize indexed features to ``x_k`` form (accept ``x_{k}`` from LaTeX parser)."""
    repl: dict[sp.Symbol, sp.Symbol] = {}
    for sym in expr.free_symbols:
        m = re.fullmatch(r"x_\{(\d+)\}", str(sym))
        if m is None:
            continue
        repl[sym] = sp.Symbol(f"x_{int(m.group(1))}")
    if not repl:
        return expr
    return expr.xreplace(repl)


def _subs_pi_symbols_for_constants(expr: sp.Expr, extra_param_names: set[str]) -> sp.Expr:
    """LaTeX ``\\pi`` often becomes a bare ``Symbol('pi')``; map to ``sympy.pi`` unless ``pi`` is a named extra."""
    if "pi" in extra_param_names or "Pi" in extra_param_names:
        return expr
    repl: dict[sp.Basic, sp.Basic] = {}
    for s in expr.free_symbols:
        if not isinstance(s, sp.Symbol):
            continue
        name = str(s)
        if name in ("pi", "Pi"):
            repl[s] = sp.pi
    return expr.xreplace(repl) if repl else expr


def _canonicalize_named_params(expr: sp.Expr, param_names: set[str]) -> sp.Expr:
    """Map products like b*a*i*s back to Symbol('bias') for known extras params."""
    keys = [k for k in sorted(param_names) if re.fullmatch(r"[A-Za-z]+", k) and len(k) >= 2]
    if not keys:
        return expr

    target_multisets: dict[str, list[str]] = {k: sorted(list(k)) for k in keys}

    def repl(node: sp.Basic) -> sp.Basic:
        if not isinstance(node, sp.Mul):
            return node
        powers = node.as_powers_dict()
        letters: list[str] = []
        for base, exp in powers.items():
            if not isinstance(base, sp.Symbol):
                return node
            if not exp.is_Integer or int(exp) != 1:
                return node
            s = str(base)
            if not re.fullmatch(r"[A-Za-z]", s):
                return node
            letters.append(s)
        letters_sorted = sorted(letters)
        for name, target in target_multisets.items():
            if letters_sorted == target:
                return sp.Symbol(name)
        return node

    return expr.replace(lambda e: isinstance(e, sp.Mul), repl)


def _merge_user_symbolic_blueprint(dd: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields (especially ``equationLatex``) from a saved user blueprint on the canvas."""
    merged = dict(dd)
    if str(merged.get("equationLatex") or "").strip():
        return merged
    uid = str(merged.get("userSymbolicFuncDatasetId") or "").strip()
    if not uid:
        return merged
    try:
        from comfy_research.api.user_symbolic_func_datasets import load_user_symbolic_node_data
    except Exception:
        return merged
    patch = load_user_symbolic_node_data(uid)
    if not patch:
        return merged
    for k, v in patch.items():
        cur = merged.get(k)
        if k not in merged or (isinstance(cur, str) and not str(cur or "").strip()):
            merged[k] = v
    return merged


def _target_evaluation_columns(
    x_np: np.ndarray,
    *,
    input_dim: int,
    input_distribution: str,
    evaluation_precision: str,
) -> list[np.ndarray]:
    """Return target-function inputs without changing model input tensors."""
    if (
        evaluation_precision == "float64"
        and input_dim == 1
        and input_distribution in {"linspace_0_1", "linspace_0_1_endpoint_excluded"}
    ):
        return [
            np.linspace(
                0.0,
                1.0,
                num=x_np.shape[0],
                endpoint=input_distribution == "linspace_0_1",
                dtype=np.float64,
            )
        ]
    dtype = np.float64 if evaluation_precision == "float64" else None
    return [
        x_np[:, index].astype(dtype, copy=False) if dtype is not None else x_np[:, index]
        for index in range(input_dim)
    ]


def build_y_numpy_fn(dd: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(x_np) -> y_np`` with ``x_np`` (N, input_dim), ``y_np`` (N, 1)."""
    dd = _merge_user_symbolic_blueprint(dd)
    latex_raw = str(dd.get("equationLatex") or "").strip()
    if not latex_raw:
        raise HTTPException(status_code=400, detail="symbolic_func_dataset: equationLatex is required.")
    input_dim = _scalar_int(dd.get("inputDim"), 10)
    out_dim = _scalar_int(dd.get("outputDim"), 1)
    evaluation_precision = str(dd.get("evaluationPrecision") or "input").strip().lower()
    input_distribution = str(dd.get("inputDistribution") or "").strip().lower()
    if input_dim < 1:
        raise HTTPException(status_code=400, detail="symbolic_func_dataset: inputDim must be >= 1.")
    if out_dim != 1:
        raise HTTPException(status_code=400, detail="symbolic_func_dataset: outputDim must be 1 in v1.")
    if evaluation_precision not in {"input", "float64"}:
        raise HTTPException(
            status_code=400,
            detail="symbolic_func_dataset: evaluationPrecision must be input or float64.",
        )

    f_sym = sp.Symbol("f")
    has_frequency = "frequency" in dd
    f_val = _scalar_float(dd.get("frequency"), 10.0) if has_frequency else None

    latex = _normalize_latex(latex_raw)
    try:
        expr = parse_latex(latex)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse LaTeX (install sympy + antlr4-python3-runtime==4.11.1): {e}",
        ) from e

    expr = sp.sympify(expr)
    expr = expr.subs({sp.Symbol("d"): sp.Integer(input_dim), sp.Symbol("D"): sp.Integer(input_dim)})
    _expr_size(expr)

    extras_raw = dd.get("extras")
    extras: dict[str, float] = {}
    if isinstance(extras_raw, dict):
        for k, v in extras_raw.items():
            if not isinstance(k, str):
                continue
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                extras[k] = float(v)

    expr = _expand_supported_sums(expr, input_dim)
    expr = _canonicalize_feature_symbols(expr)
    expr = _canonicalize_named_params(expr, set(extras.keys()))
    expr = _subs_pi_symbols_for_constants(expr, set(extras.keys()))
    _expr_size(expr)

    expected_x = {sp.Symbol(f"x_{j}") for j in range(1, input_dim + 1)}
    free = expr.free_symbols
    allowed_params = ({f_sym} if has_frequency else set()) | {sp.Symbol(k) for k in extras.keys()}
    unknown = free - expected_x - allowed_params
    if unknown:
        allowed_txt = (
            f"f, x_1..x_{input_dim}, pi (π)"
            if has_frequency
            else f"x_1..x_{input_dim}, pi (π)"
        )
        if extras:
            allowed_txt = f"{allowed_txt}, {', '.join(sorted(extras.keys()))}"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown symbols in equation (allowed: {allowed_txt}): {', '.join(sorted(str(s) for s in unknown))}",
        )

    xs_syms = [sp.Symbol(f"x_{j}") for j in range(1, input_dim + 1)]
    subs_map: dict[sp.Symbol, sp.Float] = {}
    if has_frequency and f_val is not None:
        subs_map[f_sym] = sp.Float(f_val)
    for k, v in extras.items():
        subs_map[sp.Symbol(k)] = sp.Float(v)
    expr2 = sp.simplify(expr.subs(subs_map)) if subs_map else sp.simplify(expr)
    _expr_size(expr2)

    try:
        fn = sp.lambdify(xs_syms, expr2, modules=["numpy"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not compile expression to NumPy: {e}") from e

    def y_of_x(x_np: np.ndarray) -> np.ndarray:
        if x_np.ndim != 2 or x_np.shape[1] != input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected x with shape (N, {input_dim}), got {x_np.shape}.",
            )
        cols = _target_evaluation_columns(
            x_np,
            input_dim=input_dim,
            input_distribution=input_distribution,
            evaluation_precision=evaluation_precision,
        )
        try:
            out = fn(*cols)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error evaluating symbolic y(x): {e}") from e
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full((x_np.shape[0], 1), float(arr), dtype=np.float32)
        else:
            arr = np.reshape(arr, (x_np.shape[0], -1))
        if arr.shape[1] != 1:
            raise HTTPException(
                status_code=400,
                detail=f"Symbolic expression must yield scalar per row; got width {arr.shape[1]}.",
            )
        return arr.astype(np.float32)

    return y_of_x
