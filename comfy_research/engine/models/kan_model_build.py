"""Build Kolmogorov–Arnold networks from graph node data using pykan (https://github.com/KindXiaoming/pykan)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from fastapi import HTTPException

# MultKAN maps only these string names to ``torch.nn`` modules; other strings break KANLayer.
_ALLOWED_BASE_FUN = frozenset({"silu", "identity", "zero"})


def _import_kan() -> Any:
    try:
        from kan import KAN
    except ImportError as e:
        msg = str(e).strip() or e.__class__.__name__
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not import pykan's `kan` module. Use the same Python as for `python app.py` "
                "and install project dependencies (`pip install -r requirements.txt` — pykan needs "
                "scikit-learn, tqdm, pandas, PyYAML, etc., which are not always pulled in by `pip install pykan`). "
                f"Import error: {msg}. See https://github.com/KindXiaoming/pykan."
            ),
        ) from e
    return KAN


def kan_width_list(input_dim: int, output_dim: int, depth: int, hidden_width: int) -> list[int]:
    """``[n_in] + depth * [hidden] + [n_out]`` — same depth convention as the MLP node."""
    if input_dim < 1 or output_dim < 1 or depth < 1 or hidden_width < 1:
        raise HTTPException(
            status_code=400,
            detail="KAN input_dim, output_dim, depth, and width must be >= 1.",
        )
    return [input_dim] + [hidden_width] * depth + [output_dim]


def build_kan_regression(
    input_dim: int,
    output_dim: int,
    depth: int,
    hidden_width: int,
    grid: int,
    spline_order: int,
    seed: int,
    base_fun: str,
    *,
    fast_training: bool = True,
) -> nn.Module:
    """A KAN suitable for 2D float tensor regression (MSE), with symbolic branch disabled for speed.

    When ``fast_training`` is False (e.g. KAN regularization observables are wired), ``save_act`` stays
    true and ``speed()`` is skipped so ``get_reg`` sees activations during training.
    """
    bf = str(base_fun or "silu").strip().lower()
    if bf not in _ALLOWED_BASE_FUN:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported KAN base_fun {base_fun!r}; use one of {sorted(_ALLOWED_BASE_FUN)}.",
        )
    if grid < 1 or spline_order < 1:
        raise HTTPException(status_code=400, detail="KAN grid and spline order k must be >= 1.")
    KAN = _import_kan()
    width = kan_width_list(input_dim, output_dim, depth, hidden_width)
    model = KAN(
        width=width,
        grid=grid,
        k=spline_order,
        seed=seed,
        base_fun=bf,
        symbolic_enabled=False,
        auto_save=False,
        save_act=not fast_training,
        device="cpu",
    )
    if fast_training:
        model.speed()
    return model


def build_kan_for_plot(
    input_dim: int,
    output_dim: int,
    depth: int,
    hidden_width: int,
    grid: int,
    spline_order: int,
    seed: int,
    base_fun: str,
) -> Any:
    """KAN instance suitable for ``KAN.plot()`` (needs ``save_act=True``; do not call ``speed()``)."""
    bf = str(base_fun or "silu").strip().lower()
    if bf not in _ALLOWED_BASE_FUN:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported KAN base_fun {base_fun!r}; use one of {sorted(_ALLOWED_BASE_FUN)}.",
        )
    if grid < 1 or spline_order < 1:
        raise HTTPException(status_code=400, detail="KAN grid and spline order k must be >= 1.")
    KAN = _import_kan()
    width = kan_width_list(input_dim, output_dim, depth, hidden_width)
    return KAN(
        width=width,
        grid=grid,
        k=spline_order,
        seed=seed,
        base_fun=bf,
        symbolic_enabled=False,
        auto_save=False,
        save_act=True,
        device="cpu",
    )
