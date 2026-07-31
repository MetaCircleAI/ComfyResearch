"""Synthetic data generation and CSV I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def ground_truth_loss(t: np.ndarray) -> np.ndarray:
    """4-phase demo ground truth from the plan."""
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti < 500:
            y[i] = 1.2 * np.exp(-0.03 * ti) + 0.4
        elif ti < 3000:
            y[i] = 0.38
        elif ti < 6000:
            y[i] = 0.25 + 0.13 / (1.0 + np.exp(-0.01 * (ti - 4200)))
        else:
            y[i] = 0.18 * (t[i] + 1e-6) ** (-0.4) + 0.02
    return y


GROUND_TRUTH_PHASES = [
    {"name": "exp", "t_start": 0, "t_end": 500},
    {"name": "const", "t_start": 500, "t_end": 3000},
    {"name": "sigmoid", "t_start": 3000, "t_end": 6000},
    {"name": "power", "t_start": 6000, "t_end": np.inf},
]

GROUND_TRUTH_BREAKPOINTS = [500, 3000, 6000]


def generate_synthetic_demo(
    t_max: float = 10000.0,
    step: float = 50.0,
    noise_std: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, t_max + step, step)
    y = ground_truth_loss(t)
    if noise_std > 0:
        y = y + rng.normal(0, noise_std, size=y.shape)
    return pd.DataFrame({"t": t, "loss": y})


def load_curve_csv(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    t_col = cols.get("t") or cols.get("step") or cols.get("steps")
    loss_col = cols.get("loss") or cols.get("l") or cols.get("value")
    if t_col is None or loss_col is None:
        raise ValueError("CSV must contain columns 't' and 'loss' (or aliases)")
    t = torch.tensor(df[t_col].values, dtype=torch.float32)
    loss = torch.tensor(df[loss_col].values, dtype=torch.float32)
    return t, loss, df


def smooth_curve(
    t: torch.Tensor, loss: torch.Tensor, window: int = 5
) -> torch.Tensor:
    """Simple moving average smoothing."""
    if window <= 1:
        return loss
    kernel = torch.ones(window) / window
    padded = torch.nn.functional.pad(
        loss.unsqueeze(0).unsqueeze(0),
        (window // 2, window // 2),
        mode="replicate",
    )
    smoothed = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).squeeze()
    return smoothed
