"""Differentiable phase router with ordered breakpoints."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PhaseRouter(nn.Module):
    """
    Soft segmentation over K phases using sigmoid difference gates.

    K positive segment proportions (sum to 1) define K-1 internal breakpoints.
    """

    def __init__(self, num_phases: int, t_max: float = 1.0, eps: float = 1e-4):
        super().__init__()
        if num_phases < 1:
            raise ValueError("num_phases must be >= 1")
        self.num_phases = num_phases
        self.t_max = t_max
        self.eps = eps

        if num_phases > 1:
            init = torch.ones(num_phases) / num_phases
            self.raw_segment_props = nn.Parameter(init.clone())
        else:
            self.register_parameter("raw_segment_props", None)

    def set_t_max(self, t_max: float) -> None:
        self.t_max = float(t_max)

    def segment_props(self) -> torch.Tensor:
        """Normalized segment proportions, shape (K,)."""
        if self.num_phases == 1:
            return torch.ones(1, device=self._device())
        props = torch.nn.functional.softplus(self.raw_segment_props) + self.eps
        return props / props.sum()

    def breakpoints(self) -> torch.Tensor:
        """Ordered internal breakpoints strictly inside (0, t_max), shape (K-1,)."""
        if self.num_phases == 1:
            return torch.tensor([], device=self._device())
        props = self.segment_props()
        cum = torch.cumsum(props, dim=0)
        return cum[:-1] * self.t_max

    def bounds(self) -> torch.Tensor:
        """Phase boundaries [0, b1,..., t_max], shape (K+1,)."""
        if self.num_phases == 1:
            return torch.tensor([0.0, self.t_max], device=self._device())
        b = self.breakpoints()
        return torch.cat(
            [
                torch.zeros(1, device=b.device, dtype=b.dtype),
                b,
                torch.tensor([self.t_max], device=b.device, dtype=b.dtype),
            ]
        )

    def _device(self):
        if self.raw_segment_props is not None:
            return self.raw_segment_props.device
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def forward(
        self, t: torch.Tensor, beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bounds = self.bounds()
        k = self.num_phases
        n = t.shape[0]

        if k == 1:
            w = torch.ones(n, 1, device=t.device, dtype=t.dtype)
            return w, bounds

        w_raw = torch.ones(n, k, device=t.device, dtype=t.dtype)
        for i in range(k):
            left = torch.sigmoid(beta * (t - bounds[i]))
            right = torch.sigmoid(beta * (bounds[i + 1] - t))
            w_raw[:, i] = left * right

        w = w_raw / (w_raw.sum(dim=1, keepdim=True) + self.eps)
        return w, bounds

    def hard_assign(self, t: torch.Tensor, beta: float = 500.0) -> torch.Tensor:
        w, _ = self.forward(t, beta=beta)
        return w.argmax(dim=1)

    def phase_entropy(self, t: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        w, _ = self.forward(t, beta=beta)
        ent = -(w * torch.log(w + self.eps)).sum(dim=1)
        return ent.mean()

    def set_segment_props(self, props: torch.Tensor) -> None:
        """Set segment proportions from a normalized (K,) vector."""
        props = props / props.sum()
        inv = torch.log(torch.expm1(props.clamp(min=self.eps) * 10.0))
        with torch.no_grad():
            self.raw_segment_props.copy_(inv)
