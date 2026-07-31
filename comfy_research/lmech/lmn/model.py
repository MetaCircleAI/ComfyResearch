"""Learning Mechanics Network model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.lmech.lmn.json_util import finite_float
from comfy_research.lmech.lmn.phase_router import PhaseRouter
from comfy_research.lmech.lmn.primitives import MAX_PARAMS, NUM_PRIMITIVES, PRIMITIVE_NAMES, NAME_TO_ID, PrimitiveBank


class LMN(nn.Module):
    """
    Phase-mechanics network:

        L_hat(t) = sum_k w_k(t) * sum_m pi[k,m] * phi_m(t; theta[k,m])
    """

    def __init__(
        self,
        num_phases: int,
        t_max: float = 1.0,
        lambda_pi: float = 1e-3,
        lambda_w: float = 1e-4,
    ):
        super().__init__()
        self.num_phases = num_phases
        self.lambda_pi = lambda_pi
        self.lambda_w = lambda_w

        self.router = PhaseRouter(num_phases, t_max=t_max)
        self.primitives = PrimitiveBank()
        self.pi_logits = nn.Parameter(torch.zeros(num_phases, NUM_PRIMITIVES))
        self.theta = nn.Parameter(
            torch.randn(num_phases, NUM_PRIMITIVES, MAX_PARAMS) * 0.1
        )

    def set_t_max(self, t_max: float) -> None:
        self.router.set_t_max(t_max)

    @property
    def pi(self) -> torch.Tensor:
        return F.softmax(self.pi_logits, dim=-1)

    def forward(
        self, t: torch.Tensor, beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            L_hat: (N,)
            w: (N, K) phase weights
            g: (N, K) phase outputs (mixture over primitives)
        """
        w, _ = self.router(t, beta=beta)
        pi = self.pi  # (K, M)

        # (N, K, M) primitive values
        phi = self.primitives.eval_all_phases(t, self.theta)

        # Weighted sum over primitives per phase: (N, K)
        g = (phi * pi.unsqueeze(0)).sum(dim=-1)

        # Weighted sum over phases: (N,)
        L_hat = (w * g).sum(dim=-1)
        return L_hat, w, g

    def loss(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        L_hat, w, _ = self.forward(t, beta=beta)
        mse = F.mse_loss(L_hat, y)
        l1_pi = self.pi.abs().sum()
        ent_w = self.router.phase_entropy(t, beta=beta)
        total = mse + self.lambda_pi * l1_pi - self.lambda_w * ent_w
        metrics = {
            "mse": finite_float(float(mse.item())),
            "l1_pi": float(l1_pi.item()),
            "phase_entropy": float(ent_w.item()),
            "total": finite_float(float(total.item())),
        }
        return total, metrics

    def initialize_from_data(
        self, t: torch.Tensor, y: torch.Tensor, num_init_phases: Optional[int] = None
    ) -> None:
        """Data-driven init: quantile breakpoints + per-segment primitive warm-start."""
        t_np = t.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        t_max = float(t_np.max())
        k = num_init_phases or self.num_phases
        bank = PrimitiveBank()

        # Segment-length initialization: turning points for non-monotonic curves,
        # otherwise equal-time quantiles.
        if self.num_phases > 1:
            from comfy_research.lmech.lmn.finetize import turning_point_breakpoints

            tp_bps = turning_point_breakpoints(t_np, y_np, self.num_phases)
            if tp_bps is not None:
                target_b = np.array(tp_bps, dtype=float)
            else:
                quantiles = np.linspace(0, 1, self.num_phases + 1)[1:-1]
                target_b = np.quantile(t_np, quantiles)
            bounds = np.concatenate([[0.0], target_b, [t_max]])
            lengths = np.diff(bounds)
            lengths = np.maximum(lengths, t_max * 0.01)
            props = lengths / lengths.sum()
            with torch.no_grad():
                self.router.set_segment_props(
                    torch.tensor(props, dtype=t.dtype, device=t.device)
                )

        # Assign samples to phases using current breakpoints
        bounds = self.router.bounds().detach().cpu().numpy()
        for ki in range(self.num_phases):
            lo, hi = bounds[ki], bounds[ki + 1]
            if ki < self.num_phases - 1:
                mask = (t_np >= lo) & (t_np < hi)
            else:
                mask = (t_np >= lo) & (t_np <= hi)
            if mask.sum() < 3:
                continue
            t_seg, y_seg = t_np[mask], y_np[mask]

            best_name, best_mse = "const", float("inf")
            best_raw = np.zeros(MAX_PARAMS)
            for mi, name in enumerate(PRIMITIVE_NAMES):
                raw, mse = bank.fit_single(t_seg, y_seg, name)
                if mse < best_mse:
                    best_mse = mse
                    best_name = name
                    best_raw = raw

            with torch.no_grad():
                mid = NAME_TO_ID[best_name]
                seg_mean = float(y_seg.mean())
                self.pi_logits[ki, :] = -20.0
                self.pi_logits[ki, mid] = 20.0
                for mi, pname in enumerate(PRIMITIVE_NAMES):
                    safe_raw = bank.safe_flat_raw(pname, seg_mean)
                    self.theta[ki, mi, :] = torch.tensor(
                        safe_raw, dtype=self.theta.dtype, device=self.theta.device
                    )
                self.theta[ki, mid, :] = torch.tensor(
                    best_raw, dtype=self.theta.dtype, device=self.theta.device
                )

    @torch.no_grad()
    def predict(self, t: torch.Tensor, beta: float = 200.0) -> torch.Tensor:
        L_hat, _, _ = self.forward(t, beta=beta)
        return L_hat
