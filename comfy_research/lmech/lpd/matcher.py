"""Hungarian matching for LPD breakpoint queries."""

from __future__ import annotations

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment


class BreakpointMatcher:
    def __init__(self, cost_breakpoint: float = 1.0):
        self.cost_breakpoint = cost_breakpoint

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pred_bp = outputs["pred_breakpoints"]
        bs = pred_bp.shape[0]

        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(bs):
            tgt_bp = targets[b]["breakpoints"]
            num_tgt = tgt_bp.shape[0]

            if num_tgt == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                    )
                )
                continue

            cost = torch.cdist(
                pred_bp[b].unsqueeze(1),
                tgt_bp.unsqueeze(1),
                p=1,
            )
            c = (self.cost_breakpoint * cost).cpu()
            row_idx, col_idx = linear_sum_assignment(c.numpy())
            indices.append(
                (
                    torch.as_tensor(row_idx, dtype=torch.long),
                    torch.as_tensor(col_idx, dtype=torch.long),
                )
            )
        return indices
