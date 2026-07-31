"""Breakpoint regression loss for LPD."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_research.lmech.lpd.matcher import BreakpointMatcher


class SetCriterion(nn.Module):
    """LPD training loss: matched breakpoint L1 + query objectness."""

    def __init__(
        self,
        matcher: BreakpointMatcher,
        loss_score_weight: float = 0.1,
    ):
        super().__init__()
        self.matcher = matcher
        self.loss_score_weight = loss_score_weight

    def _get_src_permutation_idx(self, indices: List[tuple]) -> tuple:
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        pred_breakpoints = outputs["pred_breakpoints"]
        pred_scores = outputs["pred_scores"]
        indices = self.matcher.forward(outputs, targets)

        num_breakpoints = sum(len(t["breakpoints"]) for t in targets)
        num_breakpoints = max(num_breakpoints, 1)

        idx = self._get_src_permutation_idx(indices)
        src_bp = pred_breakpoints[idx]
        target_bp = torch.cat(
            [t["breakpoints"][j] for t, (_, j) in zip(targets, indices)],
            dim=0,
        )
        loss_bp = F.l1_loss(src_bp, target_bp, reduction="sum") / num_breakpoints

        loss_score = pred_scores.new_tensor(0.0)
        for b, (src_idx, _) in enumerate(indices):
            target_scores = torch.zeros_like(pred_scores[b])
            if src_idx.numel() > 0:
                target_scores[src_idx] = 1.0
            loss_score = loss_score + F.binary_cross_entropy(
                pred_scores[b], target_scores
            )
        loss_score = loss_score / max(len(targets), 1)

        total = loss_bp + self.loss_score_weight * loss_score
        return {
            "loss_bp": loss_bp,
            "loss_score": loss_score,
            "loss": total,
        }
