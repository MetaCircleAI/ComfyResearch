from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException

from comfy_research.schemas.graph import Node, NodeKind

TrainerTask = Literal[
    "mse_regression",
    "token_classification",
    "cross_entropy_dense",
    "diffusion_noise",
    "vision_classification",
]


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


def _scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


class MaskedMSELoss(nn.Module):
    """Per-feature weighted MSE: mean over batch of sum_j w_j (pred_ij - y_ij)^2 / sum_j w_j."""

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask.reshape(1, -1).float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"MaskedMSELoss expects pred/target same shape; got {pred.shape} vs {target.shape}")
        if pred.dim() != 2:
            raise ValueError(f"MaskedMSELoss expects [batch, features]; got {tuple(pred.shape)}")
        f = int(pred.shape[1])
        if int(self.mask.shape[1]) != f:
            raise ValueError(f"Loss mask length {int(self.mask.shape[1])} does not match output dim {f}.")
        diff2 = (pred - target).pow(2)
        m = self.mask.to(device=diff2.device, dtype=diff2.dtype)
        wsum = m.sum().clamp_min(torch.tensor(1e-12, device=diff2.device, dtype=diff2.dtype))
        return (diff2 * m).sum() / (wsum * float(pred.shape[0]))


def _expand_context_mask_weights(weights_ctx: list[float], out_features: int, ctx_len: int) -> torch.Tensor:
    """Repeat each context-slot weight across a contiguous block of ``out_features // ctx_len`` features."""
    if ctx_len < 1:
        raise HTTPException(status_code=400, detail="lossMaskContextLength must be >= 1.")
    if out_features % ctx_len != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Output flat dim {out_features} is not divisible by lossMaskContextLength={ctx_len} "
                "(cannot group features into context slots)."
            ),
        )
    block = out_features // ctx_len
    if len(weights_ctx) != ctx_len:
        raise HTTPException(
            status_code=400,
            detail=(
                f"MSE loss mask expects {ctx_len} context-slot weights when lossMaskContextLength={ctx_len}; "
                f"got {len(weights_ctx)}."
            ),
        )
    full: list[float] = []
    for w in weights_ctx:
        full.extend([float(w)] * block)
    return torch.tensor(full, dtype=torch.float32)


def _mse_criterion_from_loss_data(loss_d: dict[str, Any], out_features: int) -> nn.Module:
    mode = _scalar_str(loss_d.get("lossMaskMode"), "all").strip().lower()
    if mode == "last_half":
        mode = "last_context"
    ctx_len = max(1, _scalar_int(loss_d.get("lossMaskContextLength"), 1))
    if mode in {"", "all", "none", "unmasked"}:
        return nn.MSELoss()
    if mode == "last_context":
        if ctx_len == 1:
            w = torch.zeros(out_features, dtype=torch.float32)
            if out_features < 1:
                raise HTTPException(status_code=400, detail="MSE last_context with context length 1 needs output dim >= 1.")
            w[-1] = 1.0
            return MaskedMSELoss(w)
        weights_ctx = [0.0] * (ctx_len - 1) + [1.0]
        w = _expand_context_mask_weights(weights_ctx, out_features, ctx_len)
        return MaskedMSELoss(w)
    if mode == "custom":
        raw = str(loss_d.get("lossMaskCustom", "")).strip()
        if not raw:
            raise HTTPException(
                status_code=400,
                detail="MSE loss mask mode 'custom' requires a non-empty lossMaskCustom (comma-separated weights).",
            )
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        try:
            weights_ctx = [float(p) for p in parts]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"MSE loss custom mask has invalid float: {e}") from e
        if any(w < 0 for w in weights_ctx):
            raise HTTPException(status_code=400, detail="MSE loss custom mask weights must be >= 0.")
        if sum(weights_ctx) <= 0:
            raise HTTPException(status_code=400, detail="MSE loss custom mask weights must sum to > 0.")
        if ctx_len == 1:
            if len(weights_ctx) != out_features:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"When lossMaskContextLength is 1, custom mask needs {out_features} weights "
                        f"(one per output coordinate); got {len(weights_ctx)}."
                    ),
                )
            return MaskedMSELoss(torch.tensor(weights_ctx, dtype=torch.float32))
        w = _expand_context_mask_weights(weights_ctx, out_features, ctx_len)
        return MaskedMSELoss(w)
    raise HTTPException(
        status_code=400,
        detail="mse_loss lossMaskMode must be one of: all, last_context, custom.",
    )


def dense_ce_memorization_a_slot_groups(loss_d: dict[str, Any], *, trainer_task: TrainerTask, ds_type: object) -> int:
    """T>1 groups flat logits into T independent V-way heads (memorization A + dense CE only)."""
    if trainer_task != "cross_entropy_dense" or str(getattr(ds_type, "value", ds_type)) != NodeKind.memorization_a_dataset.value:
        return 1
    mode = _scalar_str(loss_d.get("lossMaskMode"), "all").strip().lower()
    if mode == "last_half":
        mode = "last_context"
    if mode in ("", "all", "none", "unmasked"):
        return 1
    ctx_len = max(1, _scalar_int(loss_d.get("lossMaskContextLength"), 1))
    if ctx_len < 2:
        return 1
    if mode == "last_context":
        return ctx_len
    if mode == "custom":
        raw = str(loss_d.get("lossMaskCustom", "")).strip()
        if not raw:
            return 1
        return ctx_len
    raise HTTPException(
        status_code=400,
        detail="cross_entropy_loss lossMaskMode must be one of: all, last_context, custom.",
    )


def _ce_slot_context_weights(loss_d: dict[str, Any], ctx_len: int) -> list[float]:
    mode = _scalar_str(loss_d.get("lossMaskMode"), "all").strip().lower()
    if mode == "last_half":
        mode = "last_context"
    if mode == "last_context":
        if ctx_len < 2:
            raise HTTPException(status_code=400, detail="cross_entropy last_context mask needs lossMaskContextLength >= 2.")
        return [0.0] * (ctx_len - 1) + [1.0]
    if mode == "custom":
        raw = str(loss_d.get("lossMaskCustom", "")).strip()
        if not raw:
            raise HTTPException(
                status_code=400,
                detail="cross_entropy loss mask mode 'custom' requires a non-empty lossMaskCustom (comma-separated weights).",
            )
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        try:
            weights_ctx = [float(p) for p in parts]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"cross_entropy loss custom mask has invalid float: {e}") from e
        if any(w < 0 for w in weights_ctx):
            raise HTTPException(status_code=400, detail="cross_entropy loss custom slot weights must be >= 0.")
        if sum(weights_ctx) <= 0:
            raise HTTPException(status_code=400, detail="cross_entropy loss custom slot weights must sum to > 0.")
        if len(weights_ctx) != ctx_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"cross_entropy loss custom mask expects {ctx_len} slot weights when lossMaskContextLength={ctx_len}; "
                    f"got {len(weights_ctx)}."
                ),
            )
        return weights_ctx
    raise HTTPException(
        status_code=400,
        detail="cross_entropy_loss lossMaskMode must be one of: all, last_context, custom.",
    )


class WeightedSameLabelMultiSlotCrossEntropyLoss(nn.Module):
    """T independent V-way softmax CEs on logits ``[batch, T*V]``, same integer target per slot."""

    def __init__(
        self,
        slot_weights: torch.Tensor,
        vocab_per_slot: int,
        *,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("slot_weights", slot_weights.reshape(-1).float())
        self.vocab_per_slot = int(vocab_per_slot)
        self.label_smoothing = float(max(0.0, min(1.0, label_smoothing)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() != 2:
            raise ValueError(f"WeightedSameLabelMultiSlotCrossEntropyLoss expects pred [batch, T*V]; got {tuple(pred.shape)}")
        if target.dim() != 1:
            raise ValueError(f"WeightedSameLabelMultiSlotCrossEntropyLoss expects target [batch]; got {tuple(target.shape)}")
        b, f = int(pred.shape[0]), int(pred.shape[1])
        t = int(self.slot_weights.shape[0])
        v = self.vocab_per_slot
        if f != t * v:
            raise ValueError(
                f"WeightedSameLabelMultiSlotCrossEntropyLoss expects pred dim {t * v} (= T * V); got {f} (T={t}, V={v})."
            )
        logits = pred.reshape(b * t, v)
        tgt = target.reshape(b, 1).expand(b, t).reshape(b * t).long()
        ce_flat = F.cross_entropy(
            logits,
            tgt,
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(b, t)
        w = self.slot_weights.to(device=ce_flat.device, dtype=ce_flat.dtype)
        wsum = w.sum().clamp_min(torch.tensor(1e-12, device=ce_flat.device, dtype=ce_flat.dtype))
        return (ce_flat * w.unsqueeze(0)).sum() / (wsum * float(b))


def _cross_entropy_dense_criterion_from_loss_data(loss_d: dict[str, Any], num_logits: int) -> nn.Module:
    ls = float(max(0.0, min(1.0, _scalar_float(loss_d.get("labelSmoothing"), 0.0))))
    mode = _scalar_str(loss_d.get("lossMaskMode"), "all").strip().lower()
    if mode == "last_half":
        mode = "last_context"
    ctx_len = max(1, _scalar_int(loss_d.get("lossMaskContextLength"), 1))
    if mode in ("", "all", "none", "unmasked") or ctx_len < 2:
        return nn.CrossEntropyLoss(label_smoothing=ls)
    weights_ctx = _ce_slot_context_weights(loss_d, ctx_len)
    if num_logits % ctx_len != 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"cross_entropy slot mask needs model output dim divisible by lossMaskContextLength={ctx_len}; "
                f"got output dim {num_logits}."
            ),
        )
    v = num_logits // ctx_len
    w = torch.tensor(weights_ctx, dtype=torch.float32)
    return WeightedSameLabelMultiSlotCrossEntropyLoss(w, v, label_smoothing=ls)


class MultiSlotCrossEntropyLoss(nn.Module):
    """Cross-entropy over K independent vocab heads: pred ``[batch, K, V]``, target ``[batch, K]``."""

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        super().__init__()
        ls = float(max(0.0, min(1.0, label_smoothing)))
        self._ce = nn.CrossEntropyLoss(label_smoothing=ls)

    @property
    def label_smoothing(self) -> float:
        return float(self._ce.label_smoothing)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() != 3:
            raise ValueError("MultiSlotCrossEntropyLoss expects pred with shape [batch, K, vocab]")
        if target.dim() != 2:
            raise ValueError("MultiSlotCrossEntropyLoss expects target with shape [batch, K]")
        b, k, v = pred.shape
        return self._ce(pred.reshape(b * k, v), target.reshape(b * k).long())


class BinaryCrossEntropyWithLogitsForClassLabels(nn.Module):
    """BCE-with-logits adapter for integer class labels 0/1."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 2 and int(pred.shape[1]) == 1:
            logits = pred[:, 0]
        elif pred.dim() == 1:
            logits = pred
        else:
            raise ValueError(
                "binary_cross_entropy_with_logits_loss expects one logit per row; "
                f"got prediction shape {tuple(pred.shape)}"
            )
        labels = target.reshape(-1)
        if int(logits.shape[0]) != int(labels.shape[0]):
            raise ValueError(
                "binary_cross_entropy_with_logits_loss prediction/target batch sizes differ: "
                f"{int(logits.shape[0])} vs {int(labels.shape[0])}"
            )
        if labels.numel() and bool(((labels != 0) & (labels != 1)).any().item()):
            raise ValueError(
                "binary_cross_entropy_with_logits_loss targets must be class labels 0 or 1"
            )
        return F.binary_cross_entropy_with_logits(
            logits,
            labels.to(device=logits.device, dtype=logits.dtype),
        )


@dataclass(frozen=True)
class LossCriterionContext:
    trainer_task: TrainerTask
    target_flat_dim: int | None = None
    num_logits: int | None = None
    multi_token_targets: bool = False


LossCriterionBuilder = Callable[[dict[str, Any], LossCriterionContext], nn.Module]


def _build_mse_loss_criterion(loss_d: dict[str, Any], context: LossCriterionContext) -> nn.Module:
    if context.trainer_task == "diffusion_noise":
        return nn.MSELoss()
    if context.target_flat_dim is None:
        raise HTTPException(status_code=500, detail="Internal: mse_loss criterion requires target_flat_dim.")
    return _mse_criterion_from_loss_data(loss_d, int(context.target_flat_dim))


def _build_cross_entropy_loss_criterion(loss_d: dict[str, Any], context: LossCriterionContext) -> nn.Module:
    ls = float(max(0.0, min(1.0, _scalar_float(loss_d.get("labelSmoothing"), 0.0))))
    if context.trainer_task == "token_classification":
        if context.multi_token_targets:
            return MultiSlotCrossEntropyLoss(label_smoothing=ls)
        return nn.CrossEntropyLoss(label_smoothing=ls)
    if context.num_logits is None:
        raise HTTPException(status_code=500, detail="Internal: cross_entropy_loss criterion requires num_logits.")
    return _cross_entropy_dense_criterion_from_loss_data(loss_d, int(context.num_logits))


def _build_binary_cross_entropy_with_logits_loss_criterion(
    _loss_d: dict[str, Any],
    context: LossCriterionContext,
) -> nn.Module:
    if context.trainer_task != "cross_entropy_dense":
        raise HTTPException(
            status_code=400,
            detail="binary_cross_entropy_with_logits_loss supports dense binary classification only.",
        )
    if context.num_logits != 1:
        raise HTTPException(
            status_code=400,
            detail=(
                "binary_cross_entropy_with_logits_loss requires exactly one model output logit; "
                f"got {context.num_logits}."
            ),
        )
    return BinaryCrossEntropyWithLogitsForClassLabels()


def _build_diffusion_mse_loss_criterion(_loss_d: dict[str, Any], _context: LossCriterionContext) -> nn.Module:
    return nn.MSELoss()


LOSS_CRITERION_BUILDERS: dict[str, LossCriterionBuilder] = {
}


def primary_loss_builder_node_types() -> frozenset[str]:
    # All specialized and registered providers are supported loss builders.
    from comfy_research.nodes.registry import loss_defs_criteria

    return frozenset(LOSS_CRITERION_BUILDERS) | frozenset(loss_defs_criteria())


def build_loss_criterion_for_node(loss_node: Node, context: LossCriterionContext) -> nn.Module:
    from comfy_research.nodes.registry import loss_defs_criteria

    key = str(getattr(loss_node.type, "value", loss_node.type))
    # NodeDef 通道 generated-first(L1;model/optimizer 同款)。
    builder = loss_defs_criteria().get(key) or LOSS_CRITERION_BUILDERS.get(key)
    if builder is None:
        raise HTTPException(status_code=400, detail=f"Unsupported primary loss type: {loss_node.type}")
    return builder(dict(loss_node.data or {}), context)
