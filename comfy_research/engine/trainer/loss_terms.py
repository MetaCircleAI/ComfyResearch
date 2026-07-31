"""Loss wiring + primary/regularization loss terms (extracted from trainer_run)."""

from typing import Any

import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.losses.loss_builders import TrainerTask
from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
from comfy_research.engine.trainer.graph import _incoming_all
from comfy_research.engine.trainer.model_helpers import _flatten_features_for_mse
from comfy_research.engine.trainer.scalar import _scalar_float
from comfy_research.engine.trainer.tensor_norms import _weight_l2_norm
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.schemas.graph import Edge, Node, NodeKind

_PRIMARY_TRAINER_LOSS_KINDS = frozenset(NodeKind(node_type) for node_type in registered_builder_node_types_for("loss"))
_WEIGHT_REG_TRAINER_LOSS_KINDS = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("trainer_weight_regularizer_loss")
)
_LOSS_SOCKET_AUX_KINDS = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("trainer_loss_socket_aux")
)


def _trainer_primary_loss_tensor(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    trainer_task: TrainerTask,
    criterion: nn.Module,
    loss_scale: float,
) -> torch.Tensor:
    """Scalar loss for the trainer's main objective (must stay consistent with ``prepare_trainer_run``)."""
    if trainer_task == "mse_regression":
        return criterion(_flatten_features_for_mse(pred), _flatten_features_for_mse(y)) * loss_scale
    if trainer_task in ("cross_entropy_dense", "token_classification", "vision_classification"):
        if trainer_task == "token_classification" and pred.dim() == 3 and y.dim() == 2:
            if int(pred.shape[0]) != int(y.shape[0]) or int(pred.shape[1]) != int(y.shape[1]):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Token LM shape mismatch: expected pred [batch, seq, vocab] and targets [batch, seq]; "
                        f"got pred {tuple(int(d) for d in pred.shape)} and targets {tuple(int(d) for d in y.shape)}."
                    ),
                )
            v = int(pred.shape[-1])
            return criterion(pred.reshape(-1, v), y.reshape(-1).long()) * loss_scale
        if trainer_task == "token_classification" and pred.dim() == 3 and y.dim() == 1:
            # Datasets such as in_context_associative_recall and circle_random_walk emit one label per row
            # (next / query token) while the model returns logits at every position — supervise the last slot.
            if int(pred.shape[0]) != int(y.shape[0]):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Token LM shape mismatch: expected pred [batch, seq, vocab] and targets [batch] "
                        "(per-sequence class index); "
                        f"got pred {tuple(int(d) for d in pred.shape)} and targets {tuple(int(d) for d in y.shape)}."
                    ),
                )
            return criterion(pred[:, -1, :], y.long()) * loss_scale
        return criterion(pred, y) * loss_scale
    if trainer_task == "diffusion_noise":
        raise AssertionError("diffusion_noise loss is computed in the training loop, not via _trainer_primary_loss_tensor")
    raise AssertionError(f"unexpected trainer_task: {trainer_task!r}")


def _trainer_loss_wiring(
    edges: list[Edge], nodes: dict[str, Node], trainer_id: str
) -> tuple[Node | None, list[Node], list[Node]]:
    """Primary task loss (at most one), optional L1/L2 reg, optional L2 projection on the loss socket."""
    all_loss = _incoming_all(edges, nodes, trainer_id, "loss")
    primary = [n for n in all_loss if n.type in _PRIMARY_TRAINER_LOSS_KINDS]
    weight_regs = [n for n in all_loss if n.type in _WEIGHT_REG_TRAINER_LOSS_KINDS]
    l2_projections = [n for n in all_loss if n.type == NodeKind.l2_projection]
    unknown = [
        n
        for n in all_loss
        if n.type not in _PRIMARY_TRAINER_LOSS_KINDS and n.type not in _LOSS_SOCKET_AUX_KINDS
    ]
    if unknown:
        kinds = ", ".join(str(n.type) for n in unknown)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Trainer loss socket got unsupported node type(s): {kinds}. "
                "Use mse_loss, cross_entropy_loss, binary_cross_entropy_with_logits_loss, "
                "or diffusion_mse_loss as the primary loss; "
                "l1_reg, l2_reg, and l2_projection may be wired in addition."
            ),
        )
    if len(primary) > 1:
        kinds = ", ".join(str(n.type) for n in primary)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple primary loss nodes target trainer {trainer_id!r} loss socket ({kinds}). "
                "Wire exactly one mse_loss, cross_entropy_loss, "
                "binary_cross_entropy_with_logits_loss, or diffusion_mse_loss."
            ),
        )
    if weight_regs and not primary:
        raise HTTPException(
            status_code=400,
            detail=(
                "L1/L2 reg nodes on the trainer loss socket require a primary loss "
                "(mse_loss, cross_entropy_loss, binary_cross_entropy_with_logits_loss, "
                "or diffusion_mse_loss)."
            ),
        )
    return (primary[0] if primary else None), weight_regs, l2_projections


def _l2_projection_target_norm(on: Node) -> float:
    od = on.data if isinstance(on.data, dict) else {}
    raw = od.get("targetNorm")
    if isinstance(raw, list):
        raw = raw[0] if raw else 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _apply_l2_weight_projection(model: nn.Module, l2_projection_nodes: list[Node] | None) -> None:
    """Scale trainable weights so global L2 norm equals the first wired projection node's target."""
    if not l2_projection_nodes:
        return
    target = _l2_projection_target_norm(l2_projection_nodes[0])
    if target <= 0.0:
        return
    current = _weight_l2_norm(model)
    if current <= 1e-12:
        return
    scale = target / current
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.mul_(scale)


_KAN_REG_METRICS = frozenset(
    {
        "edge_forward_spline_n",
        "edge_forward_spline_u",
        "edge_forward_sum",
        "edge_backward",
        "node_backward",
    }
)


def _kan_reg_loss_term(model: nn.Module, on: Node) -> torch.Tensor:
    """Scalar tensor ``lamb * model.get_reg(...)`` for one ``kan_reg`` observable (pykan ``MultKAN``)."""
    dd: dict[str, Any] = on.data or {}
    metric = str(dd.get("regMetric") or "edge_forward_spline_n").strip()
    if metric not in _KAN_REG_METRICS:
        raise HTTPException(status_code=400, detail=f"Unknown KAN reg metric: {metric!r}.")
    lamb = _scalar_float(dd.get("lamb"), 0.01)
    l1 = _scalar_float(dd.get("lambL1"), 1.0)
    ent = _scalar_float(dd.get("lambEntropy"), 2.0)
    coef = _scalar_float(dd.get("lambCoef"), 0.0)
    cdiff = _scalar_float(dd.get("lambCoefDiff"), 0.0)
    get_reg = getattr(model, "get_reg", None)
    if get_reg is None:
        raise HTTPException(
            status_code=500,
            detail="Internal: KAN regularization requires a pykan KAN model with get_reg.",
        )
    return lamb * get_reg(metric, l1, ent, coef, cdiff)


def _weight_reg_loss_term(model: nn.Module, on: Node) -> torch.Tensor:
    """Scalar ``lossScale * sum(|w|)`` (L1) or ``lossScale * sum(w^2)`` (L2) over trainable params."""
    dd: dict[str, Any] = on.data or {}
    scale = _scalar_float(dd.get("lossScale"), 1.0)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        ref = next(model.parameters(), None)
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        return torch.zeros((), device=device, dtype=dtype)
    total = torch.zeros((), device=params[0].device, dtype=params[0].dtype)
    if on.type == NodeKind.l1_reg:
        for p in params:
            total = total + p.abs().sum()
    elif on.type == NodeKind.l2_reg:
        for p in params:
            total = total + (p * p).sum()
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Internal: weight reg loss term expected l1_reg or l2_reg, got {on.type!r}.",
        )
    return total * scale


def _weight_reg_loss_additions(
    model: nn.Module, weight_reg_loss_nodes: list[Node] | None
) -> torch.Tensor | None:
    if not weight_reg_loss_nodes:
        return None
    terms = [_weight_reg_loss_term(model, on) for on in weight_reg_loss_nodes]
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _extra_loss_additions(
    model: nn.Module,
    *,
    kan_regs: list[Node] | None = None,
    weight_reg_loss_nodes: list[Node] | None = None,
) -> torch.Tensor | None:
    terms: list[torch.Tensor] = []
    for on in kan_regs or []:
        terms.append(_kan_reg_loss_term(model, on))
    for on in weight_reg_loss_nodes or []:
        terms.append(_weight_reg_loss_term(model, on))
    if not terms:
        return None
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out
