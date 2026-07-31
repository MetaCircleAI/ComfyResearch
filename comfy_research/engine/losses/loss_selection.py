from __future__ import annotations

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import NodeKind


def choose_loss_kind(model_type: NodeKind, dataset_type: NodeKind) -> NodeKind:
    if has_capability(model_type, "diffusion_loss_model"):
        return NodeKind.diffusion_mse_loss
    if has_capability(model_type, "token_model") or has_capability(model_type, "vision_model"):
        return NodeKind.cross_entropy_loss
    if has_capability(dataset_type, "memorization_dataset"):
        return NodeKind.cross_entropy_loss
    return NodeKind.mse_loss
