from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.optimizers.matrix_preconditioner_optimizers import SOAP, Shampoo
from comfy_research.engine.optimizers.muon_optimizer import build_muon_with_aux_adam
from comfy_research.engine.optimizers.mup_init import build_mup_adam_param_groups, supports_mup_bundle
from comfy_research.engine.optimizers.signsgd_optimizer import SignSGD
from comfy_research.schemas.graph import Node


@dataclass(frozen=True)
class OptimizerBuildConfig:
    lr: float
    beta1: float
    beta2: float
    eps: float
    momentum: float
    weight_decay: float
    muon_ns_steps: int
    precondition_frequency: int = 10
    max_preconditioner_dim: int = 1024
    use_mup_lr_schedule: bool = False
    mup_embed_lr_mult: float = 1.0
    mup_hidden_lr_mult: float = 1.0
    mup_output_lr_mult: float = 1.0


OptimizerBuilder = Callable[[nn.Module, OptimizerBuildConfig], torch.optim.Optimizer]


def _trainable_parameters(model: nn.Module):
    """Return only parameters with requires_grad=True.

    Frozen layers (requires_grad_(False)) are excluded so optimizers
    never receive parameters that should not be updated.
    """
    return (p for p in model.parameters() if p.requires_grad)


def _build_adam_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    if config.use_mup_lr_schedule:
        if not supports_mup_bundle(model):
            raise HTTPException(
                status_code=400,
                detail=(
                    "mup_lr_schedule is only supported for models that expose μP-style Adam param groups "
                    "(token transformer, multi-token transformer, or attention-only token bundle). "
                    "Wire an lr_schedule node to the time-dependent socket if needed, or switch to a supported model."
                ),
            )
        pgroups = build_mup_adam_param_groups(
            model,
            base_lr=float(config.lr),
            embed_mult=float(config.mup_embed_lr_mult),
            hidden_mult=float(config.mup_hidden_lr_mult),
            output_mult=float(config.mup_output_lr_mult),
            weight_decay=float(config.weight_decay),
        )
        return torch.optim.Adam(
            pgroups,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=float(config.weight_decay),
        )
    return torch.optim.Adam(
        _trainable_parameters(model),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _build_sgd_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        _trainable_parameters(model),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def _build_muon_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    muon_momentum = float(config.momentum) if float(config.momentum) > 0 else 0.95
    return build_muon_with_aux_adam(
        model,
        lr=float(config.lr),
        momentum=muon_momentum,
        weight_decay=config.weight_decay,
        ns_steps=int(config.muon_ns_steps),
    )


def _build_adamw_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    # Match the legacy elif ordering: the μP-LR-schedule branch preceded adamw, so
    # adamw + mup_lr_schedule resolves to the μP Adam param-group path, not AdamW.
    if config.use_mup_lr_schedule:
        return _build_adam_optimizer(model, config)
    return torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _build_signsgd_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    return SignSGD(_trainable_parameters(model), lr=float(config.lr), weight_decay=float(config.weight_decay))


def _build_shampoo_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    return Shampoo(
        _trainable_parameters(model),
        lr=float(config.lr),
        momentum=float(config.momentum),
        eps=float(config.eps),
        weight_decay=float(config.weight_decay),
        precondition_frequency=int(config.precondition_frequency),
        max_preconditioner_dim=int(config.max_preconditioner_dim),
    )


def _build_soap_optimizer(model: nn.Module, config: OptimizerBuildConfig) -> torch.optim.Optimizer:
    return SOAP(
        _trainable_parameters(model),
        lr=float(config.lr),
        betas=(float(config.beta1), float(config.beta2)),
        eps=float(config.eps),
        weight_decay=float(config.weight_decay),
        precondition_frequency=int(config.precondition_frequency),
        max_preconditioner_dim=int(config.max_preconditioner_dim),
    )


OPTIMIZER_BUILDERS: dict[str, OptimizerBuilder] = {
}


def optimizer_builder_node_types() -> frozenset[str]:
    # All specialized and registered providers are supported optimizer builders.
    from comfy_research.nodes.registry import optimizer_defs_builders

    return frozenset(OPTIMIZER_BUILDERS) | frozenset(optimizer_defs_builders())


def build_optimizer_for_node(
    opt_node: Node,
    model: nn.Module,
    config: OptimizerBuildConfig,
) -> torch.optim.Optimizer:
    from comfy_research.nodes.registry import optimizer_defs_builders

    key = str(getattr(opt_node.type, "value", opt_node.type))
    # NodeDef 通道采用 generated-first，与 build_model_from_type 一致。
    builder = optimizer_defs_builders().get(key) or OPTIMIZER_BUILDERS.get(key)
    if builder is None:
        raise HTTPException(status_code=400, detail=f"Unsupported optimizer type: {opt_node.type}")
    return builder(model, config)
