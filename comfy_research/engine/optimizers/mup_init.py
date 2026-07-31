"""Minimal μP-style init + Adam param-group LR multipliers for bundled LMs (opt-in via graph nodes)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from comfy_research.engine.models.attention_only_model import AttentionTokenPredictBundle
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel


def _mup_init_linear(m: nn.Linear) -> None:
    fan_in = int(m.weight.shape[1])
    std = float(fan_in) ** -0.5
    nn.init.normal_(m.weight, mean=0.0, std=std)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


def _mup_init_embedding(m: nn.Embedding) -> None:
    d = int(m.embedding_dim)
    nn.init.normal_(m.weight, mean=0.0, std=float(d) ** -0.5)


def apply_mup_init(model: nn.Module) -> None:
    """Variance scaling similar to width-stable LM init; safe no-op on unknown modules."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            _mup_init_linear(mod)
        elif isinstance(mod, nn.Embedding):
            _mup_init_embedding(mod)
        elif isinstance(mod, nn.LayerNorm):
            if mod.elementwise_affine and mod.weight is not None:
                nn.init.ones_(mod.weight)
            if mod.elementwise_affine and mod.bias is not None:
                nn.init.zeros_(mod.bias)


def _param_role(name: str) -> str:
    n = name.lower()
    if "lm_head" in n or "lm_heads" in n:
        return "output"
    if "embedding" in n and "pos_embed" not in n:
        return "embed"
    return "hidden"


def build_mup_adam_param_groups(
    model: nn.Module,
    *,
    base_lr: float,
    embed_mult: float,
    hidden_mult: float,
    output_mult: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Non-overlapping param groups by name heuristic (tied embedding/output shares one tensor)."""
    embed_p: set[int] = set()
    hidden_p: set[int] = set()
    output_p: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        pid = id(p)
        role = _param_role(name)
        if role == "embed":
            embed_p.add(pid)
        elif role == "output":
            output_p.add(pid)
        else:
            hidden_p.add(pid)
    # Tied weights: same tensor may be referenced as embedding and lm_head; prefer output role.
    embed_p -= output_p
    hidden_p -= embed_p | output_p

    def collect(ids: set[int]) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        seen: set[int] = set()
        for _n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in ids and pid not in seen:
                seen.add(pid)
                out.append(p)
        return out

    groups: list[dict[str, Any]] = []
    ge = collect(embed_p)
    gh = collect(hidden_p)
    go = collect(output_p)
    if ge:
        groups.append({"params": ge, "lr": float(base_lr) * float(embed_mult)})
    if gh:
        groups.append({"params": gh, "lr": float(base_lr) * float(hidden_mult)})
    if go:
        groups.append({"params": go, "lr": float(base_lr) * float(output_mult)})
    if not groups:
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable_count == 0:
            raise ValueError(
                "μP Adam: no trainable parameters found in model. "
                "All embedding, hidden, and output parameters have requires_grad=False."
            )
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": float(base_lr)})
    return groups


def supports_mup_bundle(model: nn.Module) -> bool:
    return isinstance(model, (TokenTransformerModel, MultiTokenTransformerModel, AttentionTokenPredictBundle))
