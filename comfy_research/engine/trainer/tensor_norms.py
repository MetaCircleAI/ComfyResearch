"""Weight/gradient norm helpers for the trainer runtime (extracted from trainer_run)."""

import math

import torch
import torch.nn as nn


def _weight_l2_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().float().pow(2).sum().item())
    return s**0.5


def _weight_l1_norm(module: nn.Module) -> float:
    s = 0.0
    for p in module.parameters():
        s += float(p.detach().float().abs().sum().item())
    return s


def _gradient_l2_norm_global(module: nn.Module, *, normalize: bool = False) -> float:
    total = 0.0
    count = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += float(g.pow(2).sum().item())
        count += int(g.numel())
    n = math.sqrt(total) if total > 0.0 else 0.0
    if normalize and count > 0:
        n /= math.sqrt(float(count))
    return n


def _gradient_l2_norm_per_top_level(module: nn.Module, *, normalize: bool = False) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".", 1)[0]
        g = p.grad.detach().float()
        sums[top] = sums.get(top, 0.0) + float(g.pow(2).sum().item())
        counts[top] = counts.get(top, 0) + int(p.numel())
    out = {k: math.sqrt(v) for k, v in sums.items()}
    if normalize:
        out = {
            k: (out[k] / math.sqrt(float(counts[k])) if counts.get(k, 0) > 0 else 0.0) for k in out
        }
    return out


def _weight_l2_norm_per_top_level(module: nn.Module) -> dict[str, float]:
    sums: dict[str, float] = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            top = name.split(".", 1)[0]
            t = p.detach().float()
            sums[top] = sums.get(top, 0.0) + float(t.pow(2).sum().item())
    return {k: math.sqrt(v) for k, v in sums.items()}


MAX_OBSERVABLE_L2_TENSOR_SERIES = 64


def _parameter_l2_norms_named(
    module: nn.Module,
    *,
    use_grad: bool,
    max_params: int = MAX_OBSERVABLE_L2_TENSOR_SERIES,
    normalize_grad: bool = False,
) -> dict[str, float]:
    """Sorted full parameter names (truncated to ``max_params``) -> L2 norm of weights or gradients."""
    rows: list[tuple[str, float]] = []
    for name, p in module.named_parameters():
        if use_grad:
            if p.grad is None:
                continue
            t = p.grad.detach().float()
        else:
            t = p.detach().float()
        v = float(torch.sqrt(t.pow(2).sum()).item())
        if normalize_grad and use_grad:
            ne = int(p.numel())
            if ne > 0:
                v /= math.sqrt(float(ne))
        rows.append((name, v))
    rows.sort(key=lambda kv: kv[0])
    return {name: v for name, v in rows[:max(1, max_params)]}
