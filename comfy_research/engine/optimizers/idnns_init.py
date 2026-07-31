"""Released-IDNNs initialization for Information Bottleneck MLPs."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def apply_idnns_init(model: nn.Module, *, seed: int) -> None:
    """Initialize Linear layers in the released ``[in, out]`` RNG layout."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        std = 1.0 / math.sqrt(float(module.in_features))
        reference_layout = torch.empty(
            module.in_features,
            module.out_features,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        nn.init.trunc_normal_(
            reference_layout,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )
        with torch.no_grad():
            module.weight.copy_(reference_layout.transpose(0, 1))
            if module.bias is not None:
                module.bias.zero_()
