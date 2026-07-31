"""Checkpoint loading helpers with backward compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from comfy_research.lmech.lpd.model import CurveDETR


def is_legacy_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    """True for pre-breakpoint DETR box checkpoints."""
    keys = list(state_dict.keys())
    has_box = any("box_head" in k for k in keys)
    has_bp = any("bp_head" in k for k in keys)
    return has_box and not has_bp


def is_legacy_checkpoint_file(path: str | Path) -> bool:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return is_legacy_checkpoint(ckpt.get("state_dict", {}))


def _adapt_first_conv_weight(
    checkpoint_weight: torch.Tensor,
    target_shape: torch.Size,
) -> Optional[torch.Tensor]:
    if checkpoint_weight.shape == target_shape:
        return checkpoint_weight
    if (
        checkpoint_weight.ndim == 3
        and len(target_shape) == 3
        and checkpoint_weight.shape[0] == target_shape[0]
        and checkpoint_weight.shape[2:] == target_shape[2:]
        and checkpoint_weight.shape[1] == 1
        and target_shape[1] == 3
    ):
        # Old 1-channel (y only) → new 3-channel (y, dy, d²y).
        return checkpoint_weight.repeat(1, 3, 1) / 3.0
    return None


def load_state_dict_compat(
    model: CurveDETR,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], bool]:
    """
    Load matching weights; skip or adapt incompatible tensors.

    Returns (skip_messages, loaded_any_bp_head).
    """
    model_sd = model.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []

    for key, value in state_dict.items():
        if key not in model_sd:
            skipped.append(f"unused in model: {key}")
            continue

        target = model_sd[key]
        if value.shape == target.shape:
            to_load[key] = value
            continue

        adapted = None
        if key == "backbone.net.0.weight":
            adapted = _adapt_first_conv_weight(value, target.shape)

        if adapted is not None:
            to_load[key] = adapted
        else:
            skipped.append(
                f"skipped {key}: checkpoint {tuple(value.shape)} "
                f"vs model {tuple(target.shape)}"
            )

    model.load_state_dict(to_load, strict=False)
    loaded_bp = any(k.startswith("bp_head") for k in to_load)
    return skipped, loaded_bp
