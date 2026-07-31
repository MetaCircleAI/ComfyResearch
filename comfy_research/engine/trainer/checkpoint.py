"""Checkpoint pack/apply helpers for the trainer runtime (extracted from trainer_run)."""

import base64
import io

import torch
import torch.nn as nn
from fastapi import HTTPException


def _pack_checkpoint_b64(model: nn.Module, optimizer: torch.optim.Optimizer) -> str:
    buf = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        buf,
    )
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _apply_checkpoint_b64(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    b64: str,
    *,
    map_location: str | torch.device | None = None,
) -> None:
    loc: str | torch.device = map_location if map_location is not None else "cpu"
    raw = base64.standard_b64decode(b64.encode("ascii"))
    ckpt = torch.load(io.BytesIO(raw), map_location=loc, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])


def load_model_weights_from_checkpoint_b64(model: nn.Module, b64: str) -> None:
    """Load only ``model`` weights from trainer-style ``torch.save`` bytes (same format as streaming train)."""
    if not isinstance(b64, str) or not b64.strip():
        raise HTTPException(status_code=400, detail="Checkpoint payload is empty.")
    try:
        raw = base64.standard_b64decode(b64.encode("ascii"))
        ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode checkpoint: {e}") from e
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise HTTPException(status_code=400, detail="Checkpoint is missing a 'model' state_dict.")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Checkpoint weights do not match this model architecture: {e}",
        ) from e
