"""Expand a built ``nn.Module`` when the graph shell sets ``loopCount`` ≥ 2."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import torch.nn as nn
from fastapi import HTTPException

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import Node, NodeKind


class ModelBlockLoop(nn.Module):
    """Run the same block ``n`` times with tied parameters (output shape must match input for the next step)."""

    def __init__(self, inner: nn.Module, n: int):
        super().__init__()
        if n < 2:
            raise ValueError("ModelBlockLoop requires n >= 2")
        self.inner = inner
        self.n = int(n)

    def forward(self, x: Any) -> Any:
        for _ in range(self.n):
            x = self.inner(x)
        return x


def _read_loop_count(data: dict[str, Any] | None) -> int | None:
    if not data:
        return None
    raw = data.get("loopCount", data.get("loop_count"))
    if isinstance(raw, bool):
        return None
    if isinstance(raw, list) and raw:
        raw = raw[0]
    if isinstance(raw, (int, float)):
        fv = float(raw)
        if not math.isfinite(fv):
            return None
        return int(fv)
    if isinstance(raw, str) and raw.strip():
        try:
            return int(float(raw))
        except ValueError:
            return None
    return None


def _read_loop_share_params(data: dict[str, Any] | None) -> bool:
    if not data:
        return False
    v = data.get("loopShareParams", data.get("loop_share_params"))
    return bool(v)


def resolve_loop_repeat_config(shell: Node, core: Node, nmap: dict[str, Node]) -> tuple[int, bool]:
    """Return ``(repeat_n, share_params)`` where ``repeat_n`` is 1 when looping is off."""
    for node in (shell, core):
        d = node.data or {}
        n = _read_loop_count(d)
        if n is not None and n >= 2:
            return n, _read_loop_share_params(d)
    pid = core.parentId
    if pid:
        parent = nmap.get(pid)
        if parent is not None and (parent.type == NodeKind.combined_model or has_capability(parent.type, "mlp_family")):
            d = parent.data or {}
            n = _read_loop_count(d)
            if n is not None and n >= 2:
                return n, _read_loop_share_params(d)
    return 1, False


def expand_model_loop_stacking(
    model: nn.Module,
    loop_n: int,
    share_params: bool,
    *,
    rebuild_block: Callable[[], nn.Module] | None = None,
) -> nn.Module:
    """Stack ``loop_n`` identical blocks (fresh parameters each time unless ``share_params``).

    Prefer ``rebuild_block`` for extra copies (atomic chains / graph-built modules): ``copy.deepcopy``
    on ``nn.Module`` graphs is fragile and can raise at runtime.
    """
    if loop_n < 2:
        return model
    if share_params:
        return ModelBlockLoop(model, loop_n)
    if rebuild_block is not None:
        extra = [rebuild_block() for _ in range(1, loop_n)]
        return nn.Sequential(model, *extra)
    copies: list[nn.Module] = [model]
    try:
        copies.extend(copy.deepcopy(model) for _ in range(1, loop_n))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not duplicate this model for looping (deep copy failed). "
                "Try enabling “Share parameters across loops”, or simplify the model. "
                f"Underlying error: {e!r}"
            ),
        ) from e
    return nn.Sequential(*copies)
