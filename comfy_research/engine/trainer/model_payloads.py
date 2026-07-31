"""UI-edited parameter tensor payload application (extracted from trainer_run)."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.schemas.graph import Node


def apply_parameter_tensor_payloads_from_node(
    model: nn.Module, model_node: Node, *, strict_unresolved: bool = False
) -> int:
    """Apply edited frontend parameter tensors stored on ``model_node.data`` onto ``model``."""
    md: dict[str, Any] = model_node.data or {}
    raw = md.get("parameterTensorPayloads")
    if not isinstance(raw, dict) or not raw:
        return 0

    params = dict(model.named_parameters())

    def _resolve_param(name: str, shape: list[int]) -> nn.Parameter | None:
        wanted_shape = tuple(shape)

        def _shape_ok(p: nn.Parameter) -> bool:
            return tuple(p.shape) == wanted_shape

        direct = params.get(name)
        if direct is not None:
            return direct if _shape_ok(direct) else None
        suffix = f".{name}"
        candidates = [(k, v) for k, v in params.items() if k.endswith(suffix)]
        shape_matches = [p for _, p in candidates if _shape_ok(p)]
        if len(shape_matches) == 1:
            return shape_matches[0]
        # Accept edited names with an optional leading module prefix or index
        # (e.g. "linear.weight" / "0.weight") by trying the leaf segment.
        if "." in name:
            leaf = name.split(".")[-1]
            direct_leaf = params.get(leaf)
            if direct_leaf is not None and _shape_ok(direct_leaf):
                return direct_leaf
            leaf_suffix = f".{leaf}"
            leaf_matches = [p for k, p in params.items() if k.endswith(leaf_suffix) and _shape_ok(p)]
            if len(leaf_matches) == 1:
                return leaf_matches[0]
        return None

    applied = 0
    unresolved: list[str] = []
    with torch.no_grad():
        for name, payload in raw.items():
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=400,
                    detail=f"Edited parameter payload for {name!r} must be an object with shape/values.",
                )
            shape_raw = payload.get("shape")
            values_raw = payload.get("values")
            if not isinstance(shape_raw, list) or not all(isinstance(x, int) for x in shape_raw):
                raise HTTPException(
                    status_code=400,
                    detail=f"Edited parameter {name!r} has invalid shape payload.",
                )
            if not isinstance(values_raw, list):
                raise HTTPException(
                    status_code=400,
                    detail=f"Edited parameter {name!r} has invalid values payload.",
                )
            shape = [int(x) for x in shape_raw]
            param = _resolve_param(name, shape)
            if param is None:
                unresolved.append(name)
                continue
            expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if len(values_raw) != expected:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Edited parameter {name!r} expects {expected} values for shape {shape}, "
                        f"got {len(values_raw)}."
                    ),
                )
            if tuple(shape) != tuple(param.shape):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Edited parameter {name!r} shape {shape} does not match model shape "
                        f"{list(param.shape)}."
                    ),
                )
            try:
                flat = np.asarray(values_raw, dtype=np.float64)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Edited parameter {name!r} contains non-numeric values: {e}",
                ) from e
            if not np.isfinite(flat).all():
                raise HTTPException(
                    status_code=400,
                    detail=f"Edited parameter {name!r} contains non-finite values.",
                )
            tensor = torch.as_tensor(flat, dtype=param.dtype, device=param.device).reshape(param.shape)
            param.copy_(tensor)
            applied += 1
    if strict_unresolved and unresolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "Edited parameter tensor(s) could not be matched to this model: "
                + ", ".join(sorted(unresolved))
                + ". Refresh parameters and save again."
            ),
        )
    return applied


def apply_parameter_tensor_payloads_from_atomic_chain(model: nn.Module, chain: list[Node]) -> int:
    """Apply edited parameter payloads for each atomic chain node to its matching sequential module."""
    if not chain:
        return 0
    if not isinstance(model, nn.Sequential):
        raise HTTPException(
            status_code=500,
            detail="Internal: expected nn.Sequential when applying edited atomic-chain parameters.",
        )
    modules = list(model.children())
    if len(modules) < len(chain):
        raise HTTPException(
            status_code=500,
            detail=(
                "Internal: atomic model chain is longer than built sequential modules "
                f"({len(chain)} > {len(modules)})."
            ),
        )

    applied = 0
    for idx, chain_node in enumerate(chain):
        md: dict[str, Any] = chain_node.data or {}
        raw = md.get("parameterTensorPayloads")
        if not isinstance(raw, dict) or not raw:
            continue
        module = modules[idx]
        local_params = dict(module.named_parameters())
        for local_name in list(raw.keys()):
            if local_name not in local_params:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Edited parameter {local_name!r} does not exist for atomic layer "
                        f"{chain_node.id} ({chain_node.type.value})."
                    ),
                )
        pseudo_node = Node(
            id=chain_node.id,
            type=chain_node.type,
            position=chain_node.position,
            data={"parameterTensorPayloads": raw},
            parentId=chain_node.parentId,
            extent=chain_node.extent,
            hidden=chain_node.hidden,
            style=chain_node.style,
        )
        applied += apply_parameter_tensor_payloads_from_node(module, pseudo_node)
    return applied
