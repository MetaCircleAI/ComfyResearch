"""Layer representation specs for the Observable factory (module input/output tensors)."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.analysis.activation_collect import _compute_activation_tensors
from comfy_research.engine.analysis.model_weight_materialize import (
    _synthetic_weight_node_id,
    build_model_for_weight_node,
)
from comfy_research.engine.runs.probe_input_selection import uses_token_probe_input
from comfy_research.engine.runs.trainer_run import _scalar_int
from comfy_research.schemas.graph import Edge, Node, NodeKind, Position


def _first_tensor(x: Any) -> torch.Tensor | None:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def _infer_float_input_dim(model: nn.Module, model_node: Node) -> int:
    md: dict[str, Any] = model_node.data or {}
    for key in ("inputDim", "input_dim", "flatDim", "flat_dim"):
        d = _scalar_int(md.get(key), 0)
        if d > 0:
            return d
    if hasattr(model, "body") and isinstance(getattr(model, "body", None), nn.Module):
        d = _infer_float_input_dim(model.body, model_node)
        if d > 0:
            return d
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            return int(mod.in_features)
        if isinstance(mod, nn.Conv1d):
            return int(mod.in_channels)
    return 1


def _probe_input_for_model(model: nn.Module, model_node: Node, meta: dict[str, Any]) -> torch.Tensor:
    md: dict[str, Any] = model_node.data or {}
    model_type = str(meta.get("model_type") or model_node.type or "")

    if uses_token_probe_input(model_type, has_tokens_per_input=hasattr(model, "tokens_per_input")):
        vocab = max(2, _scalar_int(md.get("vocabSize"), 59))
        tpi = max(1, _scalar_int(md.get("tokensPerInput"), 1))
        if hasattr(model, "context_length"):
            tpi = max(tpi, int(getattr(model, "context_length", tpi)))
        return torch.randint(0, vocab, (4, tpi), dtype=torch.long)

    if hasattr(model, "embedding") and isinstance(getattr(model, "embedding", None), nn.Embedding):
        tpi = max(1, _scalar_int(md.get("tokensPerInput"), 1))
        vocab = max(2, int(model.embedding.num_embeddings))
        return torch.randint(0, vocab, (4, tpi), dtype=torch.long)

    in_dim = _infer_float_input_dim(model, model_node)
    return torch.randn(4, in_dim)


def _forward_probe(model: nn.Module, x: torch.Tensor) -> None:
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        model(x.long())
    else:
        model(x)


def _collect_leaf_module_io(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Capture input/output tensors for every leaf module during one forward pass."""
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    for name, mod in model.named_modules():
        if not name:
            continue
        if len(list(mod.children())) > 0:
            continue
        key_in = f"{name}::input"
        key_out = f"{name}::output"

        def make_pre(k: str):
            def _pre(_m: nn.Module, args: Any) -> None:
                t = _first_tensor(args)
                if t is not None:
                    store[k] = t.detach()

            return _pre

        def make_fwd(k: str):
            def _fwd(_m: nn.Module, _args: Any, output: Any) -> None:
                t = _first_tensor(output)
                if t is not None:
                    store[k] = t.detach()

            return _fwd

        handles.append(mod.register_forward_pre_hook(make_pre(key_in)))
        handles.append(mod.register_forward_hook(make_fwd(key_out)))

    try:
        with torch.no_grad():
            _forward_probe(model, x)
    finally:
        for h in handles:
            h.remove()
    return store


def _legacy_mlp_tensors(model: nn.Module, model_node: Node, x: torch.Tensor) -> dict[str, torch.Tensor]:
    md: dict[str, Any] = model_node.data or {}
    depth = _scalar_int(md.get("depth"), 2)
    if hasattr(model, "body") and isinstance(getattr(model, "body", None), nn.Sequential):
        seq = model.body
        if hasattr(model, "embedding"):
            h_in = model.embedding(x.long()).reshape(x.shape[0], -1)
        else:
            h_in = x
        return _compute_activation_tensors(seq, h_in, depth)
    if isinstance(model, nn.Sequential):
        return _compute_activation_tensors(model, x, depth)
    return {}


def _legacy_mlp_entries(tensors: dict[str, torch.Tensor], depth: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for layer in range(1, depth + 1):
        in_id = "input" if layer == 1 else f"h{layer - 1}_postact"
        out_id = f"h{layer}_postact"
        for io, rep_id in (("input", in_id), ("output", out_id)):
            if rep_id in tensors:
                entries.append(
                    {
                        "layer_index": layer,
                        "io": io,
                        "representation_id": rep_id,
                        "module_name": rep_id,
                        "shape": _representation_spec_shape(tensors[rep_id]),
                        "label": f"Layer {layer} {io}",
                    }
                )
    if "output" in tensors:
        entries.append(
            {
                "layer_index": depth,
                "io": "network_output",
                "representation_id": "output",
                "module_name": "output",
                "shape": _representation_spec_shape(tensors["output"]),
                "label": "Network output",
            }
        )
    return entries


def _layer_index_from_module_name(name: str) -> int:
    nums = [int(x) for x in re.findall(r"(?:^|\.)(\d+)(?:\.|$)", name)]
    return nums[-1] + 1 if nums else 0


def _human_module_label(module_name: str, io: str) -> str:
    short = module_name.replace("body.", "", 1) if module_name.startswith("body.") else module_name
    cls_hint = short.rsplit(".", 1)[-1]
    if io == "input":
        return f"{short} input"
    if io == "output":
        return f"{short} output"
    return f"{short} {io}"


def _module_io_entries(tensors: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in sorted(tensors.keys()):
        if "::" not in key:
            continue
        name, io = key.rsplit("::", 1)
        if io not in ("input", "output"):
            continue
        t = tensors[key]
        shape = _representation_spec_shape(t)
        if t.numel() == 0:
            continue
        entries.append(
            {
                "layer_index": _layer_index_from_module_name(name),
                "io": io,
                "representation_id": key,
                "module_name": name,
                "shape": shape,
                "label": _human_module_label(name, io),
            }
        )
    return entries


# Axis 0 of representation specs is batch; actual size is fixed at training time.
BATCH_AXIS_PLACEHOLDER = -1


def _representation_spec_shape(t: torch.Tensor) -> list[int]:
    """List shape for UI/spec: ``[batch, …]`` with batch as a placeholder until training."""
    sh = [int(x) for x in t.shape]
    if not sh:
        return []
    return [BATCH_AXIS_PLACEHOLDER, *sh[1:]]


def run_model_representation_specs(
    nodes: list[Node],
    edges: list[Edge],
    model_node_id: str,
    *,
    include_upstream_chain: bool = True,
) -> dict[str, Any]:
    """List module input/output representation tensors (shapes only)."""
    mid = model_node_id.strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_node_id is required.")
    nmap = {n.id: n for n in nodes}
    model_node = nmap.get(mid)
    if model_node is None:
        raise HTTPException(status_code=404, detail="Model node not found.")

    synth_id = _synthetic_weight_node_id(mid)
    synth_nodes = list(nodes)
    synth_edges = list(edges)
    if synth_id not in nmap:
        synth_nodes.append(
            Node(
                id=synth_id,
                type=NodeKind.model_weight_tensors,
                data={},
                position=Position(x=float(model_node.position.x), y=float(model_node.position.y)),
            )
        )
        synth_edges.append(
            Edge(
                id=f"e-{mid}-{synth_id}",
                source=mid,
                target=synth_id,
                sourceHandle="model",
                targetHandle="model",
            )
        )

    model, meta = build_model_for_weight_node(
        synth_nodes,
        synth_edges,
        synth_id,
        include_upstream_chain=include_upstream_chain,
    )
    model.eval()
    x_probe = _probe_input_for_model(model, model_node, meta)
    try:
        tensors = _collect_leaf_module_io(model, x_probe)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Probe forward failed for {meta.get('model_type', model_node.type)}: {e}",
        ) from e
    if not tensors:
        legacy = _legacy_mlp_tensors(model, model_node, x_probe)
        depth = _scalar_int(model_node.data.get("depth") if model_node.data else None, 2)
        entries = _legacy_mlp_entries(legacy, depth)
    else:
        entries = _module_io_entries(tensors)
        depth = len({e["module_name"] for e in entries})
    if not entries:
        raise HTTPException(
            status_code=400,
            detail="Could not probe any module input/output tensors for this model.",
        )
    summary = (
        f"Listed {len(entries)} module input/output representation(s) from {meta.get('model_type', '?')}"
        + (" (checkpoint)" if meta.get("used_checkpoint") else "")
    )
    return {
        "entries": entries,
        "depth": depth,
        "summary": summary,
        "meta": meta,
    }


def collect_representation_tensors(
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
) -> dict[str, torch.Tensor]:
    """One forward pass (leaf hooks or legacy sequential) → all module I/O tensors by representation id."""
    with torch.no_grad():
        try:
            hooked = _collect_leaf_module_io(model, x)
            if hooked:
                return hooked
        except Exception:
            pass
        seq: nn.Sequential | None = None
        h_in = x
        if hasattr(model, "body") and isinstance(getattr(model, "body", None), nn.Sequential):
            seq = model.body
            if hasattr(model, "embedding"):
                h_in = model.embedding(x.long()).reshape(x.shape[0], -1)
        elif isinstance(model, nn.Sequential):
            seq = model
            h_in = x
        if seq is not None:
            return _compute_activation_tensors(seq, h_in, int(depth))
    return {}


def fetch_representation_numpy(
    model: nn.Module,
    x: torch.Tensor,
    depth: int,
    representation_id: str,
    *,
    representation_tensors: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    """Activation tensor for a representation id during training forward."""
    rep = (representation_id or "").strip()
    if not rep:
        raise HTTPException(status_code=400, detail="representation_id is required.")

    if representation_tensors is not None and rep in representation_tensors:
        return representation_tensors[rep].detach().cpu().float().numpy()

    if "::" in rep:
        module_name, io = rep.rsplit("::", 1)
        modules = dict(model.named_modules())
        target = modules.get(module_name)
        if target is None:
            raise HTTPException(status_code=400, detail=f"Module {module_name!r} not found on model.")
        captured: dict[str, torch.Tensor] = {}
        handle: Any | None = None

        if io == "input":

            def _pre(_m: nn.Module, args: Any) -> None:
                t = _first_tensor(args)
                if t is not None:
                    captured["t"] = t.detach()

            handle = target.register_forward_pre_hook(_pre)
        elif io == "output":

            def _fwd(_m: nn.Module, _args: Any, output: Any) -> None:
                t = _first_tensor(output)
                if t is not None:
                    captured["t"] = t.detach()

            handle = target.register_forward_hook(_fwd)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown representation io {io!r}.")

        try:
            with torch.no_grad():
                _forward_probe(model, x)
        finally:
            if handle is not None:
                handle.remove()
        if "t" not in captured:
            raise HTTPException(status_code=400, detail=f"No {io} tensor captured for {module_name!r}.")
        return captured["t"].cpu().float().numpy()

    # Legacy MLP representation ids (h2_postact, input, output, …)
    seq: nn.Sequential | None = None
    h_in: torch.Tensor = x
    if hasattr(model, "body") and isinstance(getattr(model, "body", None), nn.Sequential):
        seq = model.body
        if hasattr(model, "embedding"):
            h_in = model.embedding(x.long()).reshape(x.shape[0], -1)
        else:
            h_in = x
    elif isinstance(model, nn.Sequential):
        seq = model
        h_in = x
    if seq is not None:
        with torch.no_grad():
            tensors = _compute_activation_tensors(seq, h_in, int(depth))
        t = tensors.get(rep)
        if t is not None:
            return t.detach().cpu().float().numpy()

    raise HTTPException(status_code=400, detail=f"Representation {rep!r} not found on model.")
