"""Teacher dataset sampler/model resolution helpers (extracted from trainer_run)."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import (
    SEQUENTIAL_MODEL_TYPES,
    build_sequential_from_flat_atomic_chain,
    collect_flat_atomic_chain_under_combined,
)
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_from_type
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_x_from_sampler_dict
from comfy_research.engine.trainer.graph import _incoming
from comfy_research.engine.trainer.model_helpers import _atomic_chain_dataset_dims
from comfy_research.engine.trainer.model_payloads import (
    apply_parameter_tensor_payloads_from_atomic_chain,
    apply_parameter_tensor_payloads_from_node,
)
from comfy_research.engine.trainer.scalar import _scalar_int, _scalar_str
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _resolve_teacher_sampler(
    edges: list[Edge],
    nmap: dict[str, Node],
    teacher_ds: Node,
    handle: str,
    fallback_sampler: Node | None,
) -> Node:
    sampler = _incoming(edges, nmap, teacher_ds.id, handle)
    if sampler is not None:
        if sampler.type != NodeKind.input_sampler:
            raise HTTPException(
                status_code=400,
                detail=f"teacher_dataset {handle} must be an input_sampler node.",
            )
        return sampler
    if fallback_sampler is not None:
        return fallback_sampler
    raise HTTPException(
        status_code=400,
        detail=f"teacher_dataset is missing {handle} (wire an input_sampler node).",
    )


def _sample_teacher_input_tensor(
    edges: list[Edge],
    nmap: dict[str, Node],
    teacher_ds: Node,
    handle: str,
    fallback_sampler: Node | None,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, Node]:
    sampler = _incoming(edges, nmap, teacher_ds.id, handle)
    if sampler is None and fallback_sampler is not None:
        sampler = fallback_sampler
    if sampler is None:
        # Backward compatibility for older graphs that wire teacher_dataset.input_distribution directly.
        legacy_rid = _incoming(edges, nmap, teacher_ds.id, "input_distribution")
        if legacy_rid is not None and legacy_rid.type == NodeKind.random_input_distribution:
            teacher_data: dict[str, Any] = teacher_ds.data or {}
            n = _scalar_int(
                teacher_data.get("testSize" if handle == "test_input" else "trainSize"),
                200 if handle == "test_input" else 800,
            )
            sampler_data = {"numSamples": max(1, n)}
            rid_data: dict[str, Any] = legacy_rid.data or {}
            x_np = sample_x_from_sampler_dict(sampler_data, rid_data, rng=rng)
            legacy_sampler = Node(id=f"{teacher_ds.id}::{handle}::legacy", type=NodeKind.input_sampler, data=sampler_data)
            return x_np, legacy_sampler
        sampler = _resolve_teacher_sampler(edges, nmap, teacher_ds, handle, fallback_sampler)
    rid = _incoming(edges, nmap, sampler.id, "distribution")
    if rid is None:
        raise HTTPException(
            status_code=400,
            detail=f"input_sampler wired to teacher_dataset {handle} is missing distribution input.",
        )
    if rid.type != NodeKind.random_input_distribution:
        raise HTTPException(
            status_code=400,
            detail="input_sampler distribution must come from a random_input_distribution node.",
        )
    sampler_data: dict[str, Any] = sampler.data or {}
    rid_data: dict[str, Any] = rid.data or {}
    x_np = sample_x_from_sampler_dict(sampler_data, rid_data, rng=rng)
    return x_np, sampler


def _build_teacher_mlp_eval(teacher_node: Node, fallback_seed: int) -> nn.Module:
    tmd: dict[str, Any] = teacher_node.data or {}
    input_dim = _scalar_int(tmd.get("inputDim"), 10)
    output_dim = _scalar_int(tmd.get("outputDim"), 1)
    depth = _scalar_int(tmd.get("depth"), 2)
    width = _scalar_int(tmd.get("width"), 64)
    act_name = _scalar_str(tmd.get("activation"), "relu")
    t_seed = _scalar_int(tmd.get("seed"), 0) if "seed" in tmd else fallback_seed
    torch.manual_seed(t_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t_seed)
    m = build_model_from_type(
        NodeKind.mlp_model,
        {
            "depth": depth,
            "width": width,
            "activation": act_name,
        },
        ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
    )
    apply_parameter_tensor_payloads_from_node(m, teacher_node, strict_unresolved=True)
    m.eval()
    return m


def _resolve_teacher_model_eval(
    teacher_src: Node | None,
    nodes: list[Node],
    edges: list[Edge],
    fallback_seed: int,
) -> tuple[nn.Module, int, int]:
    if teacher_src is None:
        raise HTTPException(
            status_code=400,
            detail="teacher_dataset is missing model (wire an mlp_model, gated_mlp_model, moe_mlp_model, or combined_model teacher).",
        )
    if has_capability(teacher_src.type, "mlp_family"):
        tmd: dict[str, Any] = teacher_src.data or {}
        input_dim = _scalar_int(tmd.get("inputDim"), 10)
        output_dim = _scalar_int(tmd.get("outputDim"), 1)
        if teacher_src.type == NodeKind.gated_mlp_model:
            depth = _scalar_int(tmd.get("depth"), 2)
            width = _scalar_int(tmd.get("width"), 64)
            act_name = _scalar_str(tmd.get("activation"), "silu")
            t_seed = _scalar_int(tmd.get("seed"), 0) if "seed" in tmd else fallback_seed
            torch.manual_seed(t_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(t_seed)
            teacher = build_model_from_type(
                NodeKind.gated_mlp_model,
                {
                    "depth": depth,
                    "width": width,
                    "activation": act_name,
                },
                ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
            )
            apply_parameter_tensor_payloads_from_node(teacher, teacher_src, strict_unresolved=True)
            teacher.eval()
            return teacher, input_dim, output_dim
        if teacher_src.type == NodeKind.moe_mlp_model:
            depth = _scalar_int(tmd.get("depth"), 2)
            width = _scalar_int(tmd.get("width"), 32)
            num_experts = _scalar_int(tmd.get("numExperts"), 4)
            act_name = _scalar_str(tmd.get("activation"), "silu")
            t_seed = _scalar_int(tmd.get("seed"), 0) if "seed" in tmd else fallback_seed
            torch.manual_seed(t_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(t_seed)
            teacher = build_model_from_type(
                NodeKind.moe_mlp_model,
                {
                    "depth": depth,
                    "width": width,
                    "numExperts": num_experts,
                    "activation": act_name,
                },
                ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
            )
            apply_parameter_tensor_payloads_from_node(teacher, teacher_src, strict_unresolved=True)
            teacher.eval()
            return teacher, input_dim, output_dim
        return _build_teacher_mlp_eval(teacher_src, fallback_seed), input_dim, output_dim
    if teacher_src.type == NodeKind.combined_model:
        chain = collect_flat_atomic_chain_under_combined(teacher_src, nodes, edges)
        if not chain or chain[-1].type not in SEQUENTIAL_MODEL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "teacher_dataset combined_model teacher requires a non-empty atomic layer chain "
                    "inside that wrapper (including inside nested combined models)."
                ),
            )
        tmd: dict[str, Any] = teacher_src.data or {}
        t_seed = _scalar_int(tmd.get("seed"), 0) if "seed" in tmd else fallback_seed
        torch.manual_seed(t_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(t_seed)
        teacher = build_sequential_from_flat_atomic_chain(chain)
        apply_parameter_tensor_payloads_from_atomic_chain(teacher, chain)
        teacher.eval()
        dims = _atomic_chain_dataset_dims(chain)
        return teacher, _scalar_int(dims.get("inputDim"), 1), _scalar_int(dims.get("outputDim"), 1)
    raise HTTPException(
        status_code=400,
        detail="teacher_dataset requires an mlp_model, gated_mlp_model, moe_mlp_model, or combined_model wired to model (teacher).",
    )

