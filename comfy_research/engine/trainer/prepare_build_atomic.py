"""Atomic-chain model provider.

Receives a ModelBuildRequest (never PrepareState) and returns a
ModelBuildResult. The chain resolution is pure (no RNG) and mirrors what
the atomic dataset provider resolved for dims — its empty-chain 400 fires
there first, before any draw. Criterion ownership remains with build_criterion_stage.
"""
from typing import Any

import torch
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import (
    build_sequential_from_atomic_tip,
    build_sequential_from_flat_atomic_chain,
    collect_atomic_layer_chain_front_to_back,
)
from comfy_research.engine.trainer.model_helpers import _atomic_chain_dataset_dims
from comfy_research.engine.trainer.provider_types import ModelBuildRequest, ModelBuildResult
from comfy_research.engine.trainer.scalar import _scalar_int


def build_atomic_model(req: ModelBuildRequest) -> ModelBuildResult:
    combined_flat_chain = req.combined_flat_chain
    edges = req.edges
    model_node = req.model_node
    nmap = req.nmap
    seed = req.seed

    # --- chain + dims (no RNG) ---
    if combined_flat_chain is not None:
        chain = combined_flat_chain
    else:
        chain = collect_atomic_layer_chain_front_to_back(model_node, edges, nmap)
    if not chain:
        raise HTTPException(status_code=400, detail="Atomic model layer chain is empty.")
    md_atomic = _atomic_chain_dataset_dims(chain)
    stack_io = (
        _scalar_int(md_atomic.get("inputDim"), 1),
        _scalar_int(md_atomic.get("outputDim"), 1),
    )
    tip_data_atomic: dict[str, Any] = model_node.data or {}
    model_seed_atomic = (
        _scalar_int(tip_data_atomic.get("seed"), 0) if "seed" in tip_data_atomic else seed
    )

    # --- model section (identical in both original branches; moved as a unit
    # with its local re-seed per freeze rule 3.
    torch.manual_seed(model_seed_atomic)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed_atomic)
    if combined_flat_chain is not None:
        model = build_sequential_from_flat_atomic_chain(combined_flat_chain)
    else:
        model = build_sequential_from_atomic_tip(model_node, edges, nmap)

    depth = 1

    return ModelBuildResult(model=model, depth=depth, stack_io=stack_io)
