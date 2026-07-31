"""Vision model provider.

Receives a ModelBuildRequest (never PrepareState) and returns a
ModelBuildResult; the model section preserves the original branch behavior —
local re-seed + build_model_for_node against the DatasetArrays dims/extras.
Criterion ownership remains with build_criterion_stage; this branch never
wrote it (result.stack_io = None).
"""
from typing import Any

import torch

from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node
from comfy_research.engine.trainer.provider_types import ModelBuildRequest, ModelBuildResult
from comfy_research.engine.trainer.scalar import _scalar_int


def build_vision_model(req: ModelBuildRequest) -> ModelBuildResult:
    arrays = req.arrays
    model_node = req.model_node

    # --- model section (local re-seed moves with the build; freeze rule 3) ---
    md_v: dict[str, Any] = model_node.data or {}
    model_seed_v = _scalar_int(md_v.get("seed"), 0)
    torch.manual_seed(model_seed_v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed_v)
    model = build_model_for_node(
        model_node,
        ModelBuildContext(
            input_channels=arrays.input_dim,
            image_size=arrays.extras["image_size"],
            num_classes=arrays.output_dim,
        ),
    )
    depth = 1

    return ModelBuildResult(model=model, depth=depth, stack_io=None)
