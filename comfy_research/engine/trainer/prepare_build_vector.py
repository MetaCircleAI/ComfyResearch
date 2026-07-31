"""Vector model provider.

Receives a ModelBuildRequest and returns a ModelBuildResult. The model-kind
dispatch validates against the resolved model output dimension. Teacher and
non-teacher paths differ only in the build output dimension. Criterion
ownership remains with build_criterion_stage.
"""
from typing import Any

import torch

from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node
from comfy_research.engine.trainer.dataset_helpers import _DATASET_MIXER_TYPES
from comfy_research.engine.trainer.provider_types import ModelBuildRequest, ModelBuildResult
from comfy_research.engine.trainer.scalar import _scalar_int, _scalar_str
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import NodeKind


def build_vector_model(req: ModelBuildRequest) -> ModelBuildResult:
    arrays = req.arrays
    ce_mem_a_slots = req.ce_mem_a_slots
    ds_train = req.ds_train
    model_node = req.model_node
    seed = req.seed
    want_kan_reg = req.want_kan_reg
    md: dict[str, Any] = model_node.data or {}
    is_kan = model_node.type == NodeKind.kan_model
    is_gated_mlp = model_node.type == NodeKind.gated_mlp_model
    is_moe_mlp = model_node.type == NodeKind.moe_mlp_model
    is_numeric_transformer = model_node.type == NodeKind.numeric_transformer_model
    is_numeric_hyena = model_node.type == NodeKind.numeric_hyena_model
    is_mpp_spatiotemporal = model_node.type == NodeKind.mpp_spatiotemporal_model
    is_afno_lite = model_node.type == NodeKind.afno_lite_spatiotemporal_model
    is_diffusion = has_capability(model_node.type, "diffusion_loss_model")
    depth = _scalar_int(md.get("depth"), 2)
    width = _scalar_int(md.get("width"), 5 if is_kan else 64)
    if is_kan:
        grid_k = _scalar_int(md.get("grid"), 3)
        spline_k = _scalar_int(md.get("k"), 3)
        base_fun_k = _scalar_str(md.get("baseFun"), "silu")
    else:
        act_name = _scalar_str(md.get("activation"), "silu" if (is_gated_mlp or is_moe_mlp) else "relu")
        num_experts = _scalar_int(md.get("numExperts"), 4) if is_moe_mlp else 1

    input_dim = arrays.input_dim
    output_dim = arrays.output_dim
    # Teacher builds against the validated teacher output_dim; non-teacher
    # against the resolved model_out_dim. Both paths preserve the established
    # behavior recorded by the golden tests.
    if ds_train.type == NodeKind.teacher_dataset:
        build_out_dim = output_dim
    else:
        if ds_train.type in _DATASET_MIXER_TYPES:
            build_out_dim = output_dim
        else:
            build_out_dim = _scalar_int(md.get("outputDim"), output_dim * max(1, ce_mem_a_slots))
    stack_io = (input_dim, build_out_dim)

    # --- model section (local re-seed moves with the builds; freeze rule 3) ---
    # Prefer explicit model ``seed``; if absent (older graphs), match prior behavior (init used dataset torch seed).
    model_seed = _scalar_int(md.get("seed"), 0) if "seed" in md else seed
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    if is_kan:
        model = build_model_for_node(
            model_node,
            ModelBuildContext(
                input_dim=input_dim,
                output_dim=build_out_dim,
                kan_fast_training=not want_kan_reg,
            ),
        )
    elif is_numeric_transformer:
        model = build_model_for_node(model_node, ModelBuildContext())
    elif is_numeric_hyena:
        model = build_model_for_node(model_node, ModelBuildContext())
    elif is_mpp_spatiotemporal:
        model = build_model_for_node(model_node, ModelBuildContext())
    elif is_afno_lite:
        model = build_model_for_node(model_node, ModelBuildContext())
    elif is_diffusion:
        model = build_model_for_node(model_node, ModelBuildContext(input_dim=input_dim))
    else:
        model = build_model_for_node(
            model_node,
            ModelBuildContext(input_dim=input_dim, output_dim=build_out_dim),
        )

    return ModelBuildResult(model=model, depth=depth, stack_io=stack_io)
