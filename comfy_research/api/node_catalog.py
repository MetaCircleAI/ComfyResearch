from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fastapi import APIRouter

from comfy_research.generated.node_manifest import load_node_manifest

router = APIRouter(prefix="/api", tags=["nodes"])

CATEGORY_ORDER = (
    "dataset",
    "optimizer",
    "model",
    "loss",
    "observables",
    "training",
    "checkpoint",
    "visualization",
    "language",
    "analysis",
)

# Registered node types reached through dedicated flows instead of the sidebar.
_HIDDEN_CATALOG_NODE_TYPES: frozenset[str] = frozenset({"combined_model", "observable_user"})

# Curated sidebar order. Types absent here sort stably to the end of their category.
_MAIN_CATALOG_NODE_ORDER: tuple[str, ...] = (
    "observable_weight_l2", "observable_weight_l1", "observable_capacity", "observable_hessian_eigenvalues",
    "observable_relu_nonlinear_count", "observable_accuracy", "observable_gradient_norm", "observable_train_test_gap",
    "observable_activation_stats", "observable_embedding_effective_rank", "observable_embedding_feature_drift", "observable_sink_attention_mass",
    "observable_attention_entropy_mean", "observable_attention_max_weight_mean", "observable_attention_head_sink_max", "observable_attention_position_bias_ratio",
    "observable_activation_norm_mean", "observable_activation_outlier_ratio", "observable_embedding_evolution", "observable_embedding_trajectory",
    "observable_last_layer_weight_norm",
    "linear_dataset", "random_noise_dataset", "memorization_a_dataset", "memorization_b_dataset",
    "symbolic_func_dataset", "token_prediction_dataset", "circle_random_walk_dataset", "circular_motion_dataset",
    "kepler_2d_dataset", "unigram_dataset", "bigram_low_rank_dataset", "random_input_distribution",
    "input_sampler", "teacher_dataset", "in_context_associative_recall_dataset", "uniform_linear_motion_dataset",
    "modular_addition_dataset", "dataset_mixer", "dataset_mixer_b", "pcfg_dataset",
    "dyck_dataset", "ngram_language_dataset", "formal_language_suite_dataset", "scan_dataset",
    "cogs_dataset", "listops_dataset", "tinystories_dataset", "phi1_style_dataset",
    "biography_lm_dataset", "relation_tuple_dataset", "synthetic_playground_dataset", "multi_hop_fact_chain_dataset",
    "mnist_dataset", "cifar10_dataset", "gaussian_blob_dataset", "shape_world_dataset", "hole_counting_dataset",
    "diffusion_pde_dataset", "reaction_diffusion_dataset", "advection_dataset", "crl_env_config", "adam_optimizer", "adamw_optimizer",
    "sgd_optimizer", "signsgd_optimizer", "muon_optimizer", "shampoo_optimizer",
    "soap_optimizer", "lr_schedule", "cyclic_lr_schedule", "cyclic_batch_schedule", "mup_lr_schedule", "mlp_model",
    "resnet_model", "keskar_c1_c2_cnn_model", "vgg11_cifar_model", "vit_model", "crl_residual_mlp", "gated_mlp_model",
    "moe_mlp_model", "mlp_token_model", "gated_mlp_token_model", "moe_mlp_token_model",
    "kan_model", "linear_layer", "activation_layer", "layer_norm_layer",
    "rms_norm_layer", "embedding_layer", "unembedding_layer", "absolute_pos_embed_layer",
    "rotary_embed_layer", "local_mixing_layer", "afno_patch_embed_layer", "afno_spectral_mixer_layer",
    "afno_encoder_block_layer", "afno_patch_decode_layer", "residual_ln_model", "attention_only_model",
    "linear_attention_model", "diagonal_ssm_token_model", "rwkv_time_mix_token_model", "hyena_like_conv_model",
    "slot_attention_token_model", "diffusion_score_model", "pairwise_rbf_layer",
    "equivariant_message_layer", "energy_readout_layer", "relative_pose_encoder_layer", "distance_contact_layer",
    "numeric_transformer_model", "numeric_hyena_model", "mpp_spatiotemporal_model", "afno_lite_spatiotemporal_model",
    "transformer_token_model", "transformer_multi_token_model", "tensor_splitter", "reshape",
    "flatten", "einsum", "softmax", "causal_mask",
    "tensor_add", "tensor_stack", "tensor_concat", "tensor_constant",
    "tensor_linspace", "elementwise_transform", "fake_tensor", "mse_loss",
    "l1_reg", "l2_reg", "l2_projection", "diffusion_mse_loss",
    "kan_reg", "cross_entropy_loss", "trainer", "crl_trainer",
    "model_checkpoint", "training_visualization", "image_dataset_displayer",
    "protein_structure_displayer", "protein_structure_comparison_viz", "interatomic_eval_viz", "docking_pose_viz",
    "agent_trace_viz", "curve_annotator", "observable_viz", "tensor_viz_general",
    "tensor_viz_0d", "tensor_viz_1d", "tensor_viz_2d", "tensor_viz_scatter",
    "visualize_kan", "comment", "hypothesis", "url_node",
    "activation", "model_weight_tensors", "effective_rank", "series_endpoint_gap",
    "smoothing_curve", "derivative_curve", "prediction",
    "mup_initialization", "tensor_selector", "dimension_permutator", "tensor_slicing",
    "pca", "svd", "statistics", "statistics2",
    "sweep_data_table", "table_viz", "curve_series_table", "curve_series_viz", "parametric_path_sampler", "regressor", "shape_checker",
    "tensor_reader", "basic_calculator",
)
_CATALOG_ORDER_INDEX: dict[str, int] = {t: i for i, t in enumerate(_MAIN_CATALOG_NODE_ORDER)}


def _catalog_children_for_linear_datasets() -> list[dict[str, str]]:
    from comfy_research.api.user_linear_datasets import catalog_children_for_linear_datasets

    return catalog_children_for_linear_datasets()


def _catalog_children_for_symbolic_func_datasets() -> list[dict[str, str]]:
    from comfy_research.api.user_symbolic_func_datasets import catalog_children_for_symbolic_func_datasets

    return catalog_children_for_symbolic_func_datasets()


def _catalog_children_for_observables() -> list[dict[str, str]]:
    from comfy_research.api.user_observables import catalog_children_for_observables

    return catalog_children_for_observables()


DYNAMIC_CHILDREN: dict[str, tuple[Callable[[], list[dict[str, str]]], ...]] = {
    "dataset": (_catalog_children_for_linear_datasets, _catalog_children_for_symbolic_func_datasets),
    "observables": (_catalog_children_for_observables,),
}


@router.get("/node-definitions")
def list_node_definitions() -> dict:
    """Frontend-authored node registry manifest for backend consumers."""
    return {"version": 1, "nodes": load_node_manifest()}


@lru_cache(maxsize=1)
def _static_catalog_categories() -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, str]]] = {category: [] for category in CATEGORY_ORDER}
    for entry in load_node_manifest():
        category = str(entry.get("category", ""))
        if category not in by_category:
            continue
        node_type = entry.get("type")
        label = entry.get("label")
        if not (isinstance(node_type, str) and isinstance(label, str) and node_type and label):
            continue
        if node_type in _HIDDEN_CATALOG_NODE_TYPES:
            continue
        by_category[category].append({"id": node_type, "label": label})

    # Apply the curated order; unknown types sort stably to the end.
    for children in by_category.values():
        children.sort(key=lambda child: _CATALOG_ORDER_INDEX.get(child["id"], len(_MAIN_CATALOG_NODE_ORDER)))

    return [
        {"id": category, "label": category, "children": by_category[category]}
        for category in CATEGORY_ORDER
        if by_category[category]
    ]


def _with_dynamic_children(category: dict[str, Any]) -> dict[str, Any]:
    children = list(category["children"])
    for child_factory in DYNAMIC_CHILDREN.get(str(category["id"]), ()):
        children.extend(child_factory())
    return {**category, "children": children}


@router.get("/node-categories")
def list_node_categories() -> dict:
    """Node library categories for the Comfy-style sidebar."""
    return {
        "version": 1,
        "categories": [_with_dynamic_children(category) for category in _static_catalog_categories()],
    }
