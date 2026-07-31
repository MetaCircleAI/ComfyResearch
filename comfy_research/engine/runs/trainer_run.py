from __future__ import annotations

import base64
import io
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from fastapi import HTTPException

from comfy_research.api.user_observables import get_user_observable_record
from comfy_research.generated.node_capabilities import has_capability, node_types_with_capability
from comfy_research.engine.analysis.observable_user_eval import (
    eval_algebra_observable_user,
    eval_observable_user_mean_abs,
    validate_observable_user_tensor_path,
)
from comfy_research.engine.models.attention_only_model import AttentionOnlyModel, AttentionTokenPredictBundle
from comfy_research.engine.models.diffusion_score_model import DiffusionScoreMLP, diffusion_noise_mse_eval_mean, diffusion_noise_mse_loss
from comfy_research.engine.models.hyena_like_numeric_model import NumericHyenaModel, read_numeric_hyena_layout_from_md
from comfy_research.engine.models.linear_attention_model import LinearAttentionOnlyModel
from comfy_research.engine.analysis.token_lm_observable_support import TOKEN_LM_EMBEDDING_OBSERVABLE_MODULES
from comfy_research.engine.models.numeric_transformer_model import NumericTransformerModel, read_numeric_transformer_layout_from_md
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.models.mpp_spatiotemporal_model import MppSpatiotemporalModel
from comfy_research.engine.datasets.pde_field_dataset_runtime import (
    build_pde_field_arrays,
    pde_field_flat_dims,
)
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.runs.train_control import get_control, register_trainer, unregister_trainer
from comfy_research.engine.models.atomic_layer_chain import (
    SEQUENTIAL_MODEL_TYPES,
    build_sequential_from_atomic_tip,
    build_sequential_from_flat_atomic_chain,
    collect_atomic_layer_chain_front_to_back,
    collect_flat_atomic_chain_under_combined,
)
from comfy_research.engine.losses.loss_builders import (
    LossCriterionContext,
    TrainerTask,
    build_loss_criterion_for_node,
    dense_ce_memorization_a_slot_groups,
)
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node, build_model_from_type
from comfy_research.engine.models.model_loop_expand import ModelBlockLoop, expand_model_loop_stacking, resolve_loop_repeat_config
from comfy_research.engine.optimizers.mup_init import apply_mup_init
from comfy_research.engine.node_builder_registry import (
    registered_trainer_model_node_types,
)
from comfy_research.engine.optimizers.optimizer_builders import OptimizerBuildConfig, build_optimizer_for_node
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs, sample_x_from_sampler_dict
from comfy_research.engine.datasets.synthetic_dataset_builders import (
    _apply_linear_map,
    _bigram_spectrum_alpha,
    _bigram_spectrum_decay_type,
    _bigram_spectrum_lamb,
    _build_bigram_low_rank_arrays,
    _build_circular_motion_arrays,
    _build_kepler_2d_arrays,
    _build_uniform_linear_motion_arrays,
    _linear_like_targets_from_x,
    _linear_regression_weight_rng,
    _memorization_b_vocab_size,
    _memorization_b_xy_numpy,
    _memorization_class_probs,
    _random_linear_weights,
    _sample_memorization_labels,
    _unigram_token_probs,
)
from comfy_research.engine.datasets.dataset_runtime import (
    LEAF_SAMPLE_HANDLERS,
    STREAMING_TOKEN_SAMPLE_HANDLERS,
    sample_default_circle_walk_tokens,
    _declared_dataset_split_sizes,
    _materialize_toy_language_lm_numpy,
    _sample_dataset_arrays_for_mixer,
)
from comfy_research.engine.optimizers.signsgd_optimizer import SignSGD
from comfy_research.engine.datasets.streaming_seed import streaming_train_step_seed
from comfy_research.engine.datasets.teacher_dataset_runtime import teacher_labels_numpy
from comfy_research.engine.analysis.tensor_metrics import effective_rank_from_matrix
from comfy_research.engine.datasets.vision_datasets_runtime import (
    build_vision_numpy_arrays,
    vision_num_classes,
    _vision_flatten_feature_matrix,
)
from comfy_research.engine.datasets.trainer_dataset_streaming import rematerialize_train_tensors_for_step
from comfy_research.engine.models.vision_models import (
    infer_vision_input_channels_height_width,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind
from comfy_research.engine.trainer.scalar import (
    _scalar_bool,
    _scalar_float,
    _scalar_int,
    _scalar_str,
)
from comfy_research.engine.trainer.graph import (
    _DATASET_NODE_TYPE_LABEL,
    _DATASET_NODE_TYPES,
    _OPTIMIZER_NODE_TYPE_LABEL,
    _OPTIMIZER_NODE_TYPES,
    _incoming,
    _incoming_all,
    _incoming_to_target,
    _node_map,
    _require,
    _require_dataset,
    _require_optimizer,
)
from comfy_research.engine.trainer.tensor_norms import (
    MAX_OBSERVABLE_L2_TENSOR_SERIES,
    _gradient_l2_norm_global,
    _gradient_l2_norm_per_top_level,
    _parameter_l2_norms_named,
    _weight_l1_norm,
    _weight_l2_norm,
    _weight_l2_norm_per_top_level,
)
from comfy_research.engine.trainer.device_runtime import (
    _optimizer_state_to_device,
    _resolve_trainer_compute_device,
    _trainer_configure_reproducible_torch,
    _trainer_force_single_thread_cpu,
    _trainer_master_rng_seed,
)
from comfy_research.engine.trainer.checkpoint import (
    _apply_checkpoint_b64,
    _pack_checkpoint_b64,
    load_model_weights_from_checkpoint_b64,
)
from comfy_research.engine.trainer.model_payloads import (
    apply_parameter_tensor_payloads_from_atomic_chain,
    apply_parameter_tensor_payloads_from_node,
)
from comfy_research.engine.trainer.model_helpers import (
    _ATOMIC_CHAIN_FIRST,
    _ATOMIC_CHAIN_LAST,
    _atomic_chain_dataset_dims,
    _count_nonlinear_relu_neurons,
    _find_first_embedding,
    _flatten_features_for_mse,
    _forward_reg,
    _mpp_spatiotemporal_core,
    _multi_token_transformer_core,
    _numeric_hyena_core,
    _numeric_transformer_core,
    _prepare_x_for_atomic_sequential,
    _regression_batch_for_model,
    _token_lm_vocab_seq_from_model,
)
from comfy_research.engine.trainer.eval_batches import (
    _batched_classification_accuracy,
    _batched_primary_loss_mean,
)
from comfy_research.engine.trainer.loss_terms import (
    _KAN_REG_METRICS,
    _LOSS_SOCKET_AUX_KINDS,
    _PRIMARY_TRAINER_LOSS_KINDS,
    _WEIGHT_REG_TRAINER_LOSS_KINDS,
    _apply_l2_weight_projection,
    _extra_loss_additions,
    _kan_reg_loss_term,
    _l2_projection_target_norm,
    _trainer_loss_wiring,
    _trainer_primary_loss_tensor,
    _weight_reg_loss_additions,
    _weight_reg_loss_term,
)
from comfy_research.engine.trainer.observable_config import (
    OBS_ENCODER_LAYER_SERIES_NODEKINDS,
    _ACTIVATION_STATS_LAYER_SUBMOD_RE,
    _EMBEDDING_OBSERVABLE_MODEL_TYPES,
    _activation_stats_bucket_from_module_name,
    _activation_stats_layer_mode,
    _activation_stats_ordered_bucket_keys,
    _gradient_norm_segments_restored_from_hist,
    _gradient_norm_tensor_segments_restored_from_hist,
    _log_step_needs_attention_cache,
    _obs_encoder_layer_mode,
    _observable_gradient_norm_normalized,
    _observable_hist_subseries_suffixes,
    _observable_l2_aggregation,
    _observable_metrics_log_enabled,
    _sanitize_observable_hist_segment,
    _sink_attention_mass_layer_mode,
    _trainer_disable_extra_observables,
)
from comfy_research.engine.trainer.observable_metrics import (
    HESSIAN_FORCE_MAX_PARAMS,
    HESSIAN_PARAM_LIMIT,
    SPECTRAL_POWER_MAX_ELEMS,
    _activation_mean_std_averages,
    _activation_mean_std_bucketed,
    _activation_norm_mean_and_outlier_per_bucket,
    _activation_norm_mean_and_outlier_ratio,
    _activation_stats_layer_series_payload_from_hist,
    _attn_entropy_mean,
    _attn_head_sink_max,
    _attn_max_weight_mean,
    _attn_position_bias_ratio,
    _attn_sink_mass,
    _effective_rank_max_per_encoder_layer,
    _hessian_loss_eigenvalues,
    _layer_bucket_flat_concat_vectors,
    _observable_multi_series_l2_payload_from_hist,
    _paired_layer_series_payload_from_hist,
    _paired_member_series_payload_from_hist,
    _softmax_attention_probs_all_layers_or_none,
    _softmax_attention_probs_or_none,
    _unwrap_model_block_loop_for_attention,
)
from comfy_research.engine.trainer.attention_relation_dsl import (
    AttentionRelationDslError,
    AttentionRelationPredicate,
    compile_attention_relation_predicate,
)
from comfy_research.engine.trainer.attention_relation_metrics import (
    MAX_SELECTED_PAIRS,
    attention_relation_score,
)
from comfy_research.engine.trainer.observable_viz import (
    _LEGACY_VIZ_TO_OBS,
    _VARIANT_TO_OBS,
    _coerce_node_type_value,
    _default_observable_viz_variant_for_obs,
    _obs_viz_stream_row,
    _observable_connected_to_trainer,
    _trainer_observable_input_handle_ok,
    find_loss_visualization_targets,
    observable_viz_metric_updates,
)
from comfy_research.engine.trainer.teacher_helpers import (
    _build_teacher_mlp_eval,
    _resolve_teacher_model_eval,
    _resolve_teacher_sampler,
    _sample_teacher_input_tensor,
)
from comfy_research.engine.trainer.context import TrainerRunContext
from comfy_research.engine.trainer.prepare_state import PrepareState
from comfy_research.engine.trainer.prepare_graph import (
    _TRAINER_MODEL_TYPE_LABEL,
    _TRAINER_REGISTERED_MODEL_TYPES,
    determine_task_stage,
    resolve_graph_stage,
    validate_compatibility_stage,
)
from comfy_research.engine.trainer.prepare_config import (
    MAX_TRAINING_STEPS,
    init_build_placeholders_stage,
    parse_hyperparams_stage,
)
from comfy_research.engine.trainer.dataset_materialize import materialize_dataset_for_training
from comfy_research.engine.trainer.prepare_build_vision import build_vision_model
from comfy_research.engine.trainer.prepare_build_atomic import build_atomic_model
from comfy_research.engine.trainer.prepare_build_vector import build_vector_model
from comfy_research.engine.trainer.prepare_build_token import build_token_model
from comfy_research.engine.trainer.recorder import ObservableRecorder
from comfy_research.engine.trainer.training_loop import (
    _loss_plot_png_b64,
    _trainer_lr_mult_for_step,
    run_training_loop,
)
from comfy_research.engine.trainer.prepare_finalize import (
    apply_payloads_and_mup_stage,
    build_criterion_stage,
    build_materialize_closures_stage,
    build_optimizer_stage,
    build_training_model_stage,
    expand_model_loop_stage,
    materialize_dataset_stage,
    place_on_device_stage,
    restore_histories_stage,
)
from comfy_research.engine.trainer.user_observable_helpers import (
    _log_step_needs_representation_cache,
    _user_observable_algebra_rec,
    _user_observable_definition_code,
    _user_observable_path_anchor,
)
from comfy_research.engine.trainer.dataset_helpers import (
    _DATASET_MIXER_TYPES,
    _DIFFUSION_NOISE_DATASET_TYPES,
    _LINEAR_LIKE_DATASET_TYPES,
    _PDE_FIELD_DATASET_TYPES,
    _TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES,
    _TOKEN_CLASSIFICATION_DATASET_TYPES,
    _TOY_LANGUAGE_TOKEN_DATASETS,
    _VECTOR_REGRESSION_DATASET_TYPES,
    _VISION_DATASET_TYPES,
    _dataset_sampling_mode_raw,
    _resolve_linear_dataset_mlp_dims,
    _sample_base_vector_dataset_arrays,
    _take_train_minibatch,
    _trainer_train_dataset_streaming,
    _vision_flatten_enabled,
)

# Deferred: gradient-noise (extra backward), histogram histories, Adam/Muon update ratios, attention entropy.
# Cap for embedding-effective-rank scans over 2D parameter slices (element count).
# Datasets whose MSE / diffusion targets are continuous vectors from a linear teacher or independent noise.
# memorization_a_dataset is excluded: it is a discrete-label benchmark and only supports cross_entropy_loss.
def prepare_trainer_run(
    nodes: list[Node],
    edges: list[Edge],
    trainer_node_id: str,
    resume: dict[str, Any] | None = None,
    *,
    hessian_oversized_policy: Literal["skip", "force"] | None = None,
    validate_only: bool = False,
) -> TrainerRunContext:
    """Validate graph, build model/tensors, and return mutable training state.

    Call this before creating a StreamingResponse so HTTPException maps to a proper 400
    (Starlette sends HTTP 200 headers before iterating a streaming body).
    """
    s = PrepareState.initial(
        nodes,
        edges,
        trainer_node_id,
        resume=resume,
        hessian_oversized_policy=hessian_oversized_policy,
        get_user_observable_record=get_user_observable_record,
        validate_only=validate_only,
    )
    resolve_graph_stage(s)
    determine_task_stage(s)
    validate_compatibility_stage(s)
    parse_hyperparams_stage(s)
    init_build_placeholders_stage(s)
    materialize_dataset_stage(s)
    build_training_model_stage(s)
    expand_model_loop_stage(s)
    apply_payloads_and_mup_stage(s)
    build_optimizer_stage(s)
    restore_histories_stage(s)
    build_criterion_stage(s)
    place_on_device_stage(s)
    build_materialize_closures_stage(s)
    return s.to_context()


def iter_trainer_events_from_context(ctx: TrainerRunContext) -> Iterator[dict[str, Any]]:
    """Stream NDJSON events from a prepared run (no validation; use prepare_trainer_run first)."""
    recorder = ObservableRecorder(ctx, get_user_observable_record=get_user_observable_record)
    recorder.restore_resume_series()
    yield from run_training_loop(ctx, recorder)


def iter_trainer_events(
    nodes: list[Node],
    edges: list[Edge],
    trainer_node_id: str,
    resume: dict[str, Any] | None = None,
    *,
    hessian_oversized_policy: Literal["skip", "force"] | None = None,
) -> Iterator[dict[str, Any]]:
    """Run training and yield NDJSON: progress, optional paused/aborted, or complete."""
    ctx = prepare_trainer_run(
        nodes,
        edges,
        trainer_node_id,
        resume,
        hessian_oversized_policy=hessian_oversized_policy,
    )
    yield from iter_trainer_events_from_context(ctx)


def run_trainer_subgraph(
    nodes: list[Node],
    edges: list[Edge],
    trainer_node_id: str,
) -> tuple[list[float], list[int], str]:
    for ev in iter_trainer_events(nodes, edges, trainer_node_id):
        if ev["type"] == "complete":
            return (
                ev["loss_history"],
                ev["step_ticks"],
                ev["plot_png_base64"],
            )
    raise RuntimeError("trainer finished without complete event")
