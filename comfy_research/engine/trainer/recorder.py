"""ObservableRecorder: owns observable state and the per-step record dispatch.

``record()`` looks up
OBSERVABLE_RECORD_HANDLERS (module
tail) by NodeKind value string; each kind's logic lives in one
_record_<kind> method (bodies verbatim from the original closure; captured
names alias self fields). Adding an observable = one method + one registry
entry. The user-observable record getter is injected (patch seam; guard
forbids importing comfy_research.api.user_observables).
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from comfy_research.engine.trainer.context import TrainerRunContext
from comfy_research.engine.trainer.eval_batches import (
    _batched_primary_loss_mean,
    _bounded_eval_batch_size,
)
from comfy_research.engine.trainer.observable_config import (
    OBS_ENCODER_LAYER_SERIES_NODEKINDS,
    _EMBEDDING_OBSERVABLE_MODEL_TYPES,
    _activation_stats_layer_mode,
    _gradient_norm_segments_restored_from_hist,
    _gradient_norm_tensor_segments_restored_from_hist,
    _log_step_needs_attention_cache,
    _obs_encoder_layer_mode,
    _observable_hist_subseries_suffixes,
    _observable_l2_aggregation,
    _observable_metrics_log_enabled,
    _sink_attention_mass_layer_mode,
)
from comfy_research.engine.trainer.observable_metrics import (
    _softmax_attention_probs_all_layers_or_none,
    _softmax_attention_probs_or_none,
)
from comfy_research.engine.trainer.scalar import _scalar_bool
from comfy_research.engine.trainer.user_observable_helpers import (
    _log_step_needs_representation_cache,
)
from comfy_research.nodes.registry import defs_recorders
from comfy_research.schemas.graph import Node, NodeKind


class ObservableRecorder:
    def __init__(self, ctx: TrainerRunContext, *, get_user_observable_record: Any) -> None:
        self.cosine_lr_min_fraction = ctx.cosine_lr_min_fraction
        self.criterion = ctx.criterion
        self.depth = ctx.depth
        self.disable_extra_observables = ctx.disable_extra_observables
        self.edges = ctx.edges
        self.grad_clip_max_norm = ctx.grad_clip_max_norm
        self.hessian_oversized_mode = ctx.hessian_oversized_mode
        self.l2_projection_nodes = ctx.l2_projection_nodes
        self.log_frequency = ctx.log_frequency
        self.loss_history = ctx.loss_history
        self.loss_scale = ctx.loss_scale
        self.lr_schedule = ctx.lr_schedule
        self.lr_warmup_steps = ctx.lr_warmup_steps
        self.minibatch_perm_seed = ctx.minibatch_perm_seed
        self.model = ctx.model
        self.nmap = ctx.nmap
        self.nodes = ctx.nodes
        self.observable_embedding_histories = ctx.observable_embedding_histories
        self.observable_attention_slice_histories = ctx.observable_attention_slice_histories
        self.observable_metric_histories = ctx.observable_metric_histories
        self.observable_nodes = ctx.observable_nodes
        self.optimizer = ctx.optimizer
        self.optimizer_base_group_lrs = ctx.optimizer_base_group_lrs
        self.reg_loss_history = ctx.reg_loss_history
        self.resuming = ctx.resuming
        self.start_step = ctx.start_step
        self.step_ticks = ctx.step_ticks
        self.epoch_ticks = ctx.epoch_ticks
        self.test_loss_history = ctx.test_loss_history
        self.test_evaluation = ctx.test_evaluation
        self.test_materialize = ctx.test_materialize
        self.test_size = ctx.test_size
        self.train_batch_size = ctx.train_batch_size
        self.eval_batch_size = _bounded_eval_batch_size(ctx.train_batch_size)
        self.train_materialize = ctx.train_materialize
        self.trainer_node_id = ctx.trainer_node_id
        self.trainer_task = ctx.trainer_task
        self.training_steps = ctx.training_steps
        self.weight_reg_loss_nodes = ctx.weight_reg_loss_nodes
        self.x_t = ctx.x_t
        self.x_test_t = ctx.x_test_t
        self.y_t = ctx.y_t
        self.y_test_t = ctx.y_test_t
        self.get_user_observable_record = get_user_observable_record
        # Per-call ``record()`` state.
        self._step = 0
        self._primary_train_val = 0.0
        self._xr = None
        self._yr = None
        self._x_test_log = None
        self._y_test_log = None
        self._metric_overrides: dict[str, float] = {}
        self._rep_tensors_cache = None
        self._attn_layers_cache = False
        self.observable_rng_generators: dict[str, torch.Generator] = {}
        # Observable state initialized before the training loop starts.
        self.attention_arrays_init: dict[str, np.ndarray] | None = None
        if isinstance(self.model, _EMBEDDING_OBSERVABLE_MODEL_TYPES):
            self.attention_arrays_init = self.model.observable_numpy_arrays()

        self.kan_regs = [on for on in self.observable_nodes if on.type == NodeKind.kan_reg]

        self.embedding_prev_for_drift: dict[str, np.ndarray | None] = {}
        self.embedding_prev_layer_flat_for_drift: dict[tuple[str, str], np.ndarray] = {}
        self.weight_displacement_init: dict[str, torch.Tensor] = {}
        self.gradient_norm_canonical_segments: dict[str, list[str]] = {}
        self.gradient_norm_seg_to_raw: dict[str, dict[str, str]] = {}
        self.gradient_norm_canonical_tensor: dict[str, list[str]] = {}
        self.gradient_norm_tensor_seg_to_raw: dict[str, dict[str, str]] = {}
        self.weight_l2_canonical_top: dict[str, list[str]] = {}
        self.weight_l2_top_seg_to_raw: dict[str, dict[str, str]] = {}
        self.weight_l2_canonical_tensor: dict[str, list[str]] = {}
        self.weight_l2_tensor_seg_to_raw: dict[str, dict[str, str]] = {}
        self.sink_attn_mass_layer_canon: dict[str, list[str]] = {}
        self.algebra_tensor_member_canon: dict[str, list[str]] = {}
        self.encoder_obs_layer_canon: dict[str, list[str]] = {}
        self.activation_stats_layer_canon: dict[str, list[str]] = {}

    def restore_resume_series(self):
        activation_stats_layer_canon = self.activation_stats_layer_canon
        encoder_obs_layer_canon = self.encoder_obs_layer_canon
        gradient_norm_canonical_segments = self.gradient_norm_canonical_segments
        gradient_norm_canonical_tensor = self.gradient_norm_canonical_tensor
        observable_metric_histories = self.observable_metric_histories
        observable_nodes = self.observable_nodes
        sink_attn_mass_layer_canon = self.sink_attn_mass_layer_canon
        weight_l2_canonical_tensor = self.weight_l2_canonical_tensor
        weight_l2_canonical_top = self.weight_l2_canonical_top
        for _gn in observable_nodes:
            if _gn.type == NodeKind.observable_gradient_norm:
                _odd = _gn.data or {}
                agg_g = _observable_l2_aggregation(_odd)
                if agg_g == "top_level_module" or _scalar_bool(_odd.get("perTopLevel"), False):
                    _rst = _gradient_norm_segments_restored_from_hist(_gn.id, observable_metric_histories)
                    if _rst:
                        gradient_norm_canonical_segments[_gn.id] = _rst
                if agg_g == "tensor":
                    _rst_t = _gradient_norm_tensor_segments_restored_from_hist(_gn.id, observable_metric_histories)
                    if _rst_t:
                        gradient_norm_canonical_tensor[_gn.id] = _rst_t
            elif _gn.type == NodeKind.observable_weight_l2:
                _owd = _gn.data or {}
                agg_w = _observable_l2_aggregation(_owd)
                if agg_w == "top_level_module":
                    _rw = _observable_hist_subseries_suffixes(_gn.id, "top", observable_metric_histories)
                    if _rw:
                        weight_l2_canonical_top[_gn.id] = _rw
                elif agg_w == "tensor":
                    _rwt = _observable_hist_subseries_suffixes(_gn.id, "tensor", observable_metric_histories)
                    if _rwt:
                        weight_l2_canonical_tensor[_gn.id] = _rwt
            elif _gn.type in OBS_ENCODER_LAYER_SERIES_NODEKINDS:
                _oed = _gn.data or {}
                if _obs_encoder_layer_mode(_oed) == "all_layers":
                    _rsk = _observable_hist_subseries_suffixes(_gn.id, "layer", observable_metric_histories)
                    if _rsk:
                        encoder_obs_layer_canon[_gn.id] = _rsk
            elif _gn.type == NodeKind.observable_sink_attention_mass:
                _sam = _gn.data or {}
                if _sink_attention_mass_layer_mode(_sam) == "all_layers":
                    _rsl = _observable_hist_subseries_suffixes(_gn.id, "layer", observable_metric_histories)
                    if _rsl:
                        sink_attn_mass_layer_canon[_gn.id] = _rsl
            elif _gn.type == NodeKind.observable_activation_stats:
                _asd = _gn.data or {}
                if _activation_stats_layer_mode(_asd) == "all_layers":
                    _rsa = _observable_hist_subseries_suffixes(_gn.id, "layer_mean", observable_metric_histories)
                    if _rsa:
                        activation_stats_layer_canon[_gn.id] = _rsa

    def _eval_test_loss(self, x_eval: torch.Tensor, y_eval: torch.Tensor):
        return _batched_primary_loss_mean(
            self.model,
            x_eval,
            y_eval,
            batch_size=self.eval_batch_size,
            trainer_task=self.trainer_task,
            criterion=self.criterion,
            loss_scale=self.loss_scale,
        )

    def record(
        self,
        step: int,
        primary_train_val: float,
        *,
        epoch: float | None = None,
        metric_overrides: dict[str, float] | None = None,
        reg_train_val: float = 0.0,
        x_log: torch.Tensor | None = None,
        y_log: torch.Tensor | None = None,
    ) -> None:
        disable_extra_observables = self.disable_extra_observables
        get_user_observable_record = self.get_user_observable_record
        loss_history = self.loss_history
        observable_nodes = self.observable_nodes
        reg_loss_history = self.reg_loss_history
        step_ticks = self.step_ticks
        epoch_ticks = self.epoch_ticks
        test_loss_history = self.test_loss_history
        test_materialize = self.test_materialize
        x_t = self.x_t
        x_test_t = self.x_test_t
        y_t = self.y_t
        xr = x_log if x_log is not None else x_t
        yr = y_log if y_log is not None else y_t
        x_test_log: torch.Tensor | None = None
        y_test_log: torch.Tensor | None = None
        step_ticks.append(step)
        epoch_ticks.append(float(step if epoch is None else epoch))
        loss_history.append(float(primary_train_val))
        reg_loss_history.append(float(reg_train_val))
        should_evaluate_test = self.test_evaluation == "log_ticks" or (
            self.test_evaluation == "final_only" and step >= self.training_steps
        )
        if x_test_t is not None and should_evaluate_test:
            x_test_log, y_test_log = test_materialize(step)
            if x_test_log is not None and y_test_log is not None:
                test_loss_history.append(self._eval_test_loss(x_test_log, y_test_log))
        self._rep_tensors_cache = None
        self._attn_layers_cache = False

        # Per-call state MUST be set before the cache prewarm below: the lazy
        # Helpers read self._xr; the original closures captured
        # xr directly, so prewarm order didn't matter pre-extraction).
        self._step = step
        self._primary_train_val = primary_train_val
        self._metric_overrides = metric_overrides or {}
        self._xr = xr
        self._yr = yr
        self._x_test_log = x_test_log
        self._y_test_log = y_test_log

        if _log_step_needs_representation_cache(observable_nodes, disable_extra_observables, get_user_observable_record=get_user_observable_record):
            self._representation_tensors_for_log()
        if _log_step_needs_attention_cache(observable_nodes, disable_extra_observables):
            self._attention_layers_for_log()

        for on in observable_nodes:
            if not _observable_metrics_log_enabled(on, disable_extra_observables):
                continue
            kind = on.type.value if isinstance(on.type, NodeKind) else str(on.type)
            handler = OBSERVABLE_RECORD_HANDLERS.get(kind)
            if handler is not None:
                handler(self, on)


    def _representation_tensors_for_log(self) -> dict[str, torch.Tensor]:
        if self._rep_tensors_cache is None:
            from comfy_research.engine.analysis.representation_specs import collect_representation_tensors

            self._rep_tensors_cache = collect_representation_tensors(self.model, self._xr, self.depth)
        return self._rep_tensors_cache

    def _attention_layers_for_log(self) -> list[torch.Tensor] | None:
        if self._attn_layers_cache is False:
            self._attn_layers_cache = _softmax_attention_probs_all_layers_or_none(self.model, self._xr)
        return self._attn_layers_cache

    def _attention_last_layer_for_log(self) -> torch.Tensor | None:
        layers = self._attention_layers_for_log()
        if layers:
            return layers[-1]
        return _softmax_attention_probs_or_none(self.model, self._xr)



# Kind -> handler registry. ``record()`` dispatches through this dict; adding an
# observable means adding a _record_* method AND one entry here (guard:
# test_observable_registry_complete_and_str_keyed). Keys are NodeKind value
# strings. Provider registration may replace this literal.
_HAND_WRITTEN_HANDLERS: dict[str, Callable[[ObservableRecorder, Node], None]] = {
}

# Combine specialized handlers with providers registered by NodeDefs.
OBSERVABLE_RECORD_HANDLERS: dict[str, Callable[[ObservableRecorder, Node], None]] = {
    **_HAND_WRITTEN_HANDLERS,
    **defs_recorders(),
}


def observable_record_handler_kinds() -> frozenset[str]:
    """Runtime source of truth for sync guards, replacing AST literal scans."""
    return frozenset(OBSERVABLE_RECORD_HANDLERS)
