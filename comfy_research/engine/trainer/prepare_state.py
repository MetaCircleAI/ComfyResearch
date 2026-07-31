"""Package-private mutable state threaded through trainer preparation stages.

Only ``comfy_research.engine.trainer`` modules may import it. Field names match
the original ``prepare_trainer_run`` locals so stage boundaries stay mechanical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from comfy_research.engine.trainer.context import TrainerRunContext
from comfy_research.engine.trainer.model_helpers import _token_lm_vocab_seq_from_model
from comfy_research.engine.trainer.scalar import _scalar_int
from comfy_research.schemas.graph import Edge, Node

_REQUIRED_CONTEXT_FIELDS: tuple[str, ...] = (
    "nodes", "edges", "nmap", "trainer_node_id", "observable_nodes", "model",
    "optimizer", "criterion", "loss_scale", "trainer_task", "x_t", "y_t",
    "training_steps", "log_frequency", "log_schedule", "log_samples",
    "log_aggregation", "log_timing", "test_evaluation", "start_step", "loss_history",
    "test_loss_history", "reg_loss_history", "step_ticks", "epoch_ticks",
    "observable_metric_histories", "observable_embedding_histories",
    "observable_attention_slice_histories", "depth",
    "test_size", "resuming", "hessian_oversized_mode", "train_batch_size",
    "minibatch_sampling",
    "train_streaming", "seed", "train_seed", "train_materialize_fn", "test_materialize_fn",
    "minibatch_perm_seed", "device",
    # x_test_t / y_test_t may legitimately be None (not _UNSET), so they are
    # required too: init_build_placeholders_stage always assigns them.
    "x_test_t", "y_test_t",
)

_UNSET: Any = object()


@dataclass(slots=True)
class PrepareState:
    # Entry inputs are set by ``initial()`` and never remain _UNSET.
    nodes: list[Node]
    edges: list[Edge]
    trainer_node_id: str
    resume: dict[str, Any] | None
    hessian_oversized_policy: Literal["skip", "force"] | None
    get_user_observable_record: Any  # injected patch seam (trainer_run global)
    validate_only: bool = False

    # --- resolve_graph_stage writes ---
    nmap: Any = _UNSET
    trainer: Any = _UNSET
    model_in: Any = _UNSET
    combined_flat_chain: Any = _UNSET
    model_node: Any = _UNSET
    is_sequential_atomic: Any = _UNSET
    opt_node: Any = _UNSET
    ds_train: Any = _UNSET
    ds_test_raw: Any = _UNSET
    legacy_optional_test_wire: Any = _UNSET
    loss_incoming: Any = _UNSET
    weight_reg_loss_nodes: Any = _UNSET
    l2_projection_nodes: Any = _UNSET

    # --- determine_task_stage writes ---
    trainer_task: Any = _UNSET
    loss_node: Any = _UNSET

    # --- validate_compatibility_stage writes ---
    observable_nodes: Any = _UNSET
    want_kan_reg: Any = _UNSET

    # --- parse_hyperparams_stage writes ---
    td: Any = _UNSET
    training_steps: Any = _UNSET
    log_frequency: Any = _UNSET
    log_schedule: Any = _UNSET
    log_samples: Any = _UNSET
    log_aggregation: Any = _UNSET
    log_timing: Any = _UNSET
    test_evaluation: Any = _UNSET
    train_batch_size: Any = _UNSET
    minibatch_sampling: Any = _UNSET
    grad_clip_max_norm: Any = _UNSET
    lr_warmup_steps: Any = _UNSET
    lr_schedule: Any = _UNSET
    cosine_lr_min_fraction: Any = _UNSET
    exponential_lr_decay_factor: Any = _UNSET
    exponential_lr_decay_epochs: Any = _UNSET
    mup_embed_lr_mult: Any = _UNSET
    mup_hidden_lr_mult: Any = _UNSET
    mup_output_lr_mult: Any = _UNSET
    use_mup_lr_schedule: Any = _UNSET
    initialization_node: Any = _UNSET
    train_size: Any = _UNSET
    test_size: Any = _UNSET
    dd_train: Any = _UNSET
    dd_test: Any = _UNSET
    train_streaming: Any = _UNSET
    seed: Any = _UNSET
    train_seed: Any = _UNSET
    rng: Any = _UNSET
    lr: Any = _UNSET
    beta1: Any = _UNSET
    beta2: Any = _UNSET
    eps: Any = _UNSET
    momentum: Any = _UNSET
    weight_decay: Any = _UNSET
    muon_ns_steps: Any = _UNSET
    precondition_frequency: Any = _UNSET
    max_preconditioner_dim: Any = _UNSET
    cyclic_lr_min: Any = _UNSET
    cyclic_lr_max: Any = _UNSET
    cyclic_lr_cycle_steps: Any = _UNSET
    cyclic_batch_min: Any = _UNSET
    cyclic_batch_max: Any = _UNSET
    cyclic_batch_cycle_steps: Any = _UNSET
    cyclic_schedule_mode: Any = _UNSET
    cyclic_cycle_epochs: Any = _UNSET
    cyclic_steps_per_epoch: Any = _UNSET
    training_data_epochs: Any = _UNSET

    # --- init_build_placeholders_stage writes ---
    x_test_t: Any = _UNSET
    y_test_t: Any = _UNSET
    loss_d: Any = _UNSET
    loss_scale: Any = _UNSET
    ce_mem_a_slots: Any = _UNSET
    stack_io: Any = _UNSET
    streaming_teacher: Any = _UNSET

    # --- dataset materialization and model-building stages write ---
    dataset_arrays: Any = _UNSET
    x_t: Any = _UNSET
    y_t: Any = _UNSET
    model: Any = _UNSET
    criterion: Any = _UNSET
    depth: Any = _UNSET

    # --- prepare_finalize stages write ---
    chain_for_payloads: Any = _UNSET
    loop_n: Any = _UNSET
    loop_share: Any = _UNSET
    optimizer: Any = _UNSET
    optimizer_base_group_lrs: Any = _UNSET
    disable_extra_observables: Any = _UNSET
    hessian_oversized_mode: Any = _UNSET
    resuming: Any = _UNSET
    resume_ckpt_b64: Any = _UNSET
    start_step: Any = _UNSET
    loss_history: Any = _UNSET
    test_loss_history: Any = _UNSET
    reg_loss_history: Any = _UNSET
    step_ticks: Any = _UNSET
    epoch_ticks: Any = _UNSET
    observable_metric_histories: Any = _UNSET
    observable_embedding_histories: Any = _UNSET
    observable_attention_slice_histories: Any = _UNSET
    device: Any = _UNSET
    master_rng_seed: Any = _UNSET
    train_materialize_fn: Any = _UNSET
    test_materialize_fn: Any = _UNSET
    minibatch_perm_seed: Any = _UNSET


    def to_context(self) -> TrainerRunContext:
        """Validate required fields and assemble the run context."""
        missing = [f for f in _REQUIRED_CONTEXT_FIELDS if getattr(self, f) is _UNSET]
        if missing:
            raise AssertionError(f"PrepareState fields never set by any stage: {missing}")
        _, tok_seq = _token_lm_vocab_seq_from_model(self.model)
        dataset_extras = self.dataset_arrays.extras
        model_seed = _scalar_int((self.model_node.data or {}).get("seed"), 0)
        train_seed_override = _scalar_int((self.td or {}).get("trainSeed"), -1)
        # Preserve the legacy paper-reproduction stream when the new
        # trainer-level seed is absent. Explicit trainSeed opts into the new
        # independent training randomness.
        run_seed = self.train_seed if train_seed_override >= 0 else model_seed
        return TrainerRunContext(
            nodes=self.nodes,
            edges=self.edges,
            nmap=self.nmap,
            trainer_node_id=self.trainer_node_id,
            observable_nodes=self.observable_nodes,
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            loss_scale=self.loss_scale,
            trainer_task=self.trainer_task,
            x_t=self.x_t,
            y_t=self.y_t,
            x_test_t=self.x_test_t,
            y_test_t=self.y_test_t,
            training_steps=self.training_steps,
            log_frequency=self.log_frequency,
            log_schedule=self.log_schedule,
            log_samples=self.log_samples,
            log_aggregation=self.log_aggregation,
            log_timing=self.log_timing,
            test_evaluation=self.test_evaluation,
            start_step=self.start_step,
            loss_history=self.loss_history,
            test_loss_history=self.test_loss_history,
            reg_loss_history=self.reg_loss_history,
            step_ticks=self.step_ticks,
            epoch_ticks=self.epoch_ticks,
            observable_metric_histories=self.observable_metric_histories,
            observable_embedding_histories=self.observable_embedding_histories,
            observable_attention_slice_histories=self.observable_attention_slice_histories,
            depth=self.depth,
            test_size=self.test_size,
            resuming=bool(self.resuming),
            hessian_oversized_mode=self.hessian_oversized_mode,
            train_batch_size=self.train_batch_size,
            minibatch_sampling=self.minibatch_sampling,
            train_streaming=self.train_streaming,
            dataset_base_seed=self.seed,
            train_materialize=self.train_materialize_fn,
            test_materialize=self.test_materialize_fn,
            minibatch_perm_seed=self.minibatch_perm_seed,
            device=self.device,
            grad_clip_max_norm=self.grad_clip_max_norm,
            lr_warmup_steps=self.lr_warmup_steps,
            lr_schedule=self.lr_schedule,
            cosine_lr_min_fraction=self.cosine_lr_min_fraction,
            exponential_lr_decay_factor=self.exponential_lr_decay_factor,
            exponential_lr_decay_epochs=self.exponential_lr_decay_epochs,
            cyclic_lr_min=self.cyclic_lr_min,
            cyclic_lr_max=self.cyclic_lr_max,
            cyclic_lr_cycle_steps=self.cyclic_lr_cycle_steps,
            cyclic_batch_min=self.cyclic_batch_min,
            cyclic_batch_max=self.cyclic_batch_max,
            cyclic_batch_cycle_steps=self.cyclic_batch_cycle_steps,
            cyclic_schedule_mode=self.cyclic_schedule_mode,
            cyclic_cycle_epochs=self.cyclic_cycle_epochs,
            cyclic_steps_per_epoch=self.cyclic_steps_per_epoch,
            training_data_epochs=self.training_data_epochs,
            train_size=self.train_size,
            token_lm_seq_len=tok_seq,
            optimizer_base_group_lrs=self.optimizer_base_group_lrs,
            weight_reg_loss_nodes=self.weight_reg_loss_nodes,
            l2_projection_nodes=self.l2_projection_nodes,
            disable_extra_observables=self.disable_extra_observables,
            training_recipe=str(dataset_extras.get("training_recipe", "standard")),
            training_recipe_pixel_mean=dataset_extras.get("pixel_mean"),
            training_recipe_global_std=float(dataset_extras.get("global_std", 1.0)),
            run_seed=run_seed,
        )

    @classmethod
    def initial(
        cls,
        nodes: list[Node],
        edges: list[Edge],
        trainer_node_id: str,
        *,
        resume: dict[str, Any] | None,
        hessian_oversized_policy: Literal["skip", "force"] | None,
        get_user_observable_record: Any,
        validate_only: bool = False,
    ) -> "PrepareState":
        return cls(
            nodes=nodes,
            edges=edges,
            trainer_node_id=trainer_node_id,
            resume=resume,
            hessian_oversized_policy=hessian_oversized_policy,
            get_user_observable_record=get_user_observable_record,
            validate_only=validate_only,
        )
