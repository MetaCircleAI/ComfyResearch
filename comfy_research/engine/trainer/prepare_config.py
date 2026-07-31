"""Trainer hyperparameter parsing and build-state initialization.

Each function reads its inputs from PrepareState and writes validated values
back to the same state object.
"""
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.datasets.dataset_runtime import _declared_dataset_split_sizes
from comfy_research.engine.losses.loss_builders import dense_ce_memorization_a_slot_groups
from comfy_research.engine.optimizers.cyclic_schedules import cbs_epochs_to_training_steps, steps_per_epoch
from comfy_research.engine.trainer.cyclic_helpers import (
    _cyclic_cycle_epochs_from_node_data,
    _cyclic_cycle_steps_from_node_data,
    _cyclic_schedule_mode_from_data,
)
from comfy_research.engine.trainer.dataset_helpers import (
    _DATASET_MIXER_TYPES,
    _TOY_LANGUAGE_TOKEN_DATASETS,
    _VISION_DATASET_TYPES,
    _trainer_train_dataset_streaming,
)
from comfy_research.engine.trainer.device_runtime import _trainer_master_rng_seed
from comfy_research.engine.trainer.graph import _incoming_to_target
from comfy_research.engine.trainer.prepare_state import PrepareState
from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int, _scalar_str
from comfy_research.schemas.graph import Node, NodeKind

MAX_TRAINING_STEPS = 200_000

# Node kinds accepted on a model's ``initialization`` socket. Guarded by
# test_initialization_socket_supported_kinds; apply dispatch lives in
# prepare_finalize.apply_payloads_and_mup_stage.
SUPPORTED_INITIALIZATION_NODE_KINDS: frozenset[NodeKind] = frozenset(
    {
        NodeKind.idnns_initialization,
        NodeKind.mup_initialization,
        NodeKind.saxe_initialization,
        NodeKind.symmetrized_mlp_init,
    }
)


def parse_hyperparams_stage(s: PrepareState) -> None:
    """Reads: ds_test_raw, ds_train, edges, legacy_optional_test_wire, model_in, model_node, nmap, opt_node, trainer
    Writes: td, training_steps, log_frequency, train_batch_size, grad_clip_max_norm, lr_warmup_steps, lr_schedule, cosine_lr_min_fraction, mup_embed_lr_mult, mup_hidden_lr_mult, mup_output_lr_mult, use_mup_lr_schedule, initialization_node, train_size, test_size, dd_train, dd_test, train_streaming, seed, train_seed, rng, lr, beta1, beta2, eps, momentum, weight_decay, muon_ns_steps, precondition_frequency, max_preconditioner_dim, cyclic_lr_min, cyclic_lr_max, cyclic_batch_min, cyclic_batch_max, cyclic_batch_cycle_steps, cyclic_schedule_mode, cyclic_cycle_epochs, cyclic_steps_per_epoch, training_data_epochs
    """
    ds_test_raw = s.ds_test_raw
    ds_train = s.ds_train
    edges = s.edges
    legacy_optional_test_wire = s.legacy_optional_test_wire
    model_in = s.model_in
    model_node = s.model_node
    nmap = s.nmap
    opt_node = s.opt_node
    trainer = s.trainer
    td: dict[str, Any] = trainer.data or {}
    training_steps = int(td.get("trainingSteps", 1000))
    log_frequency = int(td.get("logFrequency", 10))
    log_schedule = _scalar_str(td.get("logSchedule"), "fixed_interval").strip().lower()
    log_samples = max(1, _scalar_int(td.get("logSamples"), 1800))
    log_aggregation = _scalar_str(td.get("logAggregation"), "last_batch").strip().lower()
    log_timing = _scalar_str(td.get("logTiming"), "post_update").strip().lower()
    test_evaluation = _scalar_str(td.get("testEvaluation"), "log_ticks").strip().lower()
    minibatch_sampling = _scalar_str(
        td.get("minibatchSampling"), "independent_step"
    ).strip().lower()
    if training_steps < 1 or training_steps > MAX_TRAINING_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"training steps must be 1..{MAX_TRAINING_STEPS}",
        )
    if log_frequency < 1:
        raise HTTPException(status_code=400, detail="log frequency must be >= 1")
    if log_schedule not in {"fixed_interval", "idnns_logspace"}:
        raise HTTPException(
            status_code=400,
            detail="trainer logSchedule must be fixed_interval or idnns_logspace.",
        )
    if log_aggregation not in {"last_batch", "interval_sample_mean"}:
        raise HTTPException(
            status_code=400,
            detail="trainer logAggregation must be last_batch or interval_sample_mean.",
        )
    if log_timing not in {"post_update", "pre_update"}:
        raise HTTPException(
            status_code=400,
            detail="trainer logTiming must be post_update or pre_update.",
        )
    if log_timing == "pre_update" and (
        log_schedule != "fixed_interval" or log_aggregation != "last_batch"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "pre_update logging requires fixed_interval schedule and "
                "last_batch aggregation."
            ),
        )
    if test_evaluation not in {"log_ticks", "final_only", "disabled"}:
        raise HTTPException(
            status_code=400,
            detail="trainer testEvaluation must be log_ticks, final_only, or disabled.",
        )
    if minibatch_sampling not in {
        "independent_step",
        "epoch_shuffle",
        "affine_epoch",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "trainer minibatchSampling must be independent_step, "
                "epoch_shuffle, or affine_epoch."
            ),
        )

    raw_bs = td.get("batchSize", -1)
    train_batch_size = _scalar_int(raw_bs, -1)
    if train_batch_size == 0 or train_batch_size < -1:
        raise HTTPException(
            status_code=400,
            detail="trainer batchSize must be -1 (full batch) or an integer >= 1.",
        )

    grad_clip_max_norm = max(0.0, _scalar_float(td.get("gradClipMaxNorm"), 0.0))
    lr_sched_in = _incoming_to_target(edges, nmap, opt_node.id, "lr_schedule")
    mup_lr_sched_in = _incoming_to_target(edges, nmap, opt_node.id, "mup_lr_schedule")
    # Legacy graphs: μP schedule node wired only to the «lr schedule» target — accept once as layer-only.
    if lr_sched_in is not None and lr_sched_in.type == NodeKind.mup_lr_schedule:
        if mup_lr_sched_in is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Optimizer {opt_node.id!r}: connect mup_lr_schedule to the lower dot on the lr schedule "
                    f"input (mup_lr_schedule handle), not the upper lr_schedule dot, or remove the duplicate μP "
                    f"connection."
                ),
            )
        mup_lr_sched_in = lr_sched_in
        lr_sched_in = None

    lr_warmup_steps = 0
    lr_schedule = "constant"
    cosine_lr_min_fraction = 0.0
    exponential_lr_decay_factor = 0.95
    exponential_lr_decay_epochs = 1
    cyclic_lr_min = 0.0
    cyclic_lr_max = 0.0
    cyclic_lr_node: Node | None = None
    if lr_sched_in is not None:
        if lr_sched_in.type == NodeKind.cyclic_lr_schedule:
            cyclic_lr_node = lr_sched_in
            ld_clr: dict[str, Any] = lr_sched_in.data or {}
            cyclic_lr_min = _scalar_float(ld_clr.get("lrMin"), 0.001)
            cyclic_lr_max = _scalar_float(ld_clr.get("lrMax"), 0.005)
        elif lr_sched_in.type != NodeKind.lr_schedule:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Optimizer {opt_node.id!r} upper lr schedule dot expects lr_schedule or "
                    f"cyclic_lr_schedule; got {lr_sched_in.type!r}. For μP layer LR multipliers, wire "
                    f"mup_lr_schedule to the lower dot."
                ),
            )
        else:
            ld_ls: dict[str, Any] = lr_sched_in.data or {}
            lr_warmup_steps = max(0, _scalar_int(ld_ls.get("lrWarmupSteps"), 0))
            lr_schedule = _scalar_str(ld_ls.get("lrSchedule"), "constant").strip().lower()
            cosine_lr_min_fraction = float(
                max(0.0, min(1.0, _scalar_float(ld_ls.get("cosineLrMinFraction"), 0.0)))
            )
            exponential_lr_decay_factor = float(
                max(0.0, min(1.0, _scalar_float(ld_ls.get("exponentialDecayFactor"), 0.95)))
            )
            exponential_lr_decay_epochs = max(1, _scalar_int(ld_ls.get("exponentialDecayEpochs"), 1))

    batch_sched_in = _incoming_to_target(edges, nmap, s.trainer_node_id, "batch_schedule")
    cyclic_batch_node: Node | None = None
    cyclic_batch_min = 0
    cyclic_batch_max = 0
    if batch_sched_in is not None:
        if batch_sched_in.type != NodeKind.cyclic_batch_schedule:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Trainer {s.trainer_node_id!r} batch_schedule socket expects cyclic_batch_schedule; "
                    f"got {batch_sched_in.type!r}."
                ),
            )
        cyclic_batch_node = batch_sched_in
        ld_cbs: dict[str, Any] = batch_sched_in.data or {}
        cyclic_batch_min = max(1, _scalar_int(ld_cbs.get("batchMin"), 128))
        cyclic_batch_max = max(1, _scalar_int(ld_cbs.get("batchMax"), 640))
        if train_batch_size < 1:
            train_batch_size = cyclic_batch_min

    mup_embed_lr_mult = 1.0
    mup_hidden_lr_mult = 1.0
    mup_output_lr_mult = 1.0
    use_mup_lr_schedule = False
    if mup_lr_sched_in is not None:
        if mup_lr_sched_in.type != NodeKind.mup_lr_schedule:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Optimizer {opt_node.id!r} lower lr schedule dot expects a mup_lr_schedule node; "
                    f"got {mup_lr_sched_in.type!r}."
                ),
            )
        use_mup_lr_schedule = True
        if opt_node.type == NodeKind.muon_optimizer:
            raise HTTPException(
                status_code=400,
                detail="mup_lr_schedule cannot be used with muon_optimizer; use adam_optimizer.",
            )
        if opt_node.type != NodeKind.adam_optimizer:
            raise HTTPException(
                status_code=400,
                detail="mup_lr_schedule requires adam_optimizer in this build.",
            )
        md = mup_lr_sched_in.data or {}
        mup_embed_lr_mult = max(0.0, _scalar_float(md.get("mupEmbedLrMult"), 1.0))
        mup_hidden_lr_mult = max(0.0, _scalar_float(md.get("mupHiddenLrMult"), 1.0))
        mup_output_lr_mult = max(0.0, _scalar_float(md.get("mupOutputLrMult"), 1.0))
    init_in = _incoming_to_target(edges, nmap, model_in.id, "initialization")
    initialization_node = None
    if init_in is not None:
        if init_in.type not in SUPPORTED_INITIALIZATION_NODE_KINDS:
            supported = ", ".join(sorted(k.value for k in SUPPORTED_INITIALIZATION_NODE_KINDS))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {model_in.id!r} initialization socket expects one of: {supported}; "
                    f"got {init_in.type!r}."
                ),
            )
        if init_in.type == NodeKind.symmetrized_mlp_init and model_in.type != NodeKind.mlp_model:
            # 命名与文案都是 "MLP init";helper 会静默改写任何含 ≥2 个 Linear 的模型,
            # 钉住目标范围避免难解释的 silent mutation。
            raise HTTPException(
                status_code=400,
                detail=(
                    f"symmetrized_mlp_init requires an mlp_model on the model socket; "
                    f"got {model_in.type!r}."
                ),
            )
        initialization_node = init_in

    token_input_replace_prob = float(max(0.0, min(1.0, _scalar_float(td.get("inputTokenReplaceProb"), 0.0))))
    mse_input_noise_std = max(0.0, _scalar_float(td.get("mseInputNoiseStd"), 0.0))

    dd_train: dict[str, Any] = ds_train.data or {}
    train_size, default_test_size = _declared_dataset_split_sizes(ds_train, edges, nmap)
    # Unified ``dataset`` socket: train and test sizes both come from that node's ``data``.
    # Legacy: test loss is omitted unless a separate ``test_dataset`` edge was wired.
    if legacy_optional_test_wire and ds_test_raw is None:
        dd_test = dd_train
        test_size = 0
    elif ds_test_raw is not None:
        dd_test = ds_test_raw.data or {}
        _, test_size = _declared_dataset_split_sizes(ds_test_raw, edges, nmap)
    else:
        dd_test = dd_train
        if ds_train.type in _DATASET_MIXER_TYPES or ds_train.type == NodeKind.modular_addition_dataset:
            test_size = default_test_size
        else:
            test_size = 0
    if train_size < 1:
        raise HTTPException(status_code=400, detail="train size must be >= 1")
    if test_size < 0:
        raise HTTPException(status_code=400, detail="test size must be >= 0")
    train_streaming = _trainer_train_dataset_streaming(ds_train, dd_train)
    if train_batch_size > 0 and train_batch_size > train_size:
        raise HTTPException(
            status_code=400,
            detail=f"trainer batchSize ({train_batch_size}) cannot exceed train size ({train_size}).",
        )
    ref_batch_for_cycles = train_batch_size if train_batch_size > 0 else 128
    cyclic_lr_cycle_steps = 0
    if cyclic_lr_node is not None:
        cyclic_lr_cycle_steps = _cyclic_cycle_steps_from_node_data(
            cyclic_lr_node.data or {},
            train_size=train_size,
            default_ref_batch=ref_batch_for_cycles,
        )
    cyclic_batch_cycle_steps = 0
    if cyclic_batch_node is not None:
        cyclic_batch_cycle_steps = _cyclic_cycle_steps_from_node_data(
            cyclic_batch_node.data or {},
            train_size=train_size,
            default_ref_batch=ref_batch_for_cycles,
        )
        if cyclic_batch_max > train_size:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"cyclic_batch_schedule batchMax ({cyclic_batch_max}) cannot exceed "
                    f"train size ({train_size})."
                ),
            )
    cyclic_schedule_mode = "discrete_epoch"
    cyclic_cycle_epochs = 10
    cyclic_steps_per_epoch = steps_per_epoch(train_size, ref_batch_for_cycles)
    _cyclic_src = cyclic_lr_node or cyclic_batch_node
    if _cyclic_src is not None:
        _cyclic_data = _cyclic_src.data or {}
        cyclic_schedule_mode = _cyclic_schedule_mode_from_data(_cyclic_data)
        cyclic_cycle_epochs = _cyclic_cycle_epochs_from_node_data(_cyclic_data)
    training_length_mode = str(td.get("trainingLengthMode", "steps")).strip().lower()
    training_data_epochs = max(0, _scalar_int(td.get("trainingEpochs"), 0))
    if log_schedule == "idnns_logspace" and (
        training_length_mode != "epochs" or training_data_epochs < 1
    ):
        raise HTTPException(
            status_code=400,
            detail="idnns_logspace logging requires Trainer training length mode = epochs.",
        )
    if minibatch_sampling in {"epoch_shuffle", "affine_epoch"}:
        if train_streaming:
            raise HTTPException(
                status_code=400,
                detail=f"{minibatch_sampling} minibatch sampling requires a fixed training dataset.",
            )
        if cyclic_batch_node is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{minibatch_sampling} minibatch sampling cannot be combined "
                    "with cyclic_batch_schedule."
                ),
            )
    if minibatch_sampling == "affine_epoch" and (
        training_length_mode != "epochs" or training_data_epochs < 1
    ):
        raise HTTPException(
            status_code=400,
            detail="affine_epoch minibatch sampling requires Trainer training length mode = epochs.",
        )
    if training_length_mode == "epochs" and training_data_epochs > 0:
        if cyclic_batch_node is not None and cyclic_schedule_mode in ("discrete_epoch", "square_epoch"):
            training_steps = cbs_epochs_to_training_steps(
                training_data_epochs,
                train_size=train_size,
                batch_min=cyclic_batch_min,
                batch_max=cyclic_batch_max,
                cycle_length_epochs=cyclic_cycle_epochs,
                mode=cyclic_schedule_mode,
            )
        else:
            training_steps = training_data_epochs * cyclic_steps_per_epoch
        # epoch 推导出的 training_steps 同样过全局上限门。
        if training_steps < 1 or training_steps > MAX_TRAINING_STEPS:
            raise HTTPException(
                status_code=400,
                detail=f"training steps must be 1..{MAX_TRAINING_STEPS}",
            )
    if ds_train.type == NodeKind.dataset_mixer:
        seed = _scalar_int(dd_train.get("initSeed"), 0)
    elif ds_train.type in _TOY_LANGUAGE_TOKEN_DATASETS:
        seed = _scalar_int(dd_train.get("seed"), _scalar_int(dd_train.get("initSeed"), 0))
    elif ds_train.type in _VISION_DATASET_TYPES:
        seed = _scalar_int(dd_train.get("initSeed"), _scalar_int(dd_train.get("seed"), 0))
    else:
        seed = _scalar_int(dd_train.get("seed"), 0)

    # The dataset seed determines materialized examples. A trainer-level seed
    # changes only stochastic training behavior (minibatch order, augmentation,
    # dropout) so two branches can share an identical dataset subset.
    train_seed_override = _scalar_int(td.get("trainSeed"), -1)
    train_seed = train_seed_override if train_seed_override >= 0 else seed

    od: dict[str, Any] = opt_node.data or {}
    lr = _scalar_float(od.get("learningRate"), 1e-3)
    beta1 = _scalar_float(od.get("beta1"), 0.9)
    beta2 = _scalar_float(od.get("beta2"), 0.999)
    eps = _scalar_float(od.get("epsilon"), 1e-8)
    momentum = _scalar_float(od.get("momentum"), 0.0)
    muon_ns_steps = max(1, min(30, _scalar_int(od.get("newtonSchulzSteps"), 5)))
    weight_decay = _scalar_float(od.get("weightDecay"), 0.0)
    precondition_frequency = max(1, _scalar_int(od.get("preconditionFrequency"), 10))
    max_preconditioner_dim = max(1, _scalar_int(od.get("maxPreconditionerDim"), 1024))
    precondition_frequency = max(1, min(10_000, _scalar_int(od.get("preconditionFrequency"), 10)))
    max_preconditioner_dim = max(1, min(4096, _scalar_int(od.get("maxPreconditionerDim"), 1024)))

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(_trainer_master_rng_seed(train_seed, model_node))
    s.td = td
    s.training_steps = training_steps
    s.log_frequency = log_frequency
    s.log_schedule = log_schedule
    s.log_samples = log_samples
    s.log_aggregation = log_aggregation
    s.log_timing = log_timing
    s.test_evaluation = test_evaluation
    s.train_batch_size = train_batch_size
    s.minibatch_sampling = minibatch_sampling
    s.grad_clip_max_norm = grad_clip_max_norm
    s.lr_warmup_steps = lr_warmup_steps
    s.lr_schedule = lr_schedule
    s.cosine_lr_min_fraction = cosine_lr_min_fraction
    s.exponential_lr_decay_factor = exponential_lr_decay_factor
    s.exponential_lr_decay_epochs = exponential_lr_decay_epochs
    s.mup_embed_lr_mult = mup_embed_lr_mult
    s.mup_hidden_lr_mult = mup_hidden_lr_mult
    s.mup_output_lr_mult = mup_output_lr_mult
    s.use_mup_lr_schedule = use_mup_lr_schedule
    s.initialization_node = initialization_node
    s.train_size = train_size
    s.test_size = test_size
    s.dd_train = dd_train
    s.dd_test = dd_test
    s.train_streaming = train_streaming
    s.seed = seed
    s.train_seed = train_seed
    s.rng = rng
    s.lr = lr
    s.beta1 = beta1
    s.beta2 = beta2
    s.eps = eps
    s.momentum = momentum
    s.weight_decay = weight_decay
    s.muon_ns_steps = muon_ns_steps
    s.precondition_frequency = precondition_frequency
    s.max_preconditioner_dim = max_preconditioner_dim
    s.cyclic_lr_min = cyclic_lr_min
    s.cyclic_lr_max = cyclic_lr_max
    s.cyclic_lr_cycle_steps = cyclic_lr_cycle_steps
    s.cyclic_batch_min = cyclic_batch_min
    s.cyclic_batch_max = cyclic_batch_max
    s.cyclic_batch_cycle_steps = cyclic_batch_cycle_steps
    s.cyclic_schedule_mode = cyclic_schedule_mode
    s.cyclic_cycle_epochs = cyclic_cycle_epochs
    s.cyclic_steps_per_epoch = cyclic_steps_per_epoch
    s.training_data_epochs = training_data_epochs


def init_build_placeholders_stage(s: PrepareState) -> None:
    """Reads: ds_train, loss_node, trainer_task
    Writes: x_test_t, y_test_t, loss_d, loss_scale, ce_mem_a_slots, stack_io, streaming_teacher
    """
    ds_train = s.ds_train
    loss_node = s.loss_node
    trainer_task = s.trainer_task
    streaming_teacher: nn.Module | None = None

    depth: int
    x_t: torch.Tensor
    y_t: torch.Tensor
    x_test_t: torch.Tensor | None = None
    y_test_t: torch.Tensor | None = None
    model: nn.Module
    criterion: nn.Module
    loss_d: dict[str, Any] = loss_node.data or {}
    loss_scale = _scalar_float(loss_d.get("lossScale"), 1.0)
    ce_mem_a_slots = dense_ce_memorization_a_slot_groups(loss_d, trainer_task=trainer_task, ds_type=ds_train.type)
    if ce_mem_a_slots > 1 and ds_train.type == NodeKind.memorization_b_dataset:
        raise HTTPException(
            status_code=400,
            detail=(
                "cross_entropy_loss context slot masking (last context / custom with lossMaskContextLength>=2) "
                "is only supported with memorization_a_dataset: set dataset outputDim=V and model outputDim=T*V."
            ),
        )
    if ce_mem_a_slots > 1 and ds_train.type in _DATASET_MIXER_TYPES:
        raise HTTPException(
            status_code=400,
            detail="cross_entropy_loss slot masking is not supported with dataset_mixer / dataset_mixer_b.",
        )
    stack_io: tuple[int, int] | None = None
    s.x_test_t = x_test_t
    s.y_test_t = y_test_t
    s.loss_d = loss_d
    s.loss_scale = loss_scale
    s.ce_mem_a_slots = ce_mem_a_slots
    s.stack_io = stack_io
    s.streaming_teacher = streaming_teacher
