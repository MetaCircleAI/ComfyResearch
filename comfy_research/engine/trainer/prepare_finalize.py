"""Prepare stages: build dispatch and finalization.

Stage order: materialize_dataset (unified four-way dispatch via
dataset_materialize) -> build_training_model (four-way model
dispatch) -> loop expansion -> payloads+muP -> optimizer/hessian-sizing/
user-observable validation -> resume history restore -> criterion -> device
placement -> materialize closures.
Prologue/body/epilogue convention as in prepare_graph; the materialize
closures capture stage-local aliases only, never PrepareState itself.
"""
from typing import Any, Literal
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import (
    build_sequential_from_atomic_tip,
    build_sequential_from_flat_atomic_chain,
    collect_atomic_layer_chain_front_to_back,
)
from comfy_research.engine.losses.loss_builders import LossCriterionContext, build_loss_criterion_for_node
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_from_type
from comfy_research.engine.models.model_loop_expand import ModelBlockLoop, expand_model_loop_stacking, resolve_loop_repeat_config
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.optimizers.idnns_init import apply_idnns_init
from comfy_research.engine.optimizers.mup_init import apply_mup_init
from comfy_research.engine.optimizers.saxe_init import apply_saxe_init
from comfy_research.engine.analysis.observable_user_eval import validate_observable_user_tensor_path
from comfy_research.engine.optimizers.optimizer_builders import OptimizerBuildConfig, build_optimizer_for_node
from comfy_research.engine.trainer.checkpoint import _apply_checkpoint_b64
from comfy_research.engine.trainer.dataset_helpers import (
    _resolve_linear_dataset_mlp_dims,
    _sample_base_vector_dataset_arrays,
)
from comfy_research.engine.trainer.device_runtime import (
    _optimizer_state_to_device,
    _resolve_trainer_compute_device,
    _trainer_configure_reproducible_torch,
    _trainer_master_rng_seed,
)
from comfy_research.engine.trainer.model_helpers import (
    _apply_mlp_output_scale,
    _apply_symmetrized_mlp_init,
    _forward_reg,
)
from comfy_research.engine.trainer.model_payloads import (
    apply_parameter_tensor_payloads_from_atomic_chain,
    apply_parameter_tensor_payloads_from_node,
)
from comfy_research.engine.trainer.observable_config import _trainer_disable_extra_observables
from comfy_research.engine.trainer.observable_metrics import HESSIAN_PARAM_LIMIT
from comfy_research.engine.datasets.vision_datasets_runtime import vision_num_classes
from comfy_research.engine.models.vision_models import infer_vision_input_channels_height_width
from comfy_research.engine.trainer.dataset_arrays import DatasetArrays
from comfy_research.engine.trainer.dataset_materialize import materialize_dataset_for_training
from comfy_research.engine.trainer.prepare_build_atomic import build_atomic_model
from comfy_research.engine.trainer.prepare_build_token import build_token_model
from comfy_research.engine.trainer.prepare_build_vector import build_vector_model
from comfy_research.engine.trainer.prepare_build_vision import build_vision_model
from comfy_research.engine.trainer.provider_types import (
    BuildBranch,
    DatasetMaterializeRequest,
    ModelBuildRequest,
    ModelProvider,
    _build_branch,
)
from comfy_research.engine.trainer.prepare_state import PrepareState
from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int, _scalar_str
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.trainer.teacher_helpers import _sample_teacher_input_tensor
from comfy_research.engine.trainer.user_observable_helpers import (
    _user_observable_definition_code,
    _user_observable_path_anchor,
)
from comfy_research.engine.datasets.trainer_dataset_streaming import rematerialize_train_tensors_for_step
from comfy_research.schemas.graph import Node, NodeKind


# Branch-level model provider registry, locked by
# test_provider_registries_have_exact_build_branch_keys.
MODEL_PROVIDERS: dict[BuildBranch, ModelProvider] = {
    "vision": build_vision_model,
    "atomic": build_atomic_model,
    "vector": build_vector_model,
    "token": build_token_model,
}


def materialize_dataset_stage(s: PrepareState) -> None:
    """Reads: ce_mem_a_slots, combined_flat_chain, dd_test, dd_train, ds_test_raw, ds_train, edges, is_sequential_atomic, model_node, nmap, nodes, rng, seed, streaming_teacher, test_size, train_size, trainer_task, validate_only, x_test_t, y_test_t
    Writes: dataset_arrays, streaming_teacher, test_size, x_t, x_test_t, y_t, y_test_t
    """
    ds_train = s.ds_train
    streaming_teacher = s.streaming_teacher
    test_size = s.test_size
    x_test_t = s.x_test_t
    y_test_t = s.y_test_t

    branch = _build_branch(
        s.trainer_task,
        s.is_sequential_atomic,
        image_diffusion=s.model_node.type == NodeKind.unet_ddpm_model,
    )
    if s.validate_only and branch == "vision":
        # validate_only: skip real vision data build/download; zeros stand in.
        # Batch ≥2 so BatchNorm shape probes do not fail in train mode.
        ch_v, h_v, w_v = infer_vision_input_channels_height_width(s.ds_train.type, s.dd_train)
        arrays = DatasetArrays(
            x_np=np.zeros((2, ch_v, h_v, w_v), dtype=np.float32),
            y_np=np.zeros((2,), dtype=np.int64),
            x_test_np=None,
            y_test_np=None,
            input_dim=ch_v,
            output_dim=vision_num_classes(s.ds_train.type, s.dd_train),
            extras={"image_size": h_v},
        )
    else:
        arrays = materialize_dataset_for_training(DatasetMaterializeRequest(
            branch=branch,
            binary_single_logit=s.loss_node.type == NodeKind.binary_cross_entropy_with_logits_loss,
            ce_mem_a_slots=s.ce_mem_a_slots,
            combined_flat_chain=s.combined_flat_chain,
            dd_test=s.dd_test,
            dd_train=s.dd_train,
            ds_test_raw=s.ds_test_raw,
            ds_train=s.ds_train,
            edges=s.edges,
            model_node=s.model_node,
            nmap=s.nmap,
            nodes=s.nodes,
            rng=s.rng,
            seed=s.seed,
            test_size=s.test_size,
            train_size=s.train_size,
            trainer_task=s.trainer_task,
        ))

    is_teacher = ds_train.type == NodeKind.teacher_dataset
    if is_teacher:
        streaming_teacher = arrays.extras["teacher"]
        test_size = int(arrays.x_test_np.shape[0]) if arrays.x_test_np is not None else 0

    # Tensor conversion performs no RNG and preserves each branch's conditions.
    # Dispatch uses the same split as the former trainer_task test.
    x_t = torch.from_numpy(arrays.x_np)
    if branch == "vision":
        y_t = torch.from_numpy(arrays.y_np).long()
        if test_size > 0 and arrays.x_test_np is not None and arrays.y_test_np is not None:
            x_test_t = torch.from_numpy(arrays.x_test_np)
            y_test_t = torch.from_numpy(arrays.y_test_np).long()
    elif branch in ("atomic", "vector"):
        y_t = torch.from_numpy(arrays.y_np)
        if is_teacher:
            if arrays.x_test_np is not None:
                x_test_t = torch.from_numpy(arrays.x_test_np)
                y_test_t = torch.from_numpy(arrays.y_test_np)
        elif test_size > 0:
            x_test_t = torch.from_numpy(arrays.x_test_np)
            y_test_t = torch.from_numpy(arrays.y_test_np)
    else:
        y_t = torch.from_numpy(arrays.y_np)
        if arrays.x_test_np is not None and arrays.y_test_np is not None:
            x_test_t = torch.from_numpy(arrays.x_test_np)
            y_test_t = torch.from_numpy(arrays.y_test_np)

    s.dataset_arrays = arrays
    s.streaming_teacher = streaming_teacher
    s.test_size = test_size
    s.x_t = x_t
    s.x_test_t = x_test_t
    s.y_t = y_t
    s.y_test_t = y_test_t


def build_training_model_stage(s: PrepareState) -> None:
    """Reads: ce_mem_a_slots, combined_flat_chain, dataset_arrays, ds_train, edges, is_sequential_atomic, model_node, nmap, seed, trainer_task, want_kan_reg
    Writes: depth, model, and stack_io. Criterion ownership remains with
    build_criterion_stage.
    """
    branch = _build_branch(
        s.trainer_task,
        s.is_sequential_atomic,
        image_diffusion=s.model_node.type == NodeKind.unet_ddpm_model,
    )
    result = MODEL_PROVIDERS[branch](ModelBuildRequest(
        branch=branch,
        arrays=s.dataset_arrays,
        ce_mem_a_slots=s.ce_mem_a_slots,
        combined_flat_chain=s.combined_flat_chain,
        ds_train=s.ds_train,
        edges=s.edges,
        model_node=s.model_node,
        nmap=s.nmap,
        seed=s.seed,
        want_kan_reg=s.want_kan_reg,
    ))
    s.depth = result.depth
    s.model = result.model
    if result.stack_io is not None:
        s.stack_io = result.stack_io


def expand_model_loop_stage(s: PrepareState) -> None:
    """Reads: combined_flat_chain, edges, is_sequential_atomic, model, model_in, model_node, nmap, observable_nodes, stack_io, train_streaming, trainer_task, want_kan_reg
    Writes: chain_for_payloads, loop_n, loop_share, model
    """
    combined_flat_chain = s.combined_flat_chain
    edges = s.edges
    is_sequential_atomic = s.is_sequential_atomic
    model = s.model
    model_in = s.model_in
    model_node = s.model_node
    nmap = s.nmap
    observable_nodes = s.observable_nodes
    stack_io = s.stack_io
    train_streaming = s.train_streaming
    trainer_task = s.trainer_task
    want_kan_reg = s.want_kan_reg
    chain_for_payloads: list[Node] | None = None
    if is_sequential_atomic:
        chain_for_payloads = combined_flat_chain
        if chain_for_payloads is None:
            chain_for_payloads = collect_atomic_layer_chain_front_to_back(model_node, edges, nmap)

    loop_n, loop_share = resolve_loop_repeat_config(model_in, model_node, nmap)
    if train_streaming and loop_n >= 2:
        raise HTTPException(
            status_code=400,
            detail="Streaming dataset sampling is not supported when model loop count is 2 or more.",
        )
    if loop_n >= 2:
        if trainer_task not in ("mse_regression", "cross_entropy_dense"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Loop count > 1 is only supported for MSE regression or memorization cross-entropy "
                    "(MLP, numeric transformer, numeric Hyena, MPP spatiotemporal ViT, KAN, or atomic chain) models."
                ),
            )
        if stack_io is None:
            raise HTTPException(
                status_code=400,
                detail="Loop count > 1 is not supported for this trainer configuration.",
            )
        in_d, out_d = stack_io
        if in_d != out_d:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Looped models require the graph input dimension to equal the output dimension "
                    f"(got input_dim={in_d}, output_dim={out_d})."
                ),
            )
        if any(on.type == "observable_relu_nonlinear_count" for on in observable_nodes):
            raise HTTPException(
                status_code=400,
                detail="The ReLU nonlinear observable is not supported when loop count is 2 or more.",
            )
        if want_kan_reg:
            raise HTTPException(
                status_code=400,
                detail="KAN regularization is not supported when loop count is 2 or more.",
            )
        rebuild_block: Callable[[], nn.Module] | None = None
        if not loop_share:
            if is_sequential_atomic:
                if combined_flat_chain is not None:
                    flat = combined_flat_chain
                    rebuild_block = lambda: build_sequential_from_flat_atomic_chain(flat)
                else:
                    rebuild_block = lambda: build_sequential_from_atomic_tip(model_node, edges, nmap)
            elif has_capability(model_node.type, "mlp_family"):
                md_l: dict[str, Any] = model_node.data or {}
                d_l = _scalar_int(md_l.get("depth"), 2)
                w_l = _scalar_int(md_l.get("width"), 64)
                k_l = _scalar_int(md_l.get("numExperts"), 4)
                a_l = _scalar_str(
                    md_l.get("activation"),
                    "silu" if has_capability(model_node.type, "silu_default_activation_model") else "relu",
                )
                in_l, out_l = stack_io
                rebuild_block = lambda: build_model_from_type(
                    model_node.type,
                    {
                        "depth": d_l,
                        "width": w_l,
                        "numExperts": k_l,
                        "activation": a_l,
                    },
                    ModelBuildContext(input_dim=in_l, output_dim=out_l),
                )
            elif model_node.type == NodeKind.numeric_transformer_model:
                md_l = dict(model_node.data or {})
                rebuild_block = lambda: build_model_from_type(model_node.type, md_l)
            elif model_node.type == NodeKind.numeric_hyena_model:
                md_l = dict(model_node.data or {})
                rebuild_block = lambda: build_model_from_type(model_node.type, md_l)
            elif model_node.type == NodeKind.mpp_spatiotemporal_model:
                md_l = dict(model_node.data or {})
                rebuild_block = lambda: build_model_from_type(model_node.type, md_l)
            elif model_node.type == NodeKind.afno_lite_spatiotemporal_model:
                md_l = dict(model_node.data or {})
                rebuild_block = lambda: build_model_from_type(model_node.type, md_l)
            elif model_node.type == NodeKind.kan_model:
                md_l = dict(model_node.data or {})
                in_l, out_l = stack_io
                rebuild_block = lambda: build_model_from_type(
                    model_node.type,
                    md_l,
                    ModelBuildContext(input_dim=in_l, output_dim=out_l, kan_fast_training=not want_kan_reg),
                )
        model = expand_model_loop_stacking(model, loop_n, loop_share, rebuild_block=rebuild_block)
    s.chain_for_payloads = chain_for_payloads
    s.loop_n = loop_n
    s.loop_share = loop_share
    s.model = model


def apply_payloads_and_mup_stage(s: PrepareState) -> None:
    """Reads: chain_for_payloads, initialization_node, is_sequential_atomic, loop_n, loop_share, model, model_node
    Writes: (mutates model in place)
    """
    chain_for_payloads = s.chain_for_payloads
    is_sequential_atomic = s.is_sequential_atomic
    loop_n = s.loop_n
    loop_share = s.loop_share
    model = s.model
    model_node = s.model_node
    initialization_node = s.initialization_node
    if is_sequential_atomic:
        if chain_for_payloads is None:
            raise HTTPException(
                status_code=500,
                detail="Internal: missing atomic chain while applying edited parameter payloads.",
            )
        payload_target: nn.Module = model.inner if isinstance(model, ModelBlockLoop) else model
        if (
            loop_n >= 2
            and not loop_share
            and isinstance(payload_target, nn.Sequential)
            and all(isinstance(child, nn.Sequential) for child in payload_target.children())
        ):
            for child in payload_target.children():
                apply_parameter_tensor_payloads_from_atomic_chain(child, chain_for_payloads)
        else:
            apply_parameter_tensor_payloads_from_atomic_chain(payload_target, chain_for_payloads)
    else:
        apply_parameter_tensor_payloads_from_node(model, model_node, strict_unresolved=True)

    if initialization_node is not None:
        if initialization_node.type == NodeKind.idnns_initialization:
            ind: dict[str, Any] = initialization_node.data or {}
            apply_idnns_init(model, seed=_scalar_int(ind.get("seed"), 0))
        elif initialization_node.type == NodeKind.mup_initialization:
            apply_mup_init(model)
        elif initialization_node.type == NodeKind.saxe_initialization:
            ind: dict[str, Any] = initialization_node.data or {}
            apply_saxe_init(model, amplitude=_scalar_float(ind.get("amplitude"), 0.01))
        elif initialization_node.type == NodeKind.symmetrized_mlp_init:
            _apply_symmetrized_mlp_init(model, initialization_node.data or {})

    _apply_mlp_output_scale(model, model_node.data or {})


def build_optimizer_stage(s: PrepareState) -> None:
    """Reads: beta1, beta2, edges, eps, get_user_observable_record, hessian_oversized_policy, lr, max_preconditioner_dim, model, momentum, muon_ns_steps, mup_embed_lr_mult, mup_hidden_lr_mult, mup_output_lr_mult, nodes, observable_nodes, opt_node, precondition_frequency, resume, td, use_mup_lr_schedule, weight_decay
    Writes: disable_extra_observables, hessian_oversized_mode, optimizer, optimizer_base_group_lrs
    """
    beta1 = s.beta1
    beta2 = s.beta2
    edges = s.edges
    eps = s.eps
    get_user_observable_record = s.get_user_observable_record
    hessian_oversized_policy = s.hessian_oversized_policy
    lr = s.lr
    max_preconditioner_dim = s.max_preconditioner_dim
    model = s.model
    momentum = s.momentum
    muon_ns_steps = s.muon_ns_steps
    mup_embed_lr_mult = s.mup_embed_lr_mult
    mup_hidden_lr_mult = s.mup_hidden_lr_mult
    mup_output_lr_mult = s.mup_output_lr_mult
    nodes = s.nodes
    observable_nodes = s.observable_nodes
    opt_node = s.opt_node
    precondition_frequency = s.precondition_frequency
    resume = s.resume
    td = s.td
    use_mup_lr_schedule = s.use_mup_lr_schedule
    weight_decay = s.weight_decay
    optimizer = build_optimizer_for_node(
        opt_node,
        model,
        OptimizerBuildConfig(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            muon_ns_steps=muon_ns_steps,
            precondition_frequency=precondition_frequency,
            max_preconditioner_dim=max_preconditioner_dim,
            use_mup_lr_schedule=use_mup_lr_schedule,
            mup_embed_lr_mult=mup_embed_lr_mult,
            mup_hidden_lr_mult=mup_hidden_lr_mult,
            mup_output_lr_mult=mup_output_lr_mult,
        ),
    )

    optimizer_base_group_lrs = tuple(float(g["lr"]) for g in optimizer.param_groups)

    will_resume = resume is not None and isinstance(resume, dict) and bool(resume.get("checkpoint_b64"))
    disable_extra_observables = _trainer_disable_extra_observables(td)
    has_hessian_obs = (
        not disable_extra_observables
        and any(on.type == NodeKind.observable_hessian_eigenvalues for on in observable_nodes)
    )
    n_trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    hessian_oversized_mode: Literal["off", "skip", "force"] = "off"
    if has_hessian_obs and n_trainable > HESSIAN_PARAM_LIMIT:
        if hessian_oversized_policy is None and not will_resume:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "hessian_oversized",
                    "n_params": n_trainable,
                    "limit": HESSIAN_PARAM_LIMIT,
                    "message": (
                        f"This graph has {n_trainable} trainable parameters; exact loss Hessian and its "
                        f"eigenvalues are intended for models up to {HESSIAN_PARAM_LIMIT} parameters "
                        "(cost grows as O(P²) in P). Choose Skip to train without Hessian work, or Continue "
                        "to compute Hessian eigenvalues anyway (may be very slow or run out of memory)."
                    ),
                },
            )
        if hessian_oversized_policy == "force":
            hessian_oversized_mode = "force"
        else:
            hessian_oversized_mode = "skip"

    for on in observable_nodes:
        if on.type == "observable_user":
            od_u: dict[str, Any] = on.data or {}
            if _user_observable_definition_code(od_u, get_user_observable_record=get_user_observable_record):
                continue
            anchor = _user_observable_path_anchor(od_u, get_user_observable_record=get_user_observable_record)
            if not anchor:
                uid = str(od_u.get("userObservableId") or "").strip()
                hint = ""
                if uid.startswith("grokking-demo-obs-"):
                    hint = (
                        " Grokking demo observables are missing from data/user_observables.json — "
                        "restart the server (auto-seeds bundled definitions) or run "
                        "npx tsx scripts/seed_grokking_physics_demo.ts."
                    )
                elif uid:
                    hint = f" Saved observable {uid!r} was not found in data/user_observables.json."
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "User observable needs a path anchor: tensorSelectorNodeId on the node, or "
                        "userObservableId pointing to a saved observable with definition_code in the repo, "
                        "or legacy tensorVizNodeId."
                        + hint
                    ),
                )
            validate_observable_user_tensor_path(nodes, edges, anchor)
    s.disable_extra_observables = disable_extra_observables
    s.hessian_oversized_mode = hessian_oversized_mode
    s.optimizer = optimizer
    s.optimizer_base_group_lrs = optimizer_base_group_lrs


def restore_histories_stage(s: PrepareState) -> None:
    """Restore regular, epoch, embedding, and attention-observable histories.

    Reads: observable_nodes, resume, training_data_epochs, training_steps.
    Writes: epoch_ticks, loss_history, observable_attention_slice_histories,
    observable_embedding_histories, observable_metric_histories, reg_loss_history,
    start_step, step_ticks, test_loss_history.
    """
    observable_nodes = s.observable_nodes
    resume = s.resume
    training_data_epochs = s.training_data_epochs
    training_steps = s.training_steps
    resuming = resume is not None and isinstance(resume, dict) and resume.get("checkpoint_b64")
    resume_ckpt_b64: str | None = str(resume["checkpoint_b64"]) if resuming else None
    if resuming:
        loss_history = [float(x) for x in (resume.get("loss_history") or [])]
        test_loss_history = [float(x) for x in (resume.get("test_loss_history") or [])]
        reg_loss_history = [float(x) for x in (resume.get("reg_loss_history") or [])]
        step_ticks = [int(x) for x in (resume.get("step_ticks") or [])]
        epoch_ticks = [float(x) for x in (resume.get("epoch_ticks") or [])]
        if len(epoch_ticks) != len(step_ticks):
            if training_data_epochs > 0 and step_ticks:
                legacy_epoch_axis = (
                    training_steps > training_data_epochs * 2
                    and abs(step_ticks[-1] - training_data_epochs) <= max(1, training_data_epochs // 100)
                )
                if legacy_epoch_axis:
                    epoch_ticks = [float(x) for x in step_ticks]
                    step_ticks = [round(x * training_steps / training_data_epochs) for x in step_ticks]
                else:
                    epoch_ticks = [x * training_data_epochs / training_steps for x in step_ticks]
            else:
                epoch_ticks = [float(x) for x in step_ticks]
        omh_raw = resume.get("observable_metric_histories") or {}
        observable_metric_histories = {
            str(k): [float(x) for x in (v or [])] for k, v in omh_raw.items() if isinstance(v, list)
        }
        oeh_raw = resume.get("observable_embedding_histories") or {}
        observable_embedding_histories = {
            str(k): [v for v in (hist or []) if isinstance(v, list)]
            for k, hist in oeh_raw.items()
            if isinstance(hist, list)
        }
        oash_raw = resume.get("observable_attention_slice_histories") or {}
        from comfy_research.engine.trainer.attention_map_history import sanitize_history
        observable_attention_slice_histories = {str(key): sanitize_history(frames) for key, frames in oash_raw.items()}
        for on in observable_nodes:
            observable_metric_histories.setdefault(on.id, [])
            observable_embedding_histories.setdefault(on.id, [])
            observable_attention_slice_histories.setdefault(on.id, [])
        for on in observable_nodes:
            if on.type == "observable_accuracy":
                observable_metric_histories.setdefault(f"{on.id}::test", [])
            if on.type == NodeKind.observable_activation_stats:
                observable_metric_histories.setdefault(f"{on.id}::std", [])
        start_step = int(resume.get("next_step", 0))
        if start_step < 0 or start_step > training_steps:
            raise HTTPException(status_code=400, detail="Invalid resume next_step")
    else:
        loss_history = []
        test_loss_history = []
        reg_loss_history = []
        step_ticks = []
        epoch_ticks = []
        observable_metric_histories: dict[str, list[float]] = {on.id: [] for on in observable_nodes}
        for on in observable_nodes:
            if on.type == "observable_accuracy":
                observable_metric_histories[f"{on.id}::test"] = []
            if on.type == NodeKind.observable_activation_stats:
                observable_metric_histories[f"{on.id}::std"] = []
        observable_embedding_histories = {on.id: [] for on in observable_nodes}
        observable_attention_slice_histories = {on.id: [] for on in observable_nodes}
        start_step = 0
    s.loss_history = loss_history
    s.observable_embedding_histories = observable_embedding_histories
    s.observable_attention_slice_histories = observable_attention_slice_histories
    s.observable_metric_histories = observable_metric_histories
    s.reg_loss_history = reg_loss_history
    s.resume_ckpt_b64 = resume_ckpt_b64
    s.resuming = resuming
    s.start_step = start_step
    s.step_ticks = step_ticks
    s.epoch_ticks = epoch_ticks
    s.test_loss_history = test_loss_history


def build_criterion_stage(s: PrepareState) -> None:
    """Reads: ce_mem_a_slots, loss_node, model, trainer_task, x_t, y_t
    Writes: criterion
    """
    ce_mem_a_slots = s.ce_mem_a_slots
    loss_node = s.loss_node
    model = s.model
    trainer_task = s.trainer_task
    x_t = s.x_t
    y_t = s.y_t
    if trainer_task == "mse_regression":
        if y_t.dim() == 2:
            y_flat_dim = int(y_t.shape[1])
        elif y_t.dim() == 3:
            y_flat_dim = int(y_t.shape[1] * y_t.shape[2])
        else:
            raise HTTPException(
                status_code=400,
                detail="MSE regression expects targets with shape [batch, features] or [batch, context, dim].",
            )
        criterion = build_loss_criterion_for_node(
            loss_node,
            LossCriterionContext(trainer_task=trainer_task, target_flat_dim=y_flat_dim),
        )
    elif trainer_task in ("cross_entropy_dense", "vision_classification"):
        if y_t.dim() != 1:
            raise HTTPException(
                status_code=400,
                detail="Cross-entropy classification expects 1D class labels with shape [batch] (long).",
            )
        if y_t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise HTTPException(
                status_code=400,
                detail="Cross-entropy classification expects integer class labels (long) for targets.",
            )
        with torch.no_grad():
            was_training = model.training
            model.eval()
            try:
                num_logits = int(_forward_reg(model, x_t[:1]).shape[1])
            finally:
                model.train(was_training)
        criterion = build_loss_criterion_for_node(
            loss_node,
            LossCriterionContext(trainer_task=trainer_task, num_logits=num_logits),
        )
        if ce_mem_a_slots > 1:
            v_slot = num_logits // ce_mem_a_slots
            if ce_mem_a_slots * v_slot != num_logits:
                raise HTTPException(
                    status_code=500,
                    detail="Internal: cross_entropy_dense slot mask vs model output dim mismatch.",
                )
            ymax = int(y_t.max().item()) if y_t.numel() > 0 else -1
            ymin = int(y_t.min().item()) if y_t.numel() > 0 else 0
            if ymin < 0 or ymax >= v_slot:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Slot-masked cross-entropy expects each target in [0, V-1] where V = model_output_dim / T "
                        f"(T=lossMaskContextLength). Here V={v_slot} but labels span [{ymin}, {ymax}]."
                    ),
                )
    elif trainer_task == "token_classification":
        criterion = build_loss_criterion_for_node(
            loss_node,
            LossCriterionContext(
                trainer_task=trainer_task,
                multi_token_targets=isinstance(model, MultiTokenTransformerModel),
            ),
        )
    elif trainer_task == "diffusion_noise":
        criterion = build_loss_criterion_for_node(
            loss_node,
            LossCriterionContext(trainer_task=trainer_task),
        )
    s.criterion = criterion


def place_on_device_stage(s: PrepareState) -> None:
    """Reads: criterion, model, model_node, optimizer, resume_ckpt_b64, seed, streaming_teacher, td, x_t, x_test_t, y_t, y_test_t
    Writes: criterion, device, master_rng_seed, model, streaming_teacher, x_t, x_test_t, y_t, y_test_t
    """
    criterion = s.criterion
    model = s.model
    model_node = s.model_node
    optimizer = s.optimizer
    resume_ckpt_b64 = s.resume_ckpt_b64
    train_seed = s.train_seed
    streaming_teacher = s.streaming_teacher
    td = s.td
    x_t = s.x_t
    x_test_t = s.x_test_t
    y_t = s.y_t
    y_test_t = s.y_test_t
    compute_spec = str(td.get("computeDevice") or "auto").strip() or "auto"
    device = _resolve_trainer_compute_device(compute_spec)
    master_rng_seed = _trainer_master_rng_seed(train_seed, model_node)
    _trainer_configure_reproducible_torch(device, rng_seed=master_rng_seed)
    model = model.to(device)
    if streaming_teacher is not None:
        streaming_teacher = streaming_teacher.to(device)
    x_t = x_t.to(device)
    y_t = y_t.to(device)
    if x_test_t is not None:
        x_test_t = x_test_t.to(device)
    if y_test_t is not None:
        y_test_t = y_test_t.to(device)
    criterion = criterion.to(device)

    if resume_ckpt_b64:
        try:
            _apply_checkpoint_b64(model, optimizer, resume_ckpt_b64, map_location=device)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid resume checkpoint: {e}") from e
        _optimizer_state_to_device(optimizer, device)
    s.criterion = criterion
    s.device = device
    s.master_rng_seed = master_rng_seed
    s.model = model
    s.streaming_teacher = streaming_teacher
    s.x_t = x_t
    s.x_test_t = x_test_t
    s.y_t = y_t
    s.y_test_t = y_test_t


def build_materialize_closures_stage(s: PrepareState) -> None:
    """Reads: ce_mem_a_slots, device, ds_test_raw, ds_train, edges, legacy_optional_test_wire, master_rng_seed, model_node, nmap, seed, streaming_teacher, td, test_size, train_streaming, trainer_task, x_t, x_test_t, y_t, y_test_t
    Writes: minibatch_perm_seed, test_materialize_fn, train_materialize_fn
    """
    ce_mem_a_slots = s.ce_mem_a_slots
    device = s.device
    ds_test_raw = s.ds_test_raw
    ds_train = s.ds_train
    edges = s.edges
    legacy_optional_test_wire = s.legacy_optional_test_wire
    master_rng_seed = s.master_rng_seed
    model_node = s.model_node
    nmap = s.nmap
    seed = s.seed
    streaming_teacher = s.streaming_teacher
    test_size = s.test_size
    td = s.td
    train_streaming = s.train_streaming
    trainer_task = s.trainer_task
    x_t = s.x_t
    x_test_t = s.x_test_t
    y_t = s.y_t
    y_test_t = s.y_test_t
    def _train_materialize_step(step_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not train_streaming:
            return x_t, y_t
        return rematerialize_train_tensors_for_step(
            step=step_idx,
            dataset_base_seed=seed,
            trainer_task=trainer_task,
            ds_train=ds_train,
            edges=edges,
            nmap=nmap,
            model_node=model_node,
            ds_test_raw=ds_test_raw,
            legacy_optional_test_wire=legacy_optional_test_wire,
            teacher_module=streaming_teacher,
            device=device,
            memorization_a_ce_slot_groups=ce_mem_a_slots,
            leaf_sampler=_sample_base_vector_dataset_arrays,
            teacher_input_sampler=_sample_teacher_input_tensor,
            resolve_mlp_dims=_resolve_linear_dataset_mlp_dims,
        )

    def _test_materialize_step(step_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if x_test_t is None or y_test_t is None or test_size <= 0:
            return None, None
        if not train_streaming:
            return x_test_t, y_test_t
        ds_eval = ds_test_raw if ds_test_raw is not None else ds_train
        return rematerialize_train_tensors_for_step(
            step=step_idx,
            dataset_base_seed=seed ^ 0xA5A5A5A5,
            trainer_task=trainer_task,
            ds_train=ds_eval,
            edges=edges,
            nmap=nmap,
            model_node=model_node,
            ds_test_raw=ds_eval,
            legacy_optional_test_wire=False,
            teacher_module=streaming_teacher,
            device=device,
            sample_size_override=test_size,
            memorization_a_ce_slot_groups=ce_mem_a_slots,
            leaf_sampler=_sample_base_vector_dataset_arrays,
            teacher_input_sampler=_sample_teacher_input_tensor,
            resolve_mlp_dims=_resolve_linear_dataset_mlp_dims,
        )

    train_materialize_fn = _train_materialize_step
    test_materialize_fn = _test_materialize_step
    minibatch_seed_override = _scalar_int(td.get("minibatchSeed"), -1)
    minibatch_perm_seed = (
        int(master_rng_seed) + 0x51ED7A77
        if minibatch_seed_override < 0
        else minibatch_seed_override
    )
    s.minibatch_perm_seed = minibatch_perm_seed
    s.test_materialize_fn = test_materialize_fn
    s.train_materialize_fn = train_materialize_fn
