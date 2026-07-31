"""Training loop: register/step/abort/pause/complete event generator.

Extracted verbatim from iter_trainer_events_from_context, which remains the
facade in trainer_run and delegates here after constructing the
ObservableRecorder. Event dict shapes and emission order are the frontend
contract and are preserved exactly.
"""
import base64
import io
import math
import time
from collections.abc import Iterator
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch
from torch.nn.utils import clip_grad_norm_

from comfy_research.engine.analysis.snapshot_schedules import idnns_epoch_snapshots
from comfy_research.engine.optimizers.cyclic_schedules import (
    cyclic_batch_for_data_epoch,
    cyclic_batch_for_step,
    cyclic_lr_for_data_epoch,
    cyclic_lr_for_step,
    data_epoch_state_for_cyclic_batch,
    steps_per_epoch,
)
from comfy_research.engine.models.diffusion_score_model import (
    DiffusionScoreMLP,
    diffusion_noise_mse_loss,
)
from comfy_research.engine.datasets.streaming_seed import streaming_train_step_seed
from comfy_research.engine.runs.train_control import get_control, register_trainer, unregister_trainer
from comfy_research.engine.trainer.checkpoint import _pack_checkpoint_b64
from comfy_research.engine.trainer.context import TrainerRunContext
from comfy_research.engine.trainer.dataset_helpers import (
    _cifar10_crop_flip_standardize,
)
from comfy_research.engine.trainer.eval_batches import (
    _batched_primary_loss_mean,
    _bounded_eval_batch_size,
)
from comfy_research.engine.trainer.loss_terms import (
    _apply_l2_weight_projection,
    _extra_loss_additions,
    _trainer_primary_loss_tensor,
    _weight_reg_loss_additions,
)
from comfy_research.engine.trainer.model_helpers import _forward_reg
from comfy_research.engine.trainer.minibatch_sampler import TrainerMinibatchSampler
from comfy_research.engine.trainer.observable_viz import (
    find_loss_visualization_targets,
    observable_viz_metric_updates,
)
from comfy_research.engine.trainer.recorder import ObservableRecorder


def _trainer_lr_mult_for_step(
    step: int,
    *,
    training_steps: int,
    warmup_steps: int,
    schedule: str,
    cosine_min_fraction: float,
    steps_per_epoch: int = 1,
    exponential_decay_factor: float = 0.95,
    exponential_decay_epochs: int = 1,
) -> float:
    ts = max(1, int(training_steps))
    w = max(0, int(warmup_steps))
    s = max(0, min(int(step), ts - 1))
    if w > 0 and s < w:
        warm = float(s + 1) / float(w)
    else:
        warm = 1.0
    sch = str(schedule).strip().lower().replace("-", "_")
    eta = float(max(0.0, min(1.0, cosine_min_fraction)))

    def _cosine_tail_mult(prog: float) -> float:
        p = float(max(0.0, min(1.0, prog)))
        return eta + (1.0 - eta) * 0.5 * (1.0 + math.cos(math.pi * p))

    if sch == "cosine":
        if s < w:
            return warm
        denom = max(1, ts - 1 - w)
        prog = float(s - w) / float(denom)
        return warm * _cosine_tail_mult(prog)
    if sch == "stable_stable_decay":
        if s < w:
            return warm
        n_post = ts - w
        if n_post <= 0:
            return warm
        l1 = n_post // 3
        l2 = n_post // 3
        start_decay = w + l1 + l2
        if s < start_decay:
            return warm
        denom_dec = max(1, (ts - 1) - start_decay)
        prog = float(s - start_decay) / float(denom_dec)
        return warm * _cosine_tail_mult(prog)
    if sch == "exponential_epoch":
        if s < w:
            return warm
        period = max(1, int(steps_per_epoch)) * max(1, int(exponential_decay_epochs))
        exponent = max(0, s - w) // period
        factor = float(max(0.0, min(1.0, exponential_decay_factor)))
        return warm * (factor**exponent)
    return warm


def _loss_plot_png_b64(
    step_ticks: list[int],
    loss_history: list[float],
    test_loss_history: list[float],
    test_size: int,
    reg_loss_history: list[float] | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 1.35), dpi=120)
    train_plot = loss_history
    if reg_loss_history and len(reg_loss_history) == len(loss_history):
        train_plot = [a + b for a, b in zip(loss_history, reg_loss_history)]
    ax.plot(step_ticks, train_plot, color="#c084fc", linewidth=1.6, label="train")
    if test_size > 0 and len(test_loss_history) == len(step_ticks):
        ax.plot(step_ticks, test_loss_history, color="#60a5fa", linewidth=1.4, label="test")
    # Match Training Viz SVG (~5% padding) so thumbnails align with the canvas chart.
    ax.margins(x=0.05, y=0.05)
    ax.set_xlabel("step", fontsize=8, color="#9a9aa8")
    ax.set_ylabel("loss", fontsize=8, color="#9a9aa8")
    ax.tick_params(colors="#9a9aa8", labelsize=7)
    ax.set_facecolor("#121218")
    fig.patch.set_facecolor("#1a1a1f")
    for spine in ax.spines.values():
        spine.set_color("#3a3a44")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")



def run_training_loop(ctx: TrainerRunContext, recorder: ObservableRecorder) -> Iterator[dict[str, Any]]:
    """register, step-0 baseline, the step loop, abort/pause/complete events.

    Body verbatim from iter_trainer_events_from_context; the ctx
    unpack serves as the prologue and record/kan_regs alias the recorder.
    """
    nodes = ctx.nodes
    edges = ctx.edges
    nmap = ctx.nmap
    trainer_node_id = ctx.trainer_node_id
    observable_nodes = ctx.observable_nodes
    model = ctx.model
    trainer_task = ctx.trainer_task
    criterion = ctx.criterion
    loss_scale = ctx.loss_scale
    optimizer = ctx.optimizer
    x_t = ctx.x_t
    y_t = ctx.y_t
    x_test_t = ctx.x_test_t
    y_test_t = ctx.y_test_t
    training_steps = ctx.training_steps
    log_frequency = ctx.log_frequency
    log_schedule = ctx.log_schedule
    log_samples = ctx.log_samples
    log_aggregation = ctx.log_aggregation
    log_timing = ctx.log_timing
    start_step = ctx.start_step
    loss_history = ctx.loss_history
    test_loss_history = ctx.test_loss_history
    reg_loss_history = ctx.reg_loss_history
    step_ticks = ctx.step_ticks
    epoch_ticks = ctx.epoch_ticks
    observable_metric_histories = ctx.observable_metric_histories
    observable_embedding_histories = ctx.observable_embedding_histories
    depth = ctx.depth
    test_size = ctx.test_size
    resuming = ctx.resuming
    hessian_oversized_mode = ctx.hessian_oversized_mode
    train_batch_size = ctx.train_batch_size
    minibatch_sampling = ctx.minibatch_sampling
    eval_batch_size = _bounded_eval_batch_size(train_batch_size)
    train_materialize = ctx.train_materialize
    test_materialize = ctx.test_materialize
    minibatch_perm_seed = ctx.minibatch_perm_seed
    grad_clip_max_norm = ctx.grad_clip_max_norm
    lr_warmup_steps = ctx.lr_warmup_steps
    lr_schedule = ctx.lr_schedule
    cosine_lr_min_fraction = ctx.cosine_lr_min_fraction
    exponential_lr_decay_factor = ctx.exponential_lr_decay_factor
    exponential_lr_decay_epochs = ctx.exponential_lr_decay_epochs
    optimizer_base_group_lrs = ctx.optimizer_base_group_lrs
    weight_reg_loss_nodes = ctx.weight_reg_loss_nodes
    l2_projection_nodes = ctx.l2_projection_nodes
    disable_extra_observables = ctx.disable_extra_observables
    train_size = ctx.train_size
    cyclic_lr_min = ctx.cyclic_lr_min
    cyclic_lr_max = ctx.cyclic_lr_max
    cyclic_lr_cycle_steps = ctx.cyclic_lr_cycle_steps
    cyclic_batch_min = ctx.cyclic_batch_min
    cyclic_batch_max = ctx.cyclic_batch_max
    cyclic_batch_cycle_steps = ctx.cyclic_batch_cycle_steps
    cyclic_schedule_mode = ctx.cyclic_schedule_mode
    cyclic_cycle_epochs = ctx.cyclic_cycle_epochs
    cyclic_steps_per_epoch = ctx.cyclic_steps_per_epoch
    training_data_epochs = ctx.training_data_epochs
    paper_cifar_recipe = ctx.training_recipe == "jastrzbski_fig1"
    paper_pixel_mean = (
        torch.as_tensor(ctx.training_recipe_pixel_mean, device=x_t.device, dtype=x_t.dtype)
        if paper_cifar_recipe
        else None
    )
    paper_global_std = ctx.training_recipe_global_std
    paper_run_seed = ctx.run_seed
    paper_epoch = -1
    paper_permutation: torch.Tensor | None = None
    paper_augmentation_generator = torch.Generator(device=x_t.device)
    paper_epoch_correct = 0
    paper_epoch_seen = 0
    epoch_schedule_modes = ("discrete_epoch", "square_epoch")
    use_cbs_data_epoch = cyclic_batch_cycle_steps > 0 and cyclic_schedule_mode in epoch_schedule_modes
    use_clr_data_epoch = (
        cyclic_lr_cycle_steps > 0
        and cyclic_batch_cycle_steps == 0
        and cyclic_schedule_mode in epoch_schedule_modes
    )
    step_batch_size = train_batch_size
    data_epoch_index = 0
    steps_in_data_epoch = 0
    if use_cbs_data_epoch:
        data_epoch_index, steps_in_data_epoch, step_batch_size = data_epoch_state_for_cyclic_batch(
            start_step,
            train_size=train_size,
            batch_min=cyclic_batch_min,
            batch_max=cyclic_batch_max,
            cycle_length_epochs=cyclic_cycle_epochs,
            mode=cyclic_schedule_mode,  # type: ignore[arg-type]
        )
    elif use_clr_data_epoch and start_step > 0:
        data_epoch_index = start_step // max(1, cyclic_steps_per_epoch)
        steps_in_data_epoch = start_step % max(1, cyclic_steps_per_epoch)

    generic_minibatch_sampler = TrainerMinibatchSampler(
        mode=minibatch_sampling,  # type: ignore[arg-type]
        train_size=train_size,
        steps_per_epoch=cyclic_steps_per_epoch,
        seed=minibatch_perm_seed,
    )
    idnns_snapshot_epochs = (
        set(
            map(
                int,
                idnns_epoch_snapshots(
                    training_data_epochs,
                    samples=log_samples,
                ).tolist(),
            )
        )
        if log_schedule == "idnns_logspace"
        else set()
    )
    log_interval_loss_sum = 0.0
    log_interval_sample_count = 0

    def _batch_for_step(s: int) -> int:
        nonlocal step_batch_size
        if use_cbs_data_epoch:
            return step_batch_size
        if cyclic_batch_cycle_steps > 0:
            step_batch_size = cyclic_batch_for_step(
                s,
                batch_min=cyclic_batch_min,
                batch_max=cyclic_batch_max,
                mode=cyclic_schedule_mode,  # type: ignore[arg-type]
                cycle_length_epochs=cyclic_cycle_epochs,
                cycle_length_steps=cyclic_batch_cycle_steps,
                steps_per_epoch=cyclic_steps_per_epoch,
            )
        else:
            step_batch_size = train_batch_size
        return step_batch_size

    def _lr_for_step(s: int) -> float | None:
        if cyclic_lr_cycle_steps <= 0:
            return None
        if use_clr_data_epoch:
            ep = s // max(1, cyclic_steps_per_epoch)
            return cyclic_lr_for_data_epoch(
                ep,
                lr_min=cyclic_lr_min,
                lr_max=cyclic_lr_max,
                cycle_length_epochs=cyclic_cycle_epochs,
                mode=cyclic_schedule_mode,  # type: ignore[arg-type]
            )
        if use_cbs_data_epoch:
            return cyclic_lr_for_data_epoch(
                data_epoch_index,
                lr_min=cyclic_lr_min,
                lr_max=cyclic_lr_max,
                cycle_length_epochs=cyclic_cycle_epochs,
                mode=cyclic_schedule_mode,  # type: ignore[arg-type]
            )
        return cyclic_lr_for_step(
            s,
            lr_min=cyclic_lr_min,
            lr_max=cyclic_lr_max,
            mode=cyclic_schedule_mode,  # type: ignore[arg-type]
            cycle_length_epochs=cyclic_cycle_epochs,
            cycle_length_steps=cyclic_lr_cycle_steps,
            steps_per_epoch=cyclic_steps_per_epoch,
        )

    record = recorder.record
    kan_regs = recorder.kan_regs

    total = training_steps
    register_trainer(trainer_node_id)
    try:
        if not resuming:
            yield {"type": "progress", "step": 0, "total": total}
            model.eval()
            with torch.no_grad():
                xf0, yf0 = train_materialize(0)
                xf0_eval = (
                    (xf0 - paper_pixel_mean) / paper_global_std
                    if paper_pixel_mean is not None
                    else xf0
                )
                base0 = _batched_primary_loss_mean(
                    model,
                    xf0_eval,
                    yf0,
                    batch_size=eval_batch_size,
                    trainer_task=trainer_task,
                    criterion=criterion,
                    loss_scale=loss_scale,
                )
                weight_extra0 = _weight_reg_loss_additions(model, weight_reg_loss_nodes)
                reg0 = float(weight_extra0.item()) if weight_extra0 is not None else 0.0
                record(0, base0, epoch=0.0, reg_train_val=reg0, x_log=xf0_eval, y_log=yf0)
        else:
            yield {"type": "progress", "step": start_step, "total": total}

        model.train()
        train_loop_compute_seconds = 0.0
        for step in range(start_step, training_steps):
            ctrl = get_control(trainer_node_id)
            if ctrl is not None and ctrl.abort_requested:
                yield {"type": "aborted"}
                return

            xf, yf = train_materialize(step)
            _batch_for_step(step)
            if paper_cifar_recipe:
                epoch_for_step = (
                    data_epoch_index
                    if use_cbs_data_epoch
                    else step // max(1, cyclic_steps_per_epoch)
                )
                step_in_epoch = (
                    steps_in_data_epoch
                    if use_cbs_data_epoch
                    else step % max(1, cyclic_steps_per_epoch)
                )
                if epoch_for_step != paper_epoch:
                    paper_epoch = epoch_for_step
                    epoch_seed = paper_run_seed * 1_000_003 + paper_epoch
                    torch.manual_seed(epoch_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(epoch_seed)
                    permutation_generator = torch.Generator(device=xf.device)
                    permutation_generator.manual_seed(epoch_seed)
                    paper_permutation = torch.randperm(
                        int(xf.shape[0]), generator=permutation_generator, device=xf.device
                    )
                    paper_augmentation_generator.manual_seed(epoch_seed)
                    paper_epoch_correct = 0
                    paper_epoch_seen = 0
                assert paper_permutation is not None and paper_pixel_mean is not None
                start = step_in_epoch * step_batch_size
                idx = paper_permutation[start : min(start + step_batch_size, int(xf.shape[0]))]
                xb = xf.index_select(0, idx)
                yb = yf.index_select(0, idx)
                xb = _cifar10_crop_flip_standardize(
                    xb,
                    paper_pixel_mean,
                    paper_global_std,
                    generator=paper_augmentation_generator,
                )
            else:
                xb, yb = generic_minibatch_sampler.take(
                    xf,
                    yf,
                    step=step,
                    batch_size=step_batch_size,
                )
            g_step = torch.Generator(device=xb.device)
            g_step.manual_seed(int(streaming_train_step_seed(minibatch_perm_seed, step)))

            if cyclic_lr_cycle_steps > 0:
                abs_lr = _lr_for_step(step)
                if abs_lr is not None:
                    # 保 param-group 比例(μP 组不同 base LR)。
                    # abs_lr 定义首组的绝对 LR,其余组按 base 比例缩放——
                    # 非 μP 时各组 base 相同,退化为原语义。
                    ref_base = float(optimizer_base_group_lrs[0]) if optimizer_base_group_lrs else 0.0
                    for gi, group in enumerate(optimizer.param_groups):
                        base_i = (
                            float(optimizer_base_group_lrs[gi])
                            if gi < len(optimizer_base_group_lrs)
                            else ref_base
                        )
                        ratio = (base_i / ref_base) if ref_base > 0 else 1.0
                        group["lr"] = abs_lr * ratio
            else:
                lr_mult = _trainer_lr_mult_for_step(
                    step,
                    training_steps=training_steps,
                    warmup_steps=lr_warmup_steps,
                    schedule=lr_schedule,
                    cosine_min_fraction=cosine_lr_min_fraction,
                    steps_per_epoch=cyclic_steps_per_epoch,
                    exponential_decay_factor=exponential_lr_decay_factor,
                    exponential_decay_epochs=exponential_lr_decay_epochs,
                )
                for gi, group in enumerate(optimizer.param_groups):
                    base_lr_i = (
                        float(optimizer_base_group_lrs[gi])
                        if gi < len(optimizer_base_group_lrs)
                        else float(group.get("lr", 1.0))
                    )
                    group["lr"] = base_lr_i * lr_mult

            t_step_compute = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if trainer_task == "diffusion_noise":
                dm = cast(DiffusionScoreMLP, model)
                base_loss = (
                    diffusion_noise_mse_loss(dm, xb, g_step, timesteps=int(dm.max_timesteps)) * loss_scale
                )
            else:
                pred = _forward_reg(model, xb)
                base_loss = _trainer_primary_loss_tensor(
                    pred,
                    yb,
                    trainer_task=trainer_task,
                    criterion=criterion,
                    loss_scale=loss_scale,
                )
                if paper_cifar_recipe and pred.dim() >= 2 and yb.dim() == 1:
                    paper_epoch_correct += int((pred.argmax(dim=-1) == yb.long()).sum().item())
                    paper_epoch_seen += int(yb.numel())
            if log_timing == "pre_update" and step > 0 and step % log_frequency == 0:
                weight_extra = _weight_reg_loss_additions(model, weight_reg_loss_nodes)
                reg_val = float(weight_extra.item()) if weight_extra is not None else 0.0
                record(
                    step,
                    float(base_loss.item()),
                    epoch=(
                        step / max(1, cyclic_steps_per_epoch)
                        if training_data_epochs > 0
                        else float(step)
                    ),
                    reg_train_val=reg_val,
                    x_log=xf,
                    y_log=yf,
                )
            extra = _extra_loss_additions(
                model, kan_regs=kan_regs, weight_reg_loss_nodes=weight_reg_loss_nodes
            )
            if extra is not None:
                loss = base_loss + extra
            else:
                loss = base_loss
            loss.backward()
            if grad_clip_max_norm > 0:
                clip_grad_norm_(model.parameters(), grad_clip_max_norm)
            optimizer.step()
            _apply_l2_weight_projection(model, l2_projection_nodes)
            train_loop_compute_seconds += time.perf_counter() - t_step_compute
            if log_aggregation == "interval_sample_mean":
                batch_sample_count = int(yb.shape[0]) if yb.dim() > 0 else 1
                log_interval_loss_sum += float(base_loss.detach().item()) * batch_sample_count
                log_interval_sample_count += batch_sample_count
            done_steps = step + 1
            should_log = False
            log_epoch = (
                done_steps / max(1, cyclic_steps_per_epoch)
                if training_data_epochs > 0
                else float(done_steps)
            )
            cbs_epochs_complete = False
            if log_timing == "pre_update":
                should_log = False
            elif use_cbs_data_epoch:
                steps_in_data_epoch += 1
                epoch_len = steps_per_epoch(train_size, step_batch_size)
                if steps_in_data_epoch >= epoch_len:
                    data_epoch_index += 1
                    steps_in_data_epoch = 0
                    step_batch_size = cyclic_batch_for_data_epoch(
                        data_epoch_index,
                        batch_min=cyclic_batch_min,
                        batch_max=cyclic_batch_max,
                        cycle_length_epochs=cyclic_cycle_epochs,
                        mode=cyclic_schedule_mode,  # type: ignore[arg-type]
                    )
                    should_log = True
                    log_epoch = float(data_epoch_index)
                    if training_data_epochs > 0 and data_epoch_index >= training_data_epochs:
                        cbs_epochs_complete = True
            elif use_clr_data_epoch:
                if done_steps % max(1, cyclic_steps_per_epoch) == 0:
                    should_log = True
                    log_epoch = done_steps / max(1, cyclic_steps_per_epoch)
            elif log_schedule == "idnns_logspace":
                epoch_steps = max(1, cyclic_steps_per_epoch)
                if (
                    done_steps % epoch_steps == 0
                    and int(done_steps // epoch_steps) in idnns_snapshot_epochs
                ):
                    should_log = True
            elif done_steps % log_frequency == 0:
                should_log = True
            if should_log:
                weight_extra = _weight_reg_loss_additions(model, weight_reg_loss_nodes)
                reg_val = float(weight_extra.item()) if weight_extra is not None else 0.0
                logged_base_loss = float(base_loss.item())
                if log_interval_sample_count > 0:
                    logged_base_loss = (
                        log_interval_loss_sum / log_interval_sample_count
                    )
                record(
                    done_steps,
                    logged_base_loss,
                    epoch=log_epoch,
                    metric_overrides=(
                        {"train_accuracy": paper_epoch_correct / paper_epoch_seen}
                        if paper_cifar_recipe and paper_epoch_seen > 0
                        else None
                    ),
                    reg_train_val=reg_val,
                    x_log=xf,
                    y_log=yf,
                )
                log_interval_loss_sum = 0.0
                log_interval_sample_count = 0
            yield {"type": "progress", "step": done_steps, "total": total}
            if should_log:
                observable_updates = observable_viz_metric_updates(
                    edges,
                    nmap,
                    trainer_node_id,
                    observable_metric_histories,
                    observable_embedding_histories,
                    recorder.observable_attention_slice_histories,
                )
                yield {
                    "type": "metrics",
                    "step": done_steps,
                    "loss_history": list(loss_history),
                    "test_loss_history": list(test_loss_history),
                    "reg_loss_history": list(reg_loss_history),
                    "step_ticks": list(step_ticks),
                    "epoch_ticks": list(epoch_ticks),
                    "observable_viz_updates": observable_updates,
                    "observable_metric_histories": dict(observable_metric_histories),
                }

            if cbs_epochs_complete:
                break

            ctrl = get_control(trainer_node_id)
            if ctrl is not None and ctrl.abort_requested:
                yield {"type": "aborted"}
                return
            if ctrl is not None and ctrl.pause_requested:
                plot_b64 = _loss_plot_png_b64(
                    step_ticks, loss_history, test_loss_history, test_size, reg_loss_history
                )
                viz_ids = find_loss_visualization_targets(edges, nodes, trainer_node_id)
                obs_updates = observable_viz_metric_updates(
                    edges,
                    nmap,
                    trainer_node_id,
                    observable_metric_histories,
                    observable_embedding_histories,
                    recorder.observable_attention_slice_histories,
                )
                yield {
                    "type": "paused",
                    "next_step": done_steps,
                    "checkpoint_b64": _pack_checkpoint_b64(model, optimizer),
                    "loss_history": loss_history,
                    "test_loss_history": test_loss_history,
                    "reg_loss_history": reg_loss_history,
                    "step_ticks": step_ticks,
                    "epoch_ticks": epoch_ticks,
                    "plot_png_base64": plot_b64,
                    "visualization_node_ids": viz_ids,
                    "observable_viz_updates": obs_updates,
                    "observable_metric_histories": observable_metric_histories,
                    "observable_embedding_histories": observable_embedding_histories,
                    "observable_attention_slice_histories": recorder.observable_attention_slice_histories,
                    "train_loop_seconds": train_loop_compute_seconds,
                }
                return

        plot_b64 = _loss_plot_png_b64(
            step_ticks, loss_history, test_loss_history, test_size, reg_loss_history
        )

        viz_ids = find_loss_visualization_targets(edges, nodes, trainer_node_id)
        obs_updates = observable_viz_metric_updates(
            edges,
            nmap,
            trainer_node_id,
            observable_metric_histories,
            observable_embedding_histories,
            recorder.observable_attention_slice_histories,
        )
        yield {
            "type": "complete",
            "checkpoint_b64": _pack_checkpoint_b64(model, optimizer),
            "loss_history": loss_history,
            "test_loss_history": test_loss_history,
            "reg_loss_history": reg_loss_history,
            "step_ticks": step_ticks,
            "epoch_ticks": epoch_ticks,
            "plot_png_base64": plot_b64,
            "visualization_node_ids": viz_ids,
            "observable_viz_updates": obs_updates,
            "observable_metric_histories": observable_metric_histories,
            "observable_embedding_histories": observable_embedding_histories,
            "observable_attention_slice_histories": recorder.observable_attention_slice_histories,
            "train_loop_seconds": train_loop_compute_seconds,
        }
    finally:
        unregister_trainer(trainer_node_id)
