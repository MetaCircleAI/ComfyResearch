"""Parametric path sampler: linear blend w(α) = w_sb + α(w_lb − w_sb), eval loss/accuracy series."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException

from comfy_research.engine.runs.trainer_run import (
    _batched_classification_accuracy,
    _batched_primary_loss_mean,
    _incoming,
    _node_map,
    _scalar_float,
    _scalar_int,
    prepare_trainer_run,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _checkpoint_b64(data: dict[str, Any]) -> str:
    src = str(data.get("checkpointSource") or "memory").strip().lower()
    ck_file = str(data.get("checkpoint_b64") or "").strip()
    ck_mem = str(data.get("memoryCheckpoint_b64") or "").strip()
    if src == "file" and ck_file:
        return ck_file
    if ck_file:
        return ck_file
    return ck_mem


def _state_dict_from_checkpoint_b64(b64: str) -> dict[str, torch.Tensor]:
    if not b64.strip():
        raise HTTPException(status_code=400, detail="Checkpoint node has no weights (train or load first).")
    import base64
    import io

    raw = base64.standard_b64decode(b64.encode("ascii"))
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    state = ckpt.get("model")
    if not isinstance(state, dict):
        raise HTTPException(status_code=400, detail="Checkpoint is missing model state_dict.")
    return state


def _blend_state_dicts(
    sb: dict[str, torch.Tensor],
    lb: dict[str, torch.Tensor],
    alpha: float,
    *,
    trainable_keys: set[str],
    interpolate_buffers: bool = False,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k in sb:
        if k not in lb:
            continue
        if k in trainable_keys:
            out[k] = sb[k] + alpha * (lb[k] - sb[k])
        elif interpolate_buffers and torch.is_floating_point(sb[k]) and torch.is_floating_point(lb[k]):
            out[k] = sb[k] + alpha * (lb[k] - sb[k])
        else:
            out[k] = lb[k]
    return out


def _recompute_batch_norm_stats(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    *,
    batch_size: int,
    max_batches: int,
) -> None:
    """Refresh BN running statistics without changing any interpolated weights."""
    batch_norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not batch_norms:
        return
    model.eval()
    for module in batch_norms:
        # Calibration must not retain statistics from either endpoint checkpoint.
        module.reset_running_stats()
        module.train()
    with torch.no_grad():
        for index in range(0, int(x_train.shape[0]), max(1, int(batch_size))):
            if index // max(1, int(batch_size)) >= max(1, int(max_batches)):
                break
            model(x_train[index : index + max(1, int(batch_size))])
    model.eval()


def _accuracy_drop_from_path(test_accs: list[float]) -> float:
    """Return the non-negative dip below the two endpoint accuracies."""
    return min(test_accs[0], test_accs[-1]) - min(test_accs)


def _interpolation_curve_png(
    alphas: np.ndarray,
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
) -> str:
    """Return a compact self-contained figure for the LMC result panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A vertical layout keeps both curves readable inside a narrow canvas node.
    figure, (loss_ax, acc_ax) = plt.subplots(2, 1, figsize=(6.4, 6.2), constrained_layout=True)
    loss_ax.plot(alphas, train_losses, label="train loss", color="#bd5b38")
    loss_ax.plot(alphas, test_losses, label="test loss", color="#236a88")
    loss_ax.set(xlabel="alpha", ylabel="cross-entropy loss", title="Interpolation loss")
    loss_ax.legend(fontsize=8)
    acc_ax.plot(alphas, train_accs, label="train accuracy", color="#bd5b38")
    acc_ax.plot(alphas, test_accs, label="test accuracy", color="#236a88")
    acc_ax.set(xlabel="alpha", ylabel="accuracy", title="Interpolation accuracy")
    acc_ax.legend(fontsize=8)
    for axis in (loss_ax, acc_ax):
        axis.grid(alpha=0.25)
    payload = io.BytesIO()
    figure.savefig(payload, format="png", dpi=150)
    plt.close(figure)
    return base64.b64encode(payload.getvalue()).decode("ascii")


_PARAM_PATH_EVAL_BATCH = 5000

# Injected temp trainer/optimizer node ids carry this marker; user graphs must not
# use it (nor the exact injected edge ids) or the temp subgraph could collide with /
# be fed by preexisting nodes and edges.
_PARAM_PATH_RESERVED_MARKER = "::__param_path"


def _reject_reserved_param_path_ids(
    nodes: list[Node],
    edges: list[Edge],
    sampler_node_id: str,
) -> None:
    injected_edge_ids = {
        f"{sampler_node_id}::__e_model",
        f"{sampler_node_id}::__e_ds",
        f"{sampler_node_id}::__e_opt",
        f"{sampler_node_id}::__e_loss",
    }

    def _is_reserved(value: str) -> bool:
        return _PARAM_PATH_RESERVED_MARKER in value or value in injected_edge_ids

    for n in nodes:
        if _is_reserved(str(n.id)):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Node id {n.id!r} collides with ids reserved by the parametric path sampler "
                    f"(ids containing {_PARAM_PATH_RESERVED_MARKER!r} are reserved)."
                ),
            )
    for e in edges:
        for value in (e.id, e.source, e.target):
            if _is_reserved(str(value)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Edge {e.id!r} references id {value!r}, which is reserved by the "
                        f"parametric path sampler (ids containing "
                        f"{_PARAM_PATH_RESERVED_MARKER!r} are reserved)."
                    ),
                )


def run_parametric_path_sampler(
    nodes: list[Node],
    edges: list[Edge],
    sampler_node_id: str,
) -> dict[str, Any]:
    nmap = _node_map(nodes)
    sampler = nmap.get(sampler_node_id)
    if sampler is None:
        raise HTTPException(status_code=404, detail="Parametric path sampler node not found.")
    if sampler.type == NodeKind.observable_bezier_mode_connectivity:
        from comfy_research.engine.analysis.bezier_mode_connectivity import run_bezier_mode_connectivity

        return run_bezier_mode_connectivity(nodes, edges, sampler_node_id)
    is_lmc = sampler.type == NodeKind.observable_linear_interpolation_barrier
    if sampler.type not in {NodeKind.parametric_path_sampler, NodeKind.observable_linear_interpolation_barrier}:
        raise HTTPException(status_code=400, detail="Target is not a linear interpolation evaluator.")

    _reject_reserved_param_path_ids(nodes, edges, sampler_node_id)

    sd: dict[str, Any] = sampler.data or {}
    alpha_min = _scalar_float(sd.get("alphaMin"), 0.0 if is_lmc else -1.0)
    alpha_max = _scalar_float(sd.get("alphaMax"), 1.0 if is_lmc else 2.0)
    alpha_steps = max(2, _scalar_int(sd.get("alphaSteps"), 21 if is_lmc else 50))
    recompute_bn_stats = bool(sd.get("recomputeBnStats", False)) if is_lmc else False
    bn_calibration_batches = max(1, _scalar_int(sd.get("bnCalibrationBatches"), 100))
    compute_device = str(sd.get("computeDevice") or "auto").strip() or "auto"

    checkpoint_a_handle = "checkpoint_a" if is_lmc else "checkpoint_sb"
    checkpoint_b_handle = "checkpoint_b" if is_lmc else "checkpoint_lb"
    checkpoint_a_label = "A" if is_lmc else "SB"
    checkpoint_b_label = "B" if is_lmc else "LB"
    ckpt_sb_node = _incoming(edges, nmap, sampler_node_id, checkpoint_a_handle)
    ckpt_lb_node = _incoming(edges, nmap, sampler_node_id, checkpoint_b_handle)
    model_node = _incoming(edges, nmap, sampler_node_id, "model")
    ds_node = _incoming(edges, nmap, sampler_node_id, "dataset")
    loss_node = _incoming(edges, nmap, sampler_node_id, "loss")

    if ckpt_sb_node is None or ckpt_lb_node is None:
        raise HTTPException(
            status_code=400,
            detail=f"Connect checkpoint {checkpoint_a_label} and {checkpoint_b_label} model_checkpoint nodes.",
        )
    if model_node is None:
        raise HTTPException(status_code=400, detail="Connect a model architecture node.")
    if ds_node is None:
        raise HTTPException(status_code=400, detail="Connect a dataset node.")
    if loss_node is None:
        raise HTTPException(status_code=400, detail="Connect a loss node.")

    for ck, label in ((ckpt_sb_node, checkpoint_a_label), (ckpt_lb_node, checkpoint_b_label)):
        if ck.type != NodeKind.model_checkpoint:
            raise HTTPException(status_code=400, detail=f"Checkpoint {label} must be a model_checkpoint node.")

    sb_state = _state_dict_from_checkpoint_b64(_checkpoint_b64(ckpt_sb_node.data or {}))
    lb_state = _state_dict_from_checkpoint_b64(_checkpoint_b64(ckpt_lb_node.data or {}))

    tmp_trainer = Node(
        id=f"{sampler_node_id}::__param_path_trainer",
        type=NodeKind.trainer,
        data={
            "trainingSteps": 1,
            "logFrequency": 1,
            "batchSize": -1,
            "computeDevice": compute_device,
        },
    )
    tmp_opt = Node(
        id=f"{sampler_node_id}::__param_path_opt",
        type=NodeKind.adam_optimizer,
        data={"learningRate": 0.001},
    )
    run_nodes = [*nodes, tmp_trainer, tmp_opt]
    run_edges = [
        *edges,
        Edge(
            id=f"{sampler_node_id}::__e_model",
            source=model_node.id,
            sourceHandle="model",
            target=tmp_trainer.id,
            targetHandle="model",
        ),
        Edge(
            id=f"{sampler_node_id}::__e_ds",
            source=ds_node.id,
            sourceHandle="dataset",
            target=tmp_trainer.id,
            targetHandle="dataset",
        ),
        Edge(
            id=f"{sampler_node_id}::__e_opt",
            source=tmp_opt.id,
            sourceHandle="optimizer",
            target=tmp_trainer.id,
            targetHandle="optimizer",
        ),
        Edge(
            id=f"{sampler_node_id}::__e_loss",
            source=loss_node.id,
            sourceHandle="loss",
            target=tmp_trainer.id,
            targetHandle="loss",
        ),
    ]

    ctx = prepare_trainer_run(run_nodes, run_edges, tmp_trainer.id)
    model = ctx.model
    criterion = ctx.criterion
    loss_scale = ctx.loss_scale
    trainer_task = ctx.trainer_task

    x_train, y_train = ctx.x_t, ctx.y_t
    x_test = ctx.x_test_t if ctx.x_test_t is not None else ctx.x_t
    y_test = ctx.y_test_t if ctx.y_test_t is not None else ctx.y_t
    configured_eval_bs = _scalar_int(sd.get("evalBatchSize"), 0) if is_lmc else 0
    eval_bs = configured_eval_bs or (ctx.train_batch_size if ctx.train_batch_size > 0 else _PARAM_PATH_EVAL_BATCH)
    eval_bs = max(1, min(eval_bs, int(x_train.shape[0]), int(x_test.shape[0])))
    device_str = str(next(model.parameters()).device)
    trainable_keys = {name for name, param in model.named_parameters() if param.requires_grad}

    alphas = np.linspace(alpha_min, alpha_max, alpha_steps, dtype=np.float64)
    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []
    model.eval()
    for alpha in alphas:
        model.load_state_dict(
            _blend_state_dicts(
                sb_state,
                lb_state,
                float(alpha),
                trainable_keys=trainable_keys,
                interpolate_buffers=is_lmc,
            ),
            strict=False,
        )
        if recompute_bn_stats:
            _recompute_batch_norm_stats(
                model,
                x_train,
                batch_size=eval_bs,
                max_batches=bn_calibration_batches,
            )
        train_losses.append(
            _batched_primary_loss_mean(
                model,
                x_train,
                y_train,
                batch_size=eval_bs,
                trainer_task=trainer_task,
                criterion=criterion,
                loss_scale=loss_scale,
                batch_norm_batch_stats=not is_lmc,
            )
        )
        test_losses.append(
            _batched_primary_loss_mean(
                model,
                x_test,
                y_test,
                batch_size=eval_bs,
                trainer_task=trainer_task,
                criterion=criterion,
                loss_scale=loss_scale,
                batch_norm_batch_stats=not is_lmc,
            )
        )
        train_accs.append(
            _batched_classification_accuracy(
                model,
                x_train,
                y_train,
                batch_size=eval_bs,
                trainer_task=trainer_task,
                batch_norm_batch_stats=not is_lmc,
            )
        )
        test_accs.append(
            _batched_classification_accuracy(
                model,
                x_test,
                y_test,
                batch_size=eval_bs,
                trainer_task=trainer_task,
                batch_norm_batch_stats=not is_lmc,
            )
        )

    series = [
        {"metricId": "train_loss", "label": "train loss", "values": train_losses},
        {"metricId": "test_loss", "label": "test loss", "values": test_losses},
        {"metricId": "train_acc", "label": "train acc", "values": train_accs},
        {"metricId": "test_acc", "label": "test acc", "values": test_accs},
    ]
    if is_lmc:
        loss_barrier = max(test_losses) - max(test_losses[0], test_losses[-1])
        accuracy_drop = _accuracy_drop_from_path(test_accs)
        bn_note = (
            f"recomputed BatchNorm statistics using {bn_calibration_batches} train batches"
            if recompute_bn_stats
            else "used linearly interpolated floating BatchNorm buffers"
        )
        summary = (
            f"Evaluated {len(alphas)} same-init linear interpolation points on {device_str}; {bn_note}. "
            f"Test loss barrier: {loss_barrier:.6g}; accuracy drop: {accuracy_drop:.6g}."
        )
    else:
        summary = (
            f"Sampled {len(alphas)} α ∈ [{alpha_min:g}, {alpha_max:g}] on {device_str} "
            "with trainable-parameter interpolation and per-batch BatchNorm statistics "
            f"(train/test loss & acc, {len(alphas)} points × 4 series)."
        )
    result = {
        "alphaSeries": [float(a) for a in alphas],
        "series": series,
        # Legacy single-series fields (test loss) for older clients.
        "valueSeries": test_losses,
        "seriesLabel": "test loss",
        "metric": "loss",
        "split": "test",
        "summary": summary,
    }
    if is_lmc:
        result.update(
            {
                "alpha": [float(alpha) for alpha in alphas],
                "train_loss": train_losses,
                "test_loss": test_losses,
                "train_acc": train_accs,
                "test_acc": test_accs,
                "lossBarrier": float(loss_barrier),
                "accuracyDrop": float(accuracy_drop),
                "loss_barrier": float(loss_barrier),
                "accuracy_drop": float(accuracy_drop),
                "interpolationCurvePng": _interpolation_curve_png(
                    alphas, train_losses, test_losses, train_accs, test_accs
                ),
            }
        )
    return result
