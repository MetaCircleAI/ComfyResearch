"""Quadratic Bezier mode-connectivity optimization on a pair of model checkpoints."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from fastapi import HTTPException
from torch.func import functional_call

from comfy_research.engine.analysis.parametric_path_sampler import (
    _PARAM_PATH_EVAL_BATCH,
    _blend_state_dicts,
    _checkpoint_b64,
    _recompute_batch_norm_stats,
    _state_dict_from_checkpoint_b64,
)
from comfy_research.engine.runs.trainer_run import (
    _batched_classification_accuracy,
    _batched_primary_loss_mean,
    _incoming,
    _node_map,
    _scalar_float,
    _scalar_int,
    prepare_trainer_run,
)
from comfy_research.engine.trainer.loss_terms import _trainer_primary_loss_tensor
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _bezier_state_dict(
    endpoint_a: dict[str, torch.Tensor],
    endpoint_b: dict[str, torch.Tensor],
    control: dict[str, torch.Tensor],
    t: float | torch.Tensor,
    *,
    trainable_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Keep endpoints fixed while making only trainable control tensors differentiable."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, device=next(iter(endpoint_a.values())).device)
    a_coeff = (1 - t) ** 2
    c_coeff = 2 * t * (1 - t)
    b_coeff = t**2
    state: dict[str, torch.Tensor] = {}
    for key, value_a in endpoint_a.items():
        value_b = endpoint_b[key]
        if key in trainable_keys:
            state[key] = a_coeff * value_a + c_coeff * control[key] + b_coeff * value_b
        elif torch.is_floating_point(value_a) and torch.is_floating_point(value_b):
            # BatchNorm buffers have no learned control point; interpolate the endpoint buffers.
            state[key] = (1 - t) * value_a + t * value_b
        else:
            state[key] = value_b
    return state


def _curve_metrics(
    *,
    model: torch.nn.Module,
    make_state: Callable[[float], dict[str, torch.Tensor]],
    alphas: np.ndarray,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    eval_batch_size: int,
    recompute_bn_stats: bool,
    bn_calibration_batches: int,
    trainer_task: Any,
    criterion: torch.nn.Module,
    loss_scale: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    train_losses: list[float] = []
    test_losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []
    model.eval()
    for alpha in alphas:
        model.load_state_dict(make_state(float(alpha)), strict=False)
        if recompute_bn_stats:
            _recompute_batch_norm_stats(
                model,
                x_train,
                batch_size=eval_batch_size,
                max_batches=bn_calibration_batches,
            )
        train_losses.append(_batched_primary_loss_mean(
            model, x_train, y_train, batch_size=eval_batch_size,
            trainer_task=trainer_task, criterion=criterion, loss_scale=loss_scale,
        ))
        test_losses.append(_batched_primary_loss_mean(
            model, x_test, y_test, batch_size=eval_batch_size,
            trainer_task=trainer_task, criterion=criterion, loss_scale=loss_scale,
        ))
        train_accs.append(_batched_classification_accuracy(
            model, x_train, y_train, batch_size=eval_batch_size, trainer_task=trainer_task,
        ))
        test_accs.append(_batched_classification_accuracy(
            model, x_test, y_test, batch_size=eval_batch_size, trainer_task=trainer_task,
        ))
    return train_losses, test_losses, train_accs, test_accs


def run_bezier_mode_connectivity(
    nodes: list[Node],
    edges: list[Edge],
    sampler_node_id: str,
) -> dict[str, Any]:
    nmap = _node_map(nodes)
    sampler = nmap.get(sampler_node_id)
    if sampler is None or sampler.type != NodeKind.observable_bezier_mode_connectivity:
        raise HTTPException(status_code=400, detail="Target is not a Bezier mode-connectivity evaluator.")
    data: dict[str, Any] = sampler.data or {}
    checkpoint_a = _incoming(edges, nmap, sampler_node_id, "checkpoint_a")
    checkpoint_b = _incoming(edges, nmap, sampler_node_id, "checkpoint_b")
    model_node = _incoming(edges, nmap, sampler_node_id, "model")
    dataset_node = _incoming(edges, nmap, sampler_node_id, "dataset")
    loss_node = _incoming(edges, nmap, sampler_node_id, "loss")
    if any(item is None for item in (checkpoint_a, checkpoint_b, model_node, dataset_node, loss_node)):
        raise HTTPException(status_code=400, detail="Connect checkpoints A/B, model, dataset, and loss before running Bezier connectivity.")
    assert checkpoint_a is not None and checkpoint_b is not None
    assert model_node is not None and dataset_node is not None and loss_node is not None
    if checkpoint_a.type != NodeKind.model_checkpoint or checkpoint_b.type != NodeKind.model_checkpoint:
        raise HTTPException(status_code=400, detail="Bezier connectivity requires two model_checkpoint nodes.")

    endpoint_a_cpu = _state_dict_from_checkpoint_b64(_checkpoint_b64(checkpoint_a.data or {}))
    endpoint_b_cpu = _state_dict_from_checkpoint_b64(_checkpoint_b64(checkpoint_b.data or {}))
    compute_device = str(data.get("computeDevice") or "auto").strip() or "auto"
    trainer = Node(
        id=f"{sampler_node_id}::__bezier_trainer",
        type=NodeKind.trainer,
        data={"trainingSteps": 1, "logFrequency": 1, "batchSize": -1, "computeDevice": compute_device},
    )
    optimizer = Node(
        id=f"{sampler_node_id}::__bezier_optimizer",
        type=NodeKind.adam_optimizer,
        data={"learningRate": 0.001},
    )
    run_nodes = [*nodes, trainer, optimizer]
    run_edges = [
        *edges,
        Edge(id=f"{sampler_node_id}::__bezier_model", source=model_node.id, sourceHandle="model", target=trainer.id, targetHandle="model"),
        Edge(id=f"{sampler_node_id}::__bezier_dataset", source=dataset_node.id, sourceHandle="dataset", target=trainer.id, targetHandle="dataset"),
        Edge(id=f"{sampler_node_id}::__bezier_optimizer_edge", source=optimizer.id, sourceHandle="optimizer", target=trainer.id, targetHandle="optimizer"),
        Edge(id=f"{sampler_node_id}::__bezier_loss", source=loss_node.id, sourceHandle="loss", target=trainer.id, targetHandle="loss"),
    ]
    context = prepare_trainer_run(run_nodes, run_edges, trainer.id)
    model = context.model
    device = next(model.parameters()).device
    endpoint_a = {key: value.to(device) for key, value in endpoint_a_cpu.items()}
    endpoint_b = {key: value.to(device) for key, value in endpoint_b_cpu.items()}
    trainable_keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable_keys <= endpoint_a.keys() or not trainable_keys <= endpoint_b.keys():
        raise HTTPException(status_code=400, detail="Checkpoint parameters do not match the supplied model architecture.")

    control = {
        key: torch.nn.Parameter((endpoint_a[key] + endpoint_b[key]).mul(0.5))
        for key in trainable_keys
    }
    curve_steps = max(1, _scalar_int(data.get("curveOptimizationSteps"), 500))
    samples_per_step = max(1, _scalar_int(data.get("curveSamplesPerStep"), 4))
    curve_batch_size = max(1, min(_scalar_int(data.get("curveBatchSize"), 256), int(context.x_t.shape[0])))
    curve_lr = max(1e-7, _scalar_float(data.get("curveLearningRate"), 0.01))
    curve_optimizer = torch.optim.Adam(control.values(), lr=curve_lr)
    random = torch.Generator(device=device)
    random.manual_seed(0)
    model.eval()
    for _ in range(curve_steps):
        indices = torch.randint(0, int(context.x_t.shape[0]), (curve_batch_size,), device=device, generator=random)
        x_batch = context.x_t.index_select(0, indices)
        y_batch = context.y_t.index_select(0, indices)
        sampled_t = torch.rand(samples_per_step, device=device, generator=random)
        curve_optimizer.zero_grad(set_to_none=True)
        losses = []
        for t in sampled_t:
            prediction = functional_call(
                model,
                _bezier_state_dict(endpoint_a, endpoint_b, control, t, trainable_keys=trainable_keys),
                (x_batch,),
            )
            losses.append(_trainer_primary_loss_tensor(
                prediction, y_batch, trainer_task=context.trainer_task,
                criterion=context.criterion, loss_scale=context.loss_scale,
            ))
        torch.stack(losses).mean().backward()
        curve_optimizer.step()

    alpha_steps = max(2, _scalar_int(data.get("alphaSteps"), 21))
    alphas = np.linspace(0.0, 1.0, alpha_steps, dtype=np.float64)
    eval_batch_size = _scalar_int(data.get("evalBatchSize"), 256) or _PARAM_PATH_EVAL_BATCH
    recompute_bn_stats = bool(data.get("recomputeBnStats", True))
    bn_calibration_batches = max(1, _scalar_int(data.get("bnCalibrationBatches"), 100))
    x_test = context.x_test_t if context.x_test_t is not None else context.x_t
    y_test = context.y_test_t if context.y_test_t is not None else context.y_t
    eval_batch_size = max(1, min(eval_batch_size, int(context.x_t.shape[0]), int(x_test.shape[0])))
    linear_state = lambda t: _blend_state_dicts(endpoint_a, endpoint_b, t, trainable_keys=trainable_keys, interpolate_buffers=True)
    bezier_state = lambda t: _bezier_state_dict(endpoint_a, endpoint_b, control, t, trainable_keys=trainable_keys)
    linear = _curve_metrics(
        model=model, make_state=linear_state, alphas=alphas, x_train=context.x_t, y_train=context.y_t,
        x_test=x_test, y_test=y_test, eval_batch_size=eval_batch_size,
        recompute_bn_stats=recompute_bn_stats, bn_calibration_batches=bn_calibration_batches,
        trainer_task=context.trainer_task, criterion=context.criterion, loss_scale=context.loss_scale,
    )
    bezier = _curve_metrics(
        model=model, make_state=bezier_state, alphas=alphas, x_train=context.x_t, y_train=context.y_t,
        x_test=x_test, y_test=y_test, eval_batch_size=eval_batch_size,
        recompute_bn_stats=recompute_bn_stats, bn_calibration_batches=bn_calibration_batches,
        trainer_task=context.trainer_task, criterion=context.criterion, loss_scale=context.loss_scale,
    )
    linear_loss_barrier = max(linear[1]) - max(linear[1][0], linear[1][-1])
    bezier_loss_barrier = max(bezier[1]) - max(bezier[1][0], bezier[1][-1])
    return {
        "alphaSeries": [float(alpha) for alpha in alphas],
        "linearTrainLoss": linear[0], "linearTestLoss": linear[1],
        "linearTrainAcc": linear[2], "linearTestAcc": linear[3],
        "bezierTrainLoss": bezier[0], "bezierTestLoss": bezier[1],
        "bezierTrainAcc": bezier[2], "bezierTestAcc": bezier[3],
        "linearLossBarrier": float(linear_loss_barrier),
        "bezierLossBarrier": float(bezier_loss_barrier),
        "summary": (
            f"Optimized a quadratic Bezier control point for {curve_steps} Adam steps "
            f"({samples_per_step} t samples/step) on {device}. Linear test loss barrier: "
            f"{linear_loss_barrier:.6g}; Bezier test loss barrier: {bezier_loss_barrier:.6g}."
        ),
    }
