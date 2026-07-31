"""Observable metric computations and payload builders (extracted from trainer_run)."""

import math
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.attention_only_model import AttentionTokenPredictBundle
from comfy_research.engine.losses.loss_builders import TrainerTask
from comfy_research.engine.models.model_loop_expand import ModelBlockLoop
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.models.numeric_transformer_model import NumericTransformerModel
from comfy_research.engine.analysis.tensor_metrics import effective_rank_from_matrix
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel
from comfy_research.engine.trainer.loss_terms import _extra_loss_additions, _trainer_primary_loss_tensor
from comfy_research.engine.trainer.model_helpers import _forward_reg
from comfy_research.engine.trainer.observable_config import (
    _ACTIVATION_STATS_LAYER_SUBMOD_RE,
    _activation_stats_bucket_from_module_name,
)
from comfy_research.schemas.graph import Node

HESSIAN_PARAM_LIMIT = 2048


HESSIAN_FORCE_MAX_PARAMS = 50_000_000


SPECTRAL_POWER_MAX_ELEMS = 67_108_864


def _observable_fourier_metric(
    data: dict[str, Any] | None,
) -> Literal["relative_projection_mse", "amplitude_ratio"]:
    """Read the supported Fourier metric, falling back to component MSE."""
    raw_value = (data or {}).get("metric")
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else None
    return "amplitude_ratio" if str(raw_value or "").strip() == "amplitude_ratio" else "relative_projection_mse"


def _batch_feature_column(t: torch.Tensor, index: int) -> torch.Tensor | None:
    """Return a batch-aligned flattened feature column, without clamping indexes."""
    if t.dim() == 0 or int(t.shape[0]) <= 0 or int(index) < 0:
        return None
    rows = t.reshape(-1, 1) if t.dim() == 1 else t.reshape(t.shape[0], -1)
    if int(index) >= int(rows.shape[1]):
        return None
    return rows[:, int(index)].detach().float()


def _fourier_component_observable_value(
    x: torch.Tensor,
    y: torch.Tensor,
    pred: torch.Tensor,
    *,
    frequency: float,
    metric: Literal["relative_projection_mse", "amplitude_ratio"],
    input_axis: int,
    output_index: int,
) -> float:
    """Measure one sinusoidal target component; failures are non-fatal NaNs."""
    coord = _batch_feature_column(x, input_axis)
    target = _batch_feature_column(y, output_index)
    prediction = _batch_feature_column(pred, output_index)
    if coord is None or target is None or prediction is None:
        return float("nan")
    if (
        int(coord.shape[0]) < 2
        or int(target.shape[0]) != int(coord.shape[0])
        or int(prediction.shape[0]) != int(coord.shape[0])
        or not math.isfinite(float(frequency))
        or float(frequency) < 0.0
    ):
        return float("nan")
    try:
        coord64, target64, pred64 = coord.double(), target.double(), prediction.double()
        phase = 2.0 * math.pi * float(frequency) * coord64
        basis = torch.stack((torch.sin(phase), torch.cos(phase)), dim=1)
        gram = basis.T @ basis
        ridge = torch.eye(2, dtype=gram.dtype, device=gram.device) * 1e-12
        coef_target = torch.linalg.solve(gram + ridge, basis.T @ target64)
        coef_pred = torch.linalg.solve(gram + ridge, basis.T @ pred64)
        target_amplitude = torch.linalg.vector_norm(coef_target)
        target_amplitude_value = float(target_amplitude.item())
        if metric == "amplitude_ratio":
            if target_amplitude_value <= 1e-12 or not math.isfinite(target_amplitude_value):
                return float("nan")
            return float((torch.linalg.vector_norm(coef_pred) / target_amplitude).item())
        target_projection, prediction_projection = basis @ coef_target, basis @ coef_pred
        denominator = torch.mean(target_projection.pow(2))
        denominator_value = float(denominator.item())
        if denominator_value <= 1e-12 or not math.isfinite(denominator_value):
            return float("nan")
        return float((torch.mean((prediction_projection - target_projection).pow(2)) / denominator).item())
    except Exception:
        return float("nan")


def _observable_multi_series_l2_payload_from_hist(
    paired: str,
    metric_histories: dict[str, list[float]],
    *,
    breakdown: Literal["top_level_module", "tensor"],
) -> dict[str, Any] | None:
    hist_g = metric_histories.get(paired)
    if not hist_g:
        return None
    tag = "top" if breakdown == "top_level_module" else "tensor"
    prefix = f"{paired}::{tag}::"
    subs = sorted(
        k[len(prefix) :] for k in metric_histories if isinstance(k, str) and k.startswith(prefix)
    )
    series_g: list[list[float]] = [[float(x) for x in hist_g]]
    labels_g = ["global"] + subs
    ok = True
    for sn in subs:
        row = metric_histories.get(f"{paired}::{tag}::{sn}") or []
        if len(row) != len(hist_g):
            ok = False
            break
        series_g.append([float(x) for x in row])
    if not ok or not series_g:
        return None
    return {"value_histories": series_g, "series_labels": labels_g}


def _activation_stats_layer_series_payload_from_hist(
    paired: str,
    metric_histories: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Multi-series activation stats: alternating ``layer_mean`` / ``layer_std`` rows per bucket."""
    hist_ref = metric_histories.get(paired) or []
    ref_len = len(hist_ref)
    if ref_len <= 0:
        return None
    prefix_m = f"{paired}::layer_mean::"

    def _seg_sort_key(seg: str) -> tuple[int, int, str]:
        if seg == "rest":
            return (2, 0, "")
        if seg.isdigit():
            return (0, int(seg), "")
        return (1, 0, seg)

    subs = sorted(
        (k[len(prefix_m) :] for k in metric_histories if isinstance(k, str) and k.startswith(prefix_m)),
        key=_seg_sort_key,
    )
    if not subs:
        return None
    vh: list[list[float]] = []
    labels: list[str] = []
    for seg in subs:
        rm = metric_histories.get(f"{paired}::layer_mean::{seg}") or []
        rs = metric_histories.get(f"{paired}::layer_std::{seg}") or []
        if len(rm) != ref_len or len(rs) != ref_len:
            return None
        base_lbl = "Other modules" if seg == "rest" else f"Layer {seg}"
        vh.append([float(x) for x in rm])
        vh.append([float(x) for x in rs])
        labels.append(f"{base_lbl} mean")
        labels.append(f"{base_lbl} std")
    return {"value_histories": vh, "series_labels": labels}


def _paired_layer_series_payload_from_hist(
    paired: str,
    metric_histories: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Multi-series metrics stored as ``paired::layer::<i>`` (same layout as sink attention all-layers)."""
    prefix = f"{paired}::layer::"
    subs: list[str] = sorted(
        (k[len(prefix) :] for k in metric_histories if isinstance(k, str) and k.startswith(prefix)),
        key=lambda s: int(s) if s.isdigit() else 0,
    )
    hist_ref = metric_histories.get(paired) or []
    ref_len = len(hist_ref)
    if ref_len <= 0 or not subs:
        return None
    rows: list[list[float]] = []
    labels: list[str] = []
    for seg in subs:
        row = metric_histories.get(f"{paired}::layer::{seg}") or []
        if len(row) != ref_len:
            return None
        rows.append([float(x) for x in row])
        labels.append(f"Layer {seg}")
    if not rows:
        return None
    return {"value_histories": rows, "series_labels": labels}


def _paired_member_series_payload_from_hist(
    paired: str,
    metric_histories: dict[str, list[float]],
) -> dict[str, Any] | None:
    """Multi-series algebra observables stored as ``paired::member::<key>``."""
    from comfy_research.engine.analysis.observable_algebra import member_display_label

    prefix = f"{paired}::member::"
    subs: list[str] = sorted(
        k[len(prefix) :] for k in metric_histories if isinstance(k, str) and k.startswith(prefix)
    )
    hist_ref = metric_histories.get(paired) or []
    ref_len = len(hist_ref)
    if ref_len <= 0 or not subs:
        return None
    rows: list[list[float]] = []
    labels: list[str] = []
    for seg in subs:
        row = metric_histories.get(f"{paired}::member::{seg}") or []
        if len(row) != ref_len:
            return None
        rows.append([float(x) for x in row])
        labels.append(member_display_label(seg))
    if not rows:
        return None
    return {"value_histories": rows, "series_labels": labels}


def _activation_mean_std_bucketed(model: nn.Module, x_batch: torch.Tensor) -> dict[str, tuple[float, float]]:
    """Per bucket: mean/std of activations averaged over hooked Linear/Conv outputs in that bucket.

    Buckets are submodule indices matching ``.layers.{i}.`` in parameter paths (encoder blocks); all other
    hooked modules share bucket ``rest``.
    """
    bucket_means: defaultdict[str, list[float]] = defaultdict(list)
    bucket_stds: defaultdict[str, list[float]] = defaultdict(list)
    hooks: list[Any] = []

    def _make_hook(bucket: str):
        def _hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            if not isinstance(out, torch.Tensor) or out.numel() == 0:
                return
            with torch.no_grad():
                bucket_means[bucket].append(float(out.mean().item()))
                bucket_stds[bucket].append(float(out.std(unbiased=False).item()))

        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            bkey = _activation_stats_bucket_from_module_name(name)
            hooks.append(mod.register_forward_hook(_make_hook(bkey)))
    was_training = model.training
    model.train()
    try:
        _forward_reg(model, x_batch)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    out: dict[str, tuple[float, float]] = {}
    for bkey in bucket_means:
        ms = bucket_means[bkey]
        ss = bucket_stds.get(bkey) or []
        if not ms or len(ms) != len(ss):
            continue
        out[bkey] = (sum(ms) / len(ms), sum(ss) / len(ss))
    return out


def _activation_mean_std_averages(model: nn.Module, x_batch: torch.Tensor) -> tuple[float, float]:
    """Average bucket statistics with equal weight per bucket (see :func:`_activation_mean_std_bucketed`)."""
    buckets = _activation_mean_std_bucketed(model, x_batch)
    if not buckets:
        return float("nan"), float("nan")
    return (
        sum(t[0] for t in buckets.values()) / len(buckets),
        sum(t[1] for t in buckets.values()) / len(buckets),
    )


def _activation_norm_mean_and_outlier_ratio(model: nn.Module, x_batch: torch.Tensor) -> tuple[float, float]:
    """Mean of per-module ``||x||_2`` averaged over last dim, and global max|·|/mean|·| over hooked activations."""
    norms: list[float] = []
    max_abs = 0.0
    sum_abs = 0.0
    n_el = 0
    hooks: list[Any] = []

    def _hook(_m: nn.Module, _inp: Any, out: Any) -> None:
        nonlocal max_abs, sum_abs, n_el
        if not isinstance(out, torch.Tensor) or out.numel() == 0:
            return
        with torch.no_grad():
            t = out.detach().float()
            if t.dim() >= 2:
                norms.append(float(t.norm(dim=-1).mean().item()))
            a = t.abs()
            max_abs = max(max_abs, float(a.max().item()))
            sum_abs += float(a.sum().item())
            n_el += int(a.numel())

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(m.register_forward_hook(_hook))
    was_training = model.training
    model.train()
    try:
        _forward_reg(model, x_batch)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    if not norms or n_el <= 0:
        return float("nan"), float("nan")
    mean_norm = sum(norms) / len(norms)
    mean_abs = sum_abs / float(n_el)
    ratio = max_abs / (mean_abs + 1e-12)
    return mean_norm, ratio


def _activation_norm_mean_and_outlier_per_bucket(
    model: nn.Module, x_batch: torch.Tensor
) -> dict[str, tuple[float, float]]:
    """Per activation-stats bucket: (mean L2 along last dim, max|·|/mean|·|) over hooked Linear/Conv in that bucket."""
    norms_by: defaultdict[str, list[float]] = defaultdict(list)
    max_abs_by: defaultdict[str, float] = defaultdict(float)
    sum_abs_by: defaultdict[str, float] = defaultdict(float)
    n_el_by: defaultdict[str, int] = defaultdict(int)
    hooks: list[Any] = []

    def _make_hook(bucket: str):
        def _hook(_m: nn.Module, _inp: Any, out: Any) -> None:
            if not isinstance(out, torch.Tensor) or out.numel() == 0:
                return
            with torch.no_grad():
                t = out.detach().float()
                if t.dim() >= 2:
                    norms_by[bucket].append(float(t.norm(dim=-1).mean().item()))
                a = t.abs()
                max_abs_by[bucket] = max(max_abs_by[bucket], float(a.max().item()))
                sum_abs_by[bucket] += float(a.sum().item())
                n_el_by[bucket] += int(a.numel())

        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            bkey = _activation_stats_bucket_from_module_name(name)
            hooks.append(mod.register_forward_hook(_make_hook(bkey)))
    was_training = model.training
    model.train()
    try:
        _forward_reg(model, x_batch)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    out: dict[str, tuple[float, float]] = {}
    for bkey, nlist in norms_by.items():
        if not nlist:
            continue
        ne = n_el_by.get(bkey, 0)
        if ne <= 0:
            continue
        mean_norm = sum(nlist) / len(nlist)
        mean_abs = sum_abs_by[bkey] / float(ne)
        ratio = max_abs_by[bkey] / (mean_abs + 1e-12)
        out[bkey] = (mean_norm, ratio)
    return out


def _effective_rank_max_per_encoder_layer(module: nn.Module) -> dict[str, float]:
    """Largest effective rank among not-too-large 2D parameters in each ``.layers.{i}.`` bucket."""
    mats: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for name, p in module.named_parameters():
        if p.dim() != 2 or int(p.numel()) > SPECTRAL_POWER_MAX_ELEMS:
            continue
        m = _ACTIVATION_STATS_LAYER_SUBMOD_RE.search(name)
        if not m:
            continue
        key = str(int(m.group(1)))
        mats[key].append(p.detach().float().cpu().numpy())
    out: dict[str, float] = {}
    for key in sorted(mats.keys(), key=lambda s: int(s) if s.isdigit() else 0):
        best = float("nan")
        for arr in mats[key]:
            er = float(effective_rank_from_matrix(arr))
            if not math.isnan(er) and (math.isnan(best) or er > best):
                best = er
        out[key] = best
    return out


def _layer_bucket_flat_concat_vectors(module: nn.Module) -> dict[str, np.ndarray]:
    """Concatenate flattened 2D parameters per ``.layers.{i}.`` bucket (sorted by parameter name)."""
    parts: defaultdict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for name, p in module.named_parameters():
        if p.dim() != 2:
            continue
        m = _ACTIVATION_STATS_LAYER_SUBMOD_RE.search(name)
        if not m:
            continue
        key = str(int(m.group(1)))
        parts[key].append((name, p.detach().float().cpu().numpy().reshape(-1).astype(np.float64, copy=False)))
    out: dict[str, np.ndarray] = {}
    for key in sorted(parts.keys(), key=lambda s: int(s) if s.isdigit() else 0):
        row = sorted(parts[key], key=lambda t: t[0])
        if not row:
            continue
        out[key] = np.concatenate([t[1] for t in row])
    return out


def _unwrap_model_block_loop_for_attention(model: nn.Module) -> nn.Module:
    m = model
    while isinstance(m, ModelBlockLoop):
        m = m.inner
    return m


def _attention_token_ids_or_none(
    model: nn.Module, x_batch: torch.Tensor | None, batch_index: int
) -> list[int] | None:
    """Return token IDs for one attention row when positions map to single tokens."""
    if x_batch is None:
        return None
    core = _unwrap_model_block_loop_for_attention(model)
    if not isinstance(core, (AttentionTokenPredictBundle, TokenTransformerModel)) or x_batch.dim() != 2:
        return None
    try:
        batch_size = int(x_batch.shape[0])
        if batch_size <= 0:
            return None
        row = x_batch[int(batch_index) % batch_size]
        if row.dim() == 1:
            return [int(token) for token in row.detach().long().cpu().tolist()]
    except Exception:
        return None
    return None


def _softmax_attention_probs_or_none(model: nn.Module, x_batch: torch.Tensor) -> torch.Tensor | None:
    """Softmax self-attention ``[B,H,L,L]`` (last encoder layer for transformers; attention block for attention-only)."""
    core = _unwrap_model_block_loop_for_attention(model)
    try:
        with torch.no_grad():
            if isinstance(core, AttentionTokenPredictBundle):
                if x_batch.dim() != 2:
                    return None
                return core.self_attention_probs(x_batch)
            if isinstance(core, TokenTransformerModel):
                if x_batch.dim() != 2:
                    return None
                return core.self_attention_probs(x_batch.long())
            if isinstance(core, MultiTokenTransformerModel):
                if x_batch.dim() != 3:
                    return None
                return core.self_attention_probs(x_batch.long())
            if isinstance(core, NumericTransformerModel):
                return core.self_attention_probs(x_batch)
    except Exception:
        return None
    return None


def _softmax_attention_probs_all_layers_or_none(model: nn.Module, x_batch: torch.Tensor) -> list[torch.Tensor] | None:
    """Layer-wise softmax self-attention maps ``[B,H,L,L]`` (same models as last-layer helper)."""
    core = _unwrap_model_block_loop_for_attention(model)
    try:
        with torch.no_grad():
            if isinstance(core, AttentionTokenPredictBundle):
                if x_batch.dim() != 2:
                    return None
                return core.self_attention_probs_all_layers(x_batch)
            if isinstance(core, TokenTransformerModel):
                if x_batch.dim() != 2:
                    return None
                return core.self_attention_probs_all_layers(x_batch.long())
            if isinstance(core, MultiTokenTransformerModel):
                if x_batch.dim() != 3:
                    return None
                return core.self_attention_probs_all_layers(x_batch.long())
            if isinstance(core, NumericTransformerModel):
                return core.self_attention_probs_all_layers(x_batch)
    except Exception:
        return None
    return None


def _attn_sink_mass(attn: torch.Tensor, sink_idx: int) -> float:
    _, _, l, _ = attn.shape
    if l < 1:
        return float("nan")
    si = int(sink_idx) % int(l)
    return float(attn[:, :, :, si].mean())


def _attn_entropy_mean(attn: torch.Tensor) -> float:
    p = attn.clamp(min=1e-9)
    ent = -(p * p.log()).sum(dim=-1)
    return float(ent.mean())


def _attn_max_weight_mean(attn: torch.Tensor) -> float:
    return float(attn.max(dim=-1).values.mean())


def _attn_head_sink_max(attn: torch.Tensor, sink_idx: int) -> float:
    _, _, l, _ = attn.shape
    if l < 1:
        return float("nan")
    si = int(sink_idx) % int(l)
    per_h = attn[:, :, :, si].mean(dim=(0, 2))
    return float(per_h.max())


def _attn_position_bias_ratio(attn: torch.Tensor) -> float:
    pos = attn.mean(dim=(0, 1, 2))
    if pos.numel() < 1:
        return float("nan")
    m = float(pos.mean()) + 1e-12
    return float(pos[0].item() / m)


def _hessian_loss_eigenvalues(
    model: nn.Module,
    criterion: nn.Module,
    x_t: torch.Tensor,
    y_t: torch.Tensor,
    loss_scale: float,
    *,
    trainer_task: TrainerTask,
    top_k: int,
    order: Literal["descending", "ascending"],
    kan_regs: list[Node] | None = None,
    weight_reg_loss_nodes: list[Node] | None = None,
    max_params: int | None = HESSIAN_PARAM_LIMIT,
) -> list[float]:
    """Exact Hessian eigenvalues of current loss wrt trainable parameters (small models only).

    Run under ``eval()`` so dropout / other train-only stochastic layers do not advance the global
    PyTorch RNG between logging steps; otherwise weight updates diverge from runs that skip Hessian
    work (e.g. chat sweep auto-train with ``hessian_oversized_policy: "skip"`` on large models).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(int(p.numel()) for p in params)
    if n_params <= 0:
        return []
    if max_params is not None and n_params > int(max_params):
        raise HTTPException(
            status_code=400,
            detail=(
                "Hessian eigenvalue observable is limited to small models due to exact O(P^2) Hessian cost "
                f"(trainable params={n_params}, limit={max_params})."
            ),
        )

    was_training = model.training
    model.eval()
    try:
        pred = _forward_reg(model, x_t)
        loss = _trainer_primary_loss_tensor(
            pred, y_t, trainer_task=trainer_task, criterion=criterion, loss_scale=loss_scale
        )
        if kan_regs or weight_reg_loss_nodes:
            extra = _extra_loss_additions(
                model, kan_regs=kan_regs, weight_reg_loss_nodes=weight_reg_loss_nodes
            )
            if extra is not None:
                loss = loss + extra

        grads_raw = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        grad_chunks = [(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads_raw, params)]
        flat_grad = torch.cat(grad_chunks, dim=0)
        rows: list[torch.Tensor] = []
        for i in range(int(flat_grad.numel())):
            second = torch.autograd.grad(flat_grad[i], params, retain_graph=True, allow_unused=True)
            row_chunks = [(s if s is not None else torch.zeros_like(p)).reshape(-1) for s, p in zip(second, params)]
            rows.append(torch.cat(row_chunks, dim=0).detach())
        hessian = torch.stack(rows, dim=0)
        # ``linalg.eigvalsh`` is not implemented on MPS (raises NotImplementedError); keep Hessian
        # construction on the trainer device but eigen-decompose on CPU (small P by design).
        eigvals = torch.linalg.eigvalsh(hessian.detach().cpu()).numpy().astype(np.float64)
        eigvals_sorted = np.sort(eigvals)
        if order == "descending":
            eigvals_sorted = eigvals_sorted[::-1]
        k = max(1, min(int(top_k), int(eigvals_sorted.shape[0])))
        return [float(x) for x in eigvals_sorted[:k]]
    finally:
        model.train(was_training)
