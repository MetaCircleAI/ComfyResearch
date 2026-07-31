"""Optimizer dynamics sweeps for Muon vs Adam/AdamW/SGD/SignSGD.

This is a local-first reproduction harness.  It intentionally stays smaller
than paper-scale language-model runs, but it keeps the observables that are
useful for optimizer dynamics:

1. Controlled square matrix quadratics with known Hessian condition.
2. A ComfyResearch PCFG toy language-model task with ``TokenTransformerModel``.

Outputs are CSV metrics, matrix spectra, plots, and a short Markdown report.
The GPU mode is sized to produce tens of optimizer/dataset/LR batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfy_research.engine.optimizers.muon_optimizer import SingleDeviceMuonWithAuxAdam, build_muon_with_aux_adam
from comfy_research.engine.optimizers.signsgd_optimizer import SignSGD
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel
from comfy_research.engine.datasets.toy_language_pcfg_runtime import (
    PCFG_GEN_BINARY_TREE,
    PCFG_GEN_CFG_SENTENCE,
    build_pcfg_lm_arrays_from_seed,
)


OPTIMIZERS = ("muon", "adam", "adamw", "sgd", "signsgd")


@dataclass(frozen=True)
class RunSpec:
    task: str
    variant: str
    optimizer: str
    lr: float
    seed: int
    condition: float | None = None

    @property
    def run_id(self) -> str:
        cond = "none" if self.condition is None else _tag_float(self.condition)
        return (
            f"{self.task}-{self.variant}-{self.optimizer}-"
            f"lr{_tag_float(self.lr)}-cond{cond}-seed{self.seed}"
        )


@dataclass(frozen=True)
class LmConfig:
    vocab_size: int
    context_length: int
    model_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    train_n: int
    test_n: int
    batch_size: int
    eval_batch_size: int
    steps: int
    log_every: int


class Store:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.metric_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "metrics.csv"))
        self.summary_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "summary.csv"))
        self.matrix_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "matrix_stats.csv"))
        self.singular_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "singular_values.csv"))
        self.eigen_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "eigen_values.csv"))
        self.gram_eigen_rows: list[dict[str, Any]] = list(_read_csv(out_dir / "gram_eigen_values.csv"))

    def completed_run_ids(self) -> set[str]:
        return {str(row.get("run_id", "")) for row in self.summary_rows if row.get("run_id")}

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(self.out_dir / "metrics.csv", self.metric_rows)
        _write_csv(self.out_dir / "summary.csv", self.summary_rows)
        _write_csv(self.out_dir / "matrix_stats.csv", self.matrix_rows)
        _write_csv(self.out_dir / "singular_values.csv", self.singular_rows)
        _write_csv(self.out_dir / "eigen_values.csv", self.eigen_rows)
        _write_csv(self.out_dir / "gram_eigen_values.csv", self.gram_eigen_rows)


class RunLimiter:
    def __init__(self, *, offset: int = 0, max_runs: int | None = None) -> None:
        self.offset = max(0, int(offset))
        self.max_runs = None if max_runs is None or max_runs < 0 else int(max_runs)
        self.seen = 0
        self.accepted = 0

    def allow(self, _spec: RunSpec) -> bool:
        idx = self.seen
        self.seen += 1
        if idx < self.offset:
            return False
        if self.max_runs is not None and self.accepted >= self.max_runs:
            return False
        self.accepted += 1
        return True


class MatrixQuadratic(nn.Module):
    """L(W) = 0.5 * ||sqrt(A) W sqrt(B)||_F^2 with diagonal A, B."""

    def __init__(self, dim: int, condition: float, seed: int) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        init = torch.randn((dim, dim), generator=g) / math.sqrt(float(dim))
        self.W = nn.Parameter(init)
        cond = max(float(condition), 1.0)
        eig = torch.logspace(-math.log10(cond), 0.0, steps=dim)
        self.register_buffer("sqrt_left", eig.sqrt().reshape(dim, 1))
        self.register_buffer("sqrt_right", eig.sqrt().reshape(1, dim))

    def loss(self) -> torch.Tensor:
        z = self.sqrt_left * self.W * self.sqrt_right
        return 0.5 * z.square().sum()


def _tag_float(x: float) -> str:
    text = f"{float(x):.6g}"
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _select_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(raw)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return dev


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _weight_decay_for(config: dict[str, Any], optimizer_name: str) -> float:
    wds = dict(config.get("weight_decays", {}))
    return float(wds.get(optimizer_name, 0.0))


def _make_optimizer(
    name: str,
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    token_lm_muon_split: bool,
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name == "signsgd":
        return SignSGD(params, lr=lr, weight_decay=weight_decay)
    if name == "muon":
        if not token_lm_muon_split:
            return build_muon_with_aux_adam(
                model,
                lr=lr,
                momentum=0.95,
                weight_decay=weight_decay,
                ns_steps=5,
                aux_betas=(0.9, 0.95),
                aux_eps=1e-8,
            )
        muon_params: list[torch.nn.Parameter] = []
        aux_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for param_name, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_token_io = param_name.startswith("embedding.") or param_name.startswith("lm_head.")
            if p.ndim >= 2 and not is_token_io:
                muon_params.append(p)
            else:
                aux_params.append(p)
        groups: list[dict[str, Any]] = []
        if muon_params:
            groups.append(
                {
                    "params": muon_params,
                    "lr": lr,
                    "momentum": 0.95,
                    "weight_decay": weight_decay,
                    "use_muon": True,
                    "ns_steps": 5,
                }
            )
        if aux_params:
            groups.append(
                {
                    "params": aux_params,
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": weight_decay,
                    "use_muon": False,
                }
            )
        return SingleDeviceMuonWithAuxAdam(groups)
    raise ValueError(f"Unknown optimizer: {name}")


def _flat_named_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    chunks = []
    for _, p in named_params:
        if p.grad is not None:
            chunks.append(p.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def _flat_update_map(update_map: dict[str, torch.Tensor]) -> torch.Tensor:
    chunks = [u.float().reshape(-1).cpu() for u in update_map.values()]
    return torch.cat(chunks) if chunks else torch.empty(0)


def _l2_norm(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(x.float()).item())


def _param_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    chunks = [p.detach().float().reshape(-1).cpu() for _, p in named_params]
    return _l2_norm(torch.cat(chunks)) if chunks else float("nan")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    den = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(den.item()) == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / den.item())


def _effective_rank_from_singular_values(s: np.ndarray, eps: float = 1e-12) -> float:
    if s.size == 0:
        return float("nan")
    total = float(s.sum())
    if total <= eps:
        return 1.0
    p = s / total
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, eps)))))


def _spectral_entropy_from_singular_values(s: np.ndarray, eps: float = 1e-12) -> float:
    if s.size == 0:
        return float("nan")
    total = float(s.sum())
    if total <= eps:
        return 0.0
    p = s / total
    return float(-np.sum(p * np.log(np.maximum(p, eps))))


def _orthogonal_residual(update_direction: np.ndarray, eps: float = 1e-12) -> float:
    if update_direction.ndim == 1:
        mat = update_direction.reshape(1, -1)
    else:
        mat = update_direction.reshape(update_direction.shape[0], -1)
    if mat.size == 0 or float(np.linalg.norm(mat)) <= eps:
        return float("nan")
    rows, cols = mat.shape
    scale = math.sqrt(max(1.0, rows / max(cols, 1)))
    z = mat / scale
    gram = z.T @ z if rows >= cols else z @ z.T
    ident = np.eye(gram.shape[0], dtype=np.float64)
    return float(np.linalg.norm(gram - ident, ord="fro") / math.sqrt(float(gram.shape[0])))


def _record_matrix_snapshot(
    store: Store,
    spec: RunSpec,
    step: int,
    named_params: list[tuple[str, torch.nn.Parameter]],
    update_map: dict[str, torch.Tensor] | None,
) -> None:
    for name, p in named_params:
        if p.ndim < 2:
            continue
        weight = p.detach().float().cpu().numpy().reshape(p.shape[0], -1)
        _record_one_matrix(store, spec, step, name, "weight", weight, ortho_residual=float("nan"))
        if update_map and name in update_map:
            update = update_map[name].float().cpu().numpy().reshape(p.shape[0], -1)
            direction = update / max(float(spec.lr), 1e-30)
            _record_one_matrix(
                store,
                spec,
                step,
                name,
                "update",
                update,
                ortho_residual=_orthogonal_residual(direction),
            )


def _record_one_matrix(
    store: Store,
    spec: RunSpec,
    step: int,
    param_name: str,
    kind: str,
    mat: np.ndarray,
    ortho_residual: float,
) -> None:
    try:
        s = np.linalg.svd(mat, compute_uv=False)
    except np.linalg.LinAlgError:
        s = np.asarray([], dtype=np.float64)
    spectral_norm = float(s[0]) if s.size else float("nan")
    min_sv = float(s[-1]) if s.size else float("nan")
    fro = float(np.linalg.norm(mat, ord="fro"))
    cond = spectral_norm / min_sv if min_sv and min_sv > 1e-12 else float("inf")
    stable_rank = (fro * fro) / (spectral_norm * spectral_norm) if spectral_norm and spectral_norm > 1e-12 else float("nan")
    rank = int(np.sum(s > max(1e-12, spectral_norm * 1e-8))) if s.size else 0
    eig_radius = float("nan")
    eig_real_mean = float("nan")
    eig_real_std = float("nan")
    eig_imag_abs_mean = float("nan")
    if mat.shape[0] == mat.shape[1] and mat.shape[0] <= 512:
        try:
            eig = np.linalg.eigvals(mat)
        except np.linalg.LinAlgError:
            eig = np.asarray([], dtype=np.complex128)
        if eig.size:
            eig_radius = float(np.max(np.abs(eig)))
            eig_real_mean = float(np.mean(np.real(eig)))
            eig_real_std = float(np.std(np.real(eig)))
            eig_imag_abs_mean = float(np.mean(np.abs(np.imag(eig))))
        for i, value in enumerate(eig.tolist()):
            store.eigen_rows.append(
                {
                    **_row_ids(spec, step),
                    "param_name": param_name,
                    "kind": kind,
                    "index": i,
                    "real": float(np.real(value)),
                    "imag": float(np.imag(value)),
                    "abs": float(abs(value)),
                }
            )

    denom = float(max(1, mat.shape[1]))
    gram_eigs = (s * s) / denom if s.size else np.asarray([], dtype=np.float64)
    gram_top_fraction = float(gram_eigs[0] / gram_eigs.sum()) if gram_eigs.size and gram_eigs.sum() > 0 else float("nan")
    for i, value in enumerate(gram_eigs.tolist()):
        store.gram_eigen_rows.append(
            {
                **_row_ids(spec, step),
                "param_name": param_name,
                "kind": kind,
                "index": i,
                "value": float(value),
            }
        )

    store.matrix_rows.append(
        {
            **_row_ids(spec, step),
            "param_name": param_name,
            "kind": kind,
            "rows": int(mat.shape[0]),
            "cols": int(mat.shape[1]),
            "spectral_norm": spectral_norm,
            "fro_norm": fro,
            "condition_number": cond,
            "rank": rank,
            "stable_rank": stable_rank,
            "effective_rank": _effective_rank_from_singular_values(s),
            "spectral_entropy": _spectral_entropy_from_singular_values(s),
            "gram_top_fraction": gram_top_fraction,
            "orthogonal_residual": ortho_residual,
            "eigen_radius": eig_radius,
            "eigen_real_mean": eig_real_mean,
            "eigen_real_std": eig_real_std,
            "eigen_imag_abs_mean": eig_imag_abs_mean,
        }
    )
    for i, value in enumerate(s.tolist()):
        store.singular_rows.append(
            {
                **_row_ids(spec, step),
                "param_name": param_name,
                "kind": kind,
                "index": i,
                "value": float(value),
            }
        )


def _row_ids(spec: RunSpec, step: int) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "task": spec.task,
        "variant": spec.variant,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "seed": spec.seed,
        "condition": "" if spec.condition is None else spec.condition,
        "step": step,
    }


def _metric_row(
    spec: RunSpec,
    step: int,
    *,
    train_loss: float,
    eval_loss: float,
    train_accuracy: float,
    eval_accuracy: float,
    perplexity: float,
    param_norm: float,
    grad_norm: float,
    update_norm: float,
    grad_update_cosine: float,
    tokens_per_second: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        **_row_ids(spec, step),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "train_accuracy": train_accuracy,
        "eval_accuracy": eval_accuracy,
        "perplexity": perplexity,
        "param_norm": param_norm,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "grad_update_cosine": grad_update_cosine,
        "tokens_per_second": tokens_per_second,
        "elapsed_seconds": elapsed_seconds,
    }


def _summarize_run(
    store: Store,
    spec: RunSpec,
    started_at: float,
    *,
    stable: bool,
    threshold_kind: str,
    threshold_value: float,
) -> None:
    rows = [r for r in store.metric_rows if r["run_id"] == spec.run_id]
    if not rows:
        return
    final = rows[-1]
    steps_to_threshold = float("nan")
    for row in rows:
        if threshold_kind == "loss":
            value = _float(row.get("eval_loss") if _is_finite(row.get("eval_loss")) else row.get("train_loss"))
            if value <= threshold_value:
                steps_to_threshold = int(row["step"])
                break
        elif threshold_kind == "accuracy":
            value = _float(row.get("eval_accuracy") if _is_finite(row.get("eval_accuracy")) else row.get("train_accuracy"))
            if value >= threshold_value:
                steps_to_threshold = int(row["step"])
                break
    losses = [
        _float(r.get("eval_loss") if _is_finite(r.get("eval_loss")) else r.get("train_loss"))
        for r in rows
    ]
    losses = [x for x in losses if math.isfinite(x)]
    store.summary_rows.append(
        {
            **_row_ids(spec, int(final["step"])),
            "final_train_loss": _float(final.get("train_loss")),
            "final_eval_loss": _float(final.get("eval_loss")),
            "final_train_accuracy": _float(final.get("train_accuracy")),
            "final_eval_accuracy": _float(final.get("eval_accuracy")),
            "final_perplexity": _float(final.get("perplexity")),
            "min_loss": min(losses) if losses else float("nan"),
            "steps_to_threshold": steps_to_threshold,
            "threshold_kind": threshold_kind,
            "threshold_value": threshold_value,
            "stable": bool(stable),
            "wall_seconds": time.perf_counter() - started_at,
        }
    )


def run_matrix_quadratic(
    store: Store,
    *,
    config: dict[str, Any],
    device: torch.device,
    limiter: RunLimiter,
) -> None:
    qcfg = dict(config["quadratic"])
    steps = int(qcfg["steps"])
    log_every = int(qcfg["log_every"])
    dim = int(qcfg["dim"])
    lr_grid = dict(qcfg["lrs"])
    for condition in list(qcfg["conditions"]):
        for seed in list(qcfg["seeds"]):
            for optimizer_name in OPTIMIZERS:
                for lr in list(lr_grid[optimizer_name]):
                    spec = RunSpec("matrix_quadratic", "diag_hessian", optimizer_name, float(lr), int(seed), float(condition))
                    if spec.run_id in store.completed_run_ids():
                        print(f"[skip] {spec.run_id}", flush=True)
                        continue
                    if not limiter.allow(spec):
                        continue
                    print(f"[run] {spec.run_id}", flush=True)
                    started = time.perf_counter()
                    _set_seed(int(seed))
                    model = MatrixQuadratic(dim, float(condition), int(seed)).to(device)
                    optimizer = _make_optimizer(
                        optimizer_name,
                        model,
                        lr=float(lr),
                        weight_decay=_weight_decay_for(config, optimizer_name),
                        token_lm_muon_split=False,
                    )
                    named_params = list(model.named_parameters())
                    stable = True
                    loss0 = float(model.loss().detach().cpu().item())
                    store.metric_rows.append(
                        _metric_row(
                            spec,
                            0,
                            train_loss=loss0,
                            eval_loss=float("nan"),
                            train_accuracy=float("nan"),
                            eval_accuracy=float("nan"),
                            perplexity=float("nan"),
                            param_norm=_param_norm(named_params),
                            grad_norm=float("nan"),
                            update_norm=float("nan"),
                            grad_update_cosine=float("nan"),
                            tokens_per_second=float("nan"),
                            elapsed_seconds=0.0,
                        )
                    )
                    _record_matrix_snapshot(store, spec, 0, named_params, update_map=None)
                    for step in range(1, steps + 1):
                        should_log = step % log_every == 0 or step == steps
                        before = {name: p.detach().clone() for name, p in named_params} if should_log else None
                        optimizer.zero_grad(set_to_none=True)
                        loss = model.loss()
                        if not torch.isfinite(loss):
                            stable = False
                            break
                        loss.backward()
                        grad_vec = _flat_named_grads(named_params) if should_log else torch.empty(0)
                        optimizer.step()
                        if should_log:
                            assert before is not None
                            update_map = {
                                name: (before[name] - p.detach()).float().cpu()
                                for name, p in named_params
                                if name in before
                            }
                            update_vec = _flat_update_map(update_map)
                            current_loss = float(model.loss().detach().cpu().item())
                            if not math.isfinite(current_loss):
                                stable = False
                            store.metric_rows.append(
                                _metric_row(
                                    spec,
                                    step,
                                    train_loss=current_loss,
                                    eval_loss=float("nan"),
                                    train_accuracy=float("nan"),
                                    eval_accuracy=float("nan"),
                                    perplexity=float("nan"),
                                    param_norm=_param_norm(named_params),
                                    grad_norm=_l2_norm(grad_vec),
                                    update_norm=_l2_norm(update_vec),
                                    grad_update_cosine=_cosine(grad_vec, update_vec),
                                    tokens_per_second=float("nan"),
                                    elapsed_seconds=time.perf_counter() - started,
                                )
                            )
                            _record_matrix_snapshot(store, spec, step, named_params, update_map=update_map)
                        if not stable:
                            break
                    _summarize_run(
                        store,
                        spec,
                        started,
                        stable=stable,
                        threshold_kind="loss",
                        threshold_value=float(qcfg["loss_threshold"]),
                    )
                    store.write()


def _make_lm_model(cfg: LmConfig, seed: int, device: torch.device) -> TokenTransformerModel:
    _set_seed(seed)
    model = TokenTransformerModel(
        cfg.vocab_size,
        cfg.context_length,
        cfg.model_dim,
        cfg.num_heads,
        cfg.num_layers,
        cfg.ff_dim,
        tie_embedding_lm_head=True,
        causal=True,
        activation="gelu",
        encoder_backend="stable",
        encoder_dropout=0.0,
    )
    return model.to(device)


def _pcfg_data_dict(variant: str, cfg: LmConfig, seed: int) -> dict[str, Any]:
    if variant == "cfg_sentence":
        return {
            "seed": int(seed),
            "vocabSize": int(cfg.vocab_size),
            "contextLength": int(cfg.context_length),
            "pcfgGenMode": PCFG_GEN_CFG_SENTENCE,
            "pcfgGrammarId": "world_model",
            "pcfgMaxDepth": 10,
        }
    return {
        "seed": int(seed),
        "vocabSize": int(cfg.vocab_size),
        "contextLength": int(cfg.context_length),
        "pcfgGenMode": PCFG_GEN_BINARY_TREE,
        "pcfgMaxDepth": 9,
        "pcfgTermProb": 0.28,
    }


def _to_device_arrays(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(x_train_np).long().to(device),
        torch.from_numpy(y_train_np).long().to(device),
        torch.from_numpy(x_test_np).long().to(device),
        torch.from_numpy(y_test_np).long().to(device),
    )


def _lm_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).long())


@torch.no_grad()
def _eval_lm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[float, float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    try:
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + int(batch_size)]
            yb = y[start : start + int(batch_size)]
            logits = model(xb)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1).long(),
                reduction="sum",
            )
            total_loss += float(loss_sum.detach().cpu().item())
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().detach().cpu().item())
            total_tokens += int(yb.numel())
    finally:
        model.train(was_training)
    if total_tokens <= 0:
        return float("nan"), float("nan"), float("nan")
    loss = total_loss / float(total_tokens)
    return loss, correct / float(total_tokens), math.exp(min(50.0, loss))


def run_pcfg_lm(
    store: Store,
    *,
    config: dict[str, Any],
    device: torch.device,
    limiter: RunLimiter,
) -> None:
    raw = dict(config["pcfg_lm"])
    lm_cfg = LmConfig(
        vocab_size=int(raw["vocab_size"]),
        context_length=int(raw["context_length"]),
        model_dim=int(raw["model_dim"]),
        num_heads=int(raw["num_heads"]),
        num_layers=int(raw["num_layers"]),
        ff_dim=int(raw["ff_dim"]),
        train_n=int(raw["train_n"]),
        test_n=int(raw["test_n"]),
        batch_size=int(raw["batch_size"]),
        eval_batch_size=int(raw["eval_batch_size"]),
        steps=int(raw["steps"]),
        log_every=int(raw["log_every"]),
    )
    lr_grid = dict(raw["lrs"])
    variants = list(raw["variants"])
    for variant in variants:
        for seed in list(raw["seeds"]):
            data = _pcfg_data_dict(str(variant), lm_cfg, int(seed))
            arrays = build_pcfg_lm_arrays_from_seed(data, lm_cfg.train_n, lm_cfg.test_n)
            x_train, y_train, x_test, y_test = _to_device_arrays(*arrays, device)
            batch_gen = torch.Generator(device="cpu").manual_seed(30_000 + int(seed))
            for optimizer_name in OPTIMIZERS:
                for lr in list(lr_grid[optimizer_name]):
                    spec = RunSpec("pcfg_lm", str(variant), optimizer_name, float(lr), int(seed), None)
                    if spec.run_id in store.completed_run_ids():
                        print(f"[skip] {spec.run_id}", flush=True)
                        continue
                    if not limiter.allow(spec):
                        continue
                    print(f"[run] {spec.run_id}", flush=True)
                    started = time.perf_counter()
                    _set_seed(int(seed))
                    model = _make_lm_model(lm_cfg, int(seed), device)
                    optimizer = _make_optimizer(
                        optimizer_name,
                        model,
                        lr=float(lr),
                        weight_decay=_weight_decay_for(config, optimizer_name),
                        token_lm_muon_split=True,
                    )
                    named_params = list(model.named_parameters())
                    stable = True
                    eval0_loss, eval0_acc, ppl0 = _eval_lm(model, x_test, y_test, batch_size=lm_cfg.eval_batch_size)
                    store.metric_rows.append(
                        _metric_row(
                            spec,
                            0,
                            train_loss=float("nan"),
                            eval_loss=eval0_loss,
                            train_accuracy=float("nan"),
                            eval_accuracy=eval0_acc,
                            perplexity=ppl0,
                            param_norm=_param_norm(named_params),
                            grad_norm=float("nan"),
                            update_norm=float("nan"),
                            grad_update_cosine=float("nan"),
                            tokens_per_second=0.0,
                            elapsed_seconds=0.0,
                        )
                    )
                    _record_matrix_snapshot(store, spec, 0, named_params, update_map=None)
                    tokens_seen = 0
                    for step in range(1, lm_cfg.steps + 1):
                        should_log = step % lm_cfg.log_every == 0 or step == lm_cfg.steps
                        idx = torch.randint(
                            0,
                            int(x_train.shape[0]),
                            (min(lm_cfg.batch_size, int(x_train.shape[0])),),
                            generator=batch_gen,
                            device="cpu",
                        ).to(device)
                        xb = x_train.index_select(0, idx)
                        yb = y_train.index_select(0, idx)
                        tokens_seen += int(yb.numel())
                        before = {name: p.detach().clone() for name, p in named_params} if should_log else None
                        optimizer.zero_grad(set_to_none=True)
                        loss = _lm_loss(model, xb, yb)
                        if not torch.isfinite(loss):
                            stable = False
                            break
                        loss.backward()
                        grad_vec = _flat_named_grads(named_params) if should_log else torch.empty(0)
                        optimizer.step()
                        if should_log:
                            _sync_device(device)
                            assert before is not None
                            update_map = {
                                name: (before[name] - p.detach()).float().cpu()
                                for name, p in named_params
                                if name in before
                            }
                            update_vec = _flat_update_map(update_map)
                            eval_loss, eval_acc, ppl = _eval_lm(
                                model,
                                x_test,
                                y_test,
                                batch_size=lm_cfg.eval_batch_size,
                            )
                            elapsed = time.perf_counter() - started
                            if not math.isfinite(eval_loss):
                                stable = False
                            store.metric_rows.append(
                                _metric_row(
                                    spec,
                                    step,
                                    train_loss=float(loss.detach().cpu().item()),
                                    eval_loss=eval_loss,
                                    train_accuracy=float("nan"),
                                    eval_accuracy=eval_acc,
                                    perplexity=ppl,
                                    param_norm=_param_norm(named_params),
                                    grad_norm=_l2_norm(grad_vec),
                                    update_norm=_l2_norm(update_vec),
                                    grad_update_cosine=_cosine(grad_vec, update_vec),
                                    tokens_per_second=tokens_seen / max(elapsed, 1e-12),
                                    elapsed_seconds=elapsed,
                                )
                            )
                            _record_matrix_snapshot(store, spec, step, named_params, update_map=update_map)
                        if not stable:
                            break
                    _summarize_run(
                        store,
                        spec,
                        started,
                        stable=stable,
                        threshold_kind="loss",
                        threshold_value=float(raw["loss_threshold"]),
                    )
                    store.write()


def _mode_defaults(mode: str) -> dict[str, Any]:
    common = {
        "weight_decays": {
            "muon": 0.0,
            "adam": 0.0,
            "adamw": 0.01,
            "sgd": 0.0,
            "signsgd": 0.0,
        },
    }
    if mode == "smoke":
        return {
            **common,
            "quadratic": {
                "seeds": [0],
                "conditions": [1.0, 100.0],
                "steps": 30,
                "log_every": 10,
                "dim": 12,
                "loss_threshold": 1e-3,
                "lrs": {
                    "muon": [0.02],
                    "adam": [0.03],
                    "adamw": [0.03],
                    "sgd": [0.1],
                    "signsgd": [0.002],
                },
            },
            "pcfg_lm": {
                "seeds": [0],
                "variants": ["binary_tree"],
                "steps": 12,
                "log_every": 6,
                "train_n": 128,
                "test_n": 64,
                "batch_size": 16,
                "eval_batch_size": 64,
                "vocab_size": 64,
                "context_length": 12,
                "model_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "ff_dim": 32,
                "loss_threshold": 2.5,
                "lrs": {
                    "muon": [0.003],
                    "adam": [0.003],
                    "adamw": [0.003],
                    "sgd": [0.05],
                    "signsgd": [0.001],
                },
            },
        }
    if mode == "gpu":
        return {
            **common,
            "quadratic": {
                "seeds": [0, 1],
                "conditions": [1.0, 30.0, 300.0],
                "steps": 220,
                "log_every": 22,
                "dim": 32,
                "loss_threshold": 1e-3,
                "lrs": {
                    "muon": [0.01, 0.03],
                    "adam": [0.01, 0.03],
                    "adamw": [0.01, 0.03],
                    "sgd": [0.05, 0.15],
                    "signsgd": [0.0005, 0.0015],
                },
            },
            "pcfg_lm": {
                "seeds": [0, 1],
                "variants": ["binary_tree", "cfg_sentence"],
                "steps": 420,
                "log_every": 42,
                "train_n": 8192,
                "test_n": 1024,
                "batch_size": 256,
                "eval_batch_size": 512,
                "vocab_size": 256,
                "context_length": 32,
                "model_dim": 128,
                "num_heads": 4,
                "num_layers": 2,
                "ff_dim": 256,
                "loss_threshold": 2.0,
                "lrs": {
                    "muon": [0.0015, 0.003],
                    "adam": [0.0015, 0.003],
                    "adamw": [0.0015, 0.003],
                    "sgd": [0.04, 0.08],
                    "signsgd": [0.0003, 0.0008],
                },
            },
        }
    return {
        **common,
        "quadratic": {
            "seeds": [0, 1],
            "conditions": [1.0, 30.0, 300.0],
            "steps": 100,
            "log_every": 20,
            "dim": 20,
            "loss_threshold": 1e-3,
            "lrs": {
                "muon": [0.01, 0.03],
                "adam": [0.01, 0.03],
                "adamw": [0.01, 0.03],
                "sgd": [0.05, 0.15],
                "signsgd": [0.0005, 0.0015],
            },
        },
        "pcfg_lm": {
            "seeds": [0],
            "variants": ["binary_tree", "cfg_sentence"],
            "steps": 90,
            "log_every": 15,
            "train_n": 1024,
            "test_n": 256,
            "batch_size": 64,
            "eval_batch_size": 128,
            "vocab_size": 128,
            "context_length": 16,
            "model_dim": 48,
            "num_heads": 4,
            "num_layers": 1,
            "ff_dim": 96,
            "loss_threshold": 2.2,
            "lrs": {
                "muon": [0.002, 0.004],
                "adam": [0.002, 0.004],
                "adamw": [0.002, 0.004],
                "sgd": [0.04, 0.08],
                "signsgd": [0.0005, 0.001],
            },
        },
    }


def _best_summary_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in summary_rows:
        groups.setdefault((row["task"], row.get("variant", ""), row.get("condition", ""), row["optimizer"]), []).append(row)
    best: list[dict[str, Any]] = []
    for (task, variant, condition, optimizer), rows in groups.items():
        by_lr: dict[float, list[dict[str, str]]] = {}
        for row in rows:
            by_lr.setdefault(_float(row["lr"]), []).append(row)
        candidates = []
        for lr, lr_rows in by_lr.items():
            losses = []
            accs = []
            steps = []
            for row in lr_rows:
                loss = _float(row.get("final_eval_loss"))
                if not math.isfinite(loss):
                    loss = _float(row.get("final_train_loss"))
                if math.isfinite(loss):
                    losses.append(loss)
                acc = _float(row.get("final_eval_accuracy"))
                if math.isfinite(acc):
                    accs.append(acc)
                st = _float(row.get("steps_to_threshold"))
                if math.isfinite(st):
                    steps.append(st)
            candidates.append(
                {
                    "task": task,
                    "variant": variant,
                    "condition": condition,
                    "optimizer": optimizer,
                    "lr": lr,
                    "seeds": len({r["seed"] for r in lr_rows}),
                    "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                    "mean_accuracy": float(np.mean(accs)) if accs else float("nan"),
                    "mean_steps_to_threshold": float(np.mean(steps)) if steps else float("nan"),
                }
            )
        candidates = sorted(candidates, key=lambda r: (r["mean_loss"], r["lr"]))
        if candidates:
            best.append(candidates[0])
    return sorted(best, key=lambda r: (r["task"], r["variant"], _float(r["condition"], -1.0), r["optimizer"]))


def _best_run_ids(summary_rows: list[dict[str, str]]) -> set[str]:
    ids: set[str] = set()
    best = _best_summary_rows(summary_rows)
    for b in best:
        for row in summary_rows:
            if row["task"] != b["task"]:
                continue
            if row.get("variant", "") != b["variant"]:
                continue
            if row.get("condition", "") != b["condition"]:
                continue
            if row["optimizer"] != b["optimizer"]:
                continue
            if abs(_float(row["lr"]) - float(b["lr"])) > 1e-12:
                continue
            ids.add(row["run_id"])
    return ids


def _mean_by_step(rows: list[dict[str, str]], value_key: str) -> tuple[list[int], list[float]]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        value = _float(row.get(value_key))
        if math.isfinite(value):
            by_step.setdefault(int(float(row["step"])), []).append(value)
    steps = sorted(by_step)
    return steps, [float(np.mean(by_step[s])) for s in steps]


def make_plots(out_dir: Path) -> None:
    metrics = _read_csv(out_dir / "metrics.csv")
    summary = _read_csv(out_dir / "summary.csv")
    matrix = _read_csv(out_dir / "matrix_stats.csv")
    eigen = _read_csv(out_dir / "eigen_values.csv")
    gram = _read_csv(out_dir / "gram_eigen_values.csv")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if metrics and summary:
        best = _best_run_ids(summary)
        _plot_losses(metrics, best, plot_dir / "loss_curves_best_lrs.png")
        _plot_final_loss(summary, plot_dir / "final_loss_best_lrs.png")
        _plot_grad_update_cosine(metrics, best, plot_dir / "grad_update_cosine.png")
    if matrix and summary:
        best = _best_run_ids(summary)
        _plot_effective_rank(matrix, best, plot_dir / "effective_rank_over_time.png")
        _plot_orthogonal_residual(matrix, best, plot_dir / "update_orthogonal_residual.png")
    if gram and summary:
        _plot_gram_hist(gram, _best_run_ids(summary), plot_dir / "final_gram_eigen_hist.png")
    if eigen and summary:
        _plot_eigen_scatter(eigen, _best_run_ids(summary), plot_dir / "final_square_weight_eigen_scatter.png")


def _panel_keys(rows: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    keys = sorted({(r["task"], r.get("variant", ""), r.get("condition", "")) for r in rows})
    return keys


def _subplot_grid(n: int) -> tuple[int, int]:
    cols = min(3, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


def _plot_losses(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best]
    if not rows:
        return
    keys = _panel_keys(rows)
    nrows, ncols = _subplot_grid(len(keys))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, (task, variant, condition) in zip(axes.reshape(-1), keys):
        ax.axis("on")
        for opt in OPTIMIZERS:
            rs = [
                r
                for r in rows
                if r["task"] == task and r.get("variant", "") == variant and r.get("condition", "") == condition and r["optimizer"] == opt
            ]
            key = "eval_loss" if task == "pcfg_lm" else "train_loss"
            xs, ys = _mean_by_step(rs, key)
            if xs:
                ax.plot(xs, ys, label=opt)
        title = f"{task}/{variant}"
        if condition:
            title += f" cond={condition}"
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_final_loss(summary_rows: list[dict[str, str]], path: Path) -> None:
    best = _best_summary_rows(summary_rows)
    if not best:
        return
    labels = []
    vals = []
    for row in best:
        labels.append(f"{row['task']}\n{row['variant']}\n{row['condition'] or '-'}\n{row['optimizer']}")
        vals.append(float(row["mean_loss"]))
    fig, ax = plt.subplots(figsize=(max(8, 0.38 * len(labels)), 4.2))
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)), labels, rotation=90)
    ax.set_ylabel("best mean final loss")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_grad_update_cosine(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best and int(float(r["step"])) > 0]
    if not rows:
        return
    keys = _panel_keys(rows)
    nrows, ncols = _subplot_grid(len(keys))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, (task, variant, condition) in zip(axes.reshape(-1), keys):
        ax.axis("on")
        for opt in OPTIMIZERS:
            rs = [
                r
                for r in rows
                if r["task"] == task and r.get("variant", "") == variant and r.get("condition", "") == condition and r["optimizer"] == opt
            ]
            xs, ys = _mean_by_step(rs, "grad_update_cosine")
            if xs:
                ax.plot(xs, ys, label=opt)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{task}/{variant}" + (f" cond={condition}" if condition else ""))
        ax.set_xlabel("step")
        ax.set_ylabel("cos(grad, update)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_effective_rank(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best and r["kind"] == "weight"]
    rows = [r for r in rows if r["param_name"] in {"W", "encoder.layers.0.self_attn.w_q.weight", "encoder.layers.0.linear1.weight"}]
    if not rows:
        return
    params = sorted({r["param_name"] for r in rows})
    fig, axes = plt.subplots(1, len(params), figsize=(5.2 * len(params), 3.7), squeeze=False)
    for ax, param in zip(axes[0], params):
        for opt in OPTIMIZERS:
            rs = [r for r in rows if r["param_name"] == param and r["optimizer"] == opt]
            xs, ys = _mean_by_step(rs, "effective_rank")
            if xs:
                ax.plot(xs, ys, label=opt)
        ax.set_title(param)
        ax.set_xlabel("step")
        ax.set_ylabel("effective rank")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_orthogonal_residual(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best and r["kind"] == "update" and _is_finite(r.get("orthogonal_residual"))]
    rows = [r for r in rows if r["param_name"] in {"W", "encoder.layers.0.self_attn.w_q.weight", "encoder.layers.0.linear1.weight"}]
    if not rows:
        return
    params = sorted({r["param_name"] for r in rows})
    fig, axes = plt.subplots(1, len(params), figsize=(5.2 * len(params), 3.7), squeeze=False)
    for ax, param in zip(axes[0], params):
        for opt in OPTIMIZERS:
            rs = [r for r in rows if r["param_name"] == param and r["optimizer"] == opt]
            xs, ys = _mean_by_step(rs, "orthogonal_residual")
            if xs:
                ax.plot(xs, ys, label=opt)
        ax.set_title(param)
        ax.set_xlabel("step")
        ax.set_ylabel("orthogonal residual")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_gram_hist(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best and r["kind"] == "weight"]
    rows = [r for r in rows if r["param_name"] in {"W", "encoder.layers.0.self_attn.w_q.weight"}]
    if not rows:
        return
    final_by_run = {rid: max(int(float(r["step"])) for r in rows if r["run_id"] == rid) for rid in {r["run_id"] for r in rows}}
    rows = [r for r in rows if int(float(r["step"])) == final_by_run.get(r["run_id"])]
    params = sorted({r["param_name"] for r in rows})
    fig, axes = plt.subplots(1, len(params), figsize=(5.2 * len(params), 3.7), squeeze=False)
    for ax, param in zip(axes[0], params):
        for opt in OPTIMIZERS:
            vals = [_float(r["value"]) for r in rows if r["param_name"] == param and r["optimizer"] == opt]
            vals = [v for v in vals if math.isfinite(v) and v > 0]
            if vals:
                ax.hist(np.log10(vals), bins=20, alpha=0.35, label=opt)
        ax.set_title(param)
        ax.set_xlabel("log10 Gram eigenvalue")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_eigen_scatter(rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in rows if r["run_id"] in best and r["kind"] == "weight"]
    rows = [r for r in rows if r["param_name"] in {"W", "encoder.layers.0.self_attn.w_q.weight"}]
    if not rows:
        return
    final_by_run = {rid: max(int(float(r["step"])) for r in rows if r["run_id"] == rid) for rid in {r["run_id"] for r in rows}}
    rows = [r for r in rows if int(float(r["step"])) == final_by_run.get(r["run_id"])]
    params = sorted({r["param_name"] for r in rows})
    fig, axes = plt.subplots(1, len(params), figsize=(5.0 * len(params), 4.0), squeeze=False)
    for ax, param in zip(axes[0], params):
        for opt in OPTIMIZERS:
            xs = [_float(r["real"]) for r in rows if r["param_name"] == param and r["optimizer"] == opt]
            ys = [_float(r["imag"]) for r in rows if r["param_name"] == param and r["optimizer"] == opt]
            if xs and ys:
                ax.scatter(xs, ys, s=8, alpha=0.45, label=opt)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
        ax.set_title(param)
        ax.set_xlabel("real")
        ax.set_ylabel("imag")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, config: dict[str, Any]) -> None:
    summary = _read_csv(out_dir / "summary.csv")
    matrix = _read_csv(out_dir / "matrix_stats.csv")
    eigen = _read_csv(out_dir / "eigen_values.csv")
    gram = _read_csv(out_dir / "gram_eigen_values.csv")
    lines: list[str] = []
    lines.append("# Optimizer Dynamics Report\n")
    lines.append(f"Output directory: `{out_dir.as_posix()}`\n")
    lines.append("## Setup\n")
    lines.append("- Optimizers: Muon, Adam, AdamW, SGD, SignSGD.")
    lines.append("- Tasks: square matrix quadratic and ComfyResearch PCFG token LM.")
    lines.append("- Matrix observables: singular values, square-matrix eigenvalues, Gram eigenvalues, effective rank, stable rank, condition number, update orthogonal residual.")
    lines.append(f"- Config mode: `{config.get('mode')}`, device: `{config.get('device')}`.")
    lines.append("")
    lines.append("## Best LR Summary\n")
    if not summary:
        lines.append("No completed runs yet.")
    else:
        lines.append("| task | variant | condition | optimizer | lr | seeds | mean loss | mean acc | mean steps threshold |")
        lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for row in _best_summary_rows(summary):
            lines.append(
                "| {task} | {variant} | {condition} | {optimizer} | {lr:.4g} | {seeds} | {loss:.4g} | {acc} | {steps} |".format(
                    task=row["task"],
                    variant=row["variant"],
                    condition=row["condition"] or "-",
                    optimizer=row["optimizer"],
                    lr=float(row["lr"]),
                    seeds=int(row["seeds"]),
                    loss=float(row["mean_loss"]),
                    acc=_fmt_num(float(row["mean_accuracy"])),
                    steps=_fmt_num(float(row["mean_steps_to_threshold"])),
                )
            )
    lines.append("")
    lines.append("## Dynamics Notes\n")
    lines.extend(_format_dynamics_notes(summary, matrix, eigen, gram))
    lines.append("")
    lines.append("## Plots\n")
    for plot in sorted((out_dir / "plots").glob("*.png")) if (out_dir / "plots").exists() else []:
        lines.append(f"- ![{plot.stem}](plots/{plot.name})")
    lines.append("")
    lines.append("## Commands\n")
    lines.append("```powershell")
    lines.append(".\\.venv\\Scripts\\python scripts\\optimizer_dynamics_research.py --mode smoke --out runs\\optimizer_dynamics\\smoke")
    lines.append(".\\.venv\\Scripts\\python scripts\\optimizer_dynamics_research.py --mode local --out runs\\optimizer_dynamics\\local --device auto")
    lines.append(".\\.venv\\Scripts\\python scripts\\optimizer_dynamics_research.py --mode gpu --task lm --out runs\\optimizer_dynamics\\gpu_lm --device auto")
    lines.append("```")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _format_dynamics_notes(
    summary: list[dict[str, str]],
    matrix: list[dict[str, str]],
    eigen: list[dict[str, str]],
    gram: list[dict[str, str]],
) -> list[str]:
    lines: list[str] = []
    if not summary:
        return ["- No summary rows are available yet."]
    best = _best_summary_rows(summary)
    groups = sorted({(r["task"], r["variant"], r["condition"]) for r in best})
    for task, variant, condition in groups:
        rows = [r for r in best if r["task"] == task and r["variant"] == variant and r["condition"] == condition]
        ranked = sorted(rows, key=lambda r: r["mean_loss"])
        if ranked:
            top = ranked[0]
            label = f"{task}/{variant}" + (f"/condition={condition}" if condition else "")
            lines.append(
                f"- `{label}` best mean final loss is `{top['mean_loss']:.4g}` from `{top['optimizer']}` at lr `{top['lr']:.4g}`."
            )
    muon_res = [
        _float(r.get("orthogonal_residual"))
        for r in matrix
        if r.get("optimizer") == "muon" and r.get("kind") == "update" and _is_finite(r.get("orthogonal_residual"))
    ]
    if muon_res:
        lines.append(
            f"- Muon update orthogonal residual median `{np.median(muon_res):.4g}`, p95 `{np.percentile(muon_res, 95):.4g}`."
        )
    if eigen:
        lines.append(f"- Square-matrix eigenvalue samples written: `{len(eigen)}` rows.")
    if gram:
        lines.append(f"- Gram eigenvalue samples written for square and rectangular weights: `{len(gram)}` rows.")
    unstable = [r for r in summary if str(r.get("stable", "")).lower() != "true"]
    if unstable:
        lines.append(f"- `{len(unstable)}` runs stopped early or became numerically unstable.")
    return lines


def _fmt_num(x: float) -> str:
    if not math.isfinite(x):
        return "-"
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "local", "gpu"], default="smoke")
    parser.add_argument("--task", choices=["all", "quadratic", "lm", "plots", "report"], default="all")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-runs", type=int, default=-1, help="Cap accepted run batches; -1 means no cap.")
    parser.add_argument("--run-offset", type=int, default=0, help="Skip this many planned batches before running.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _mode_defaults(args.mode)
    out_dir = args.out or Path("runs") / "optimizer_dynamics" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    torch.set_float32_matmul_precision("high")
    config = {**config, "mode": args.mode, "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.task in {"plots", "report"}:
        if args.task == "plots":
            make_plots(out_dir)
        else:
            write_report(out_dir, config)
        return

    limiter = RunLimiter(offset=args.run_offset, max_runs=args.max_runs)
    store = Store(out_dir)
    started = time.perf_counter()
    if args.task in {"all", "quadratic"}:
        run_matrix_quadratic(store, config=config, device=device, limiter=limiter)
    if args.task in {"all", "lm"}:
        run_pcfg_lm(store, config=config, device=device, limiter=limiter)
    store.write()
    if not args.skip_plots:
        make_plots(out_dir)
    if not args.skip_report:
        write_report(out_dir, config)
    print(
        f"Wrote optimizer dynamics outputs to {out_dir} "
        f"({limiter.accepted} run batches, {time.perf_counter() - started:.1f}s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
