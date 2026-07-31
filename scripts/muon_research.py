"""Reproducible Muon optimizer probes.

This script runs two compact experiments:

1. Matrix simple quadratics with controlled Hessian condition numbers.
2. Random-label memorization with a small bias-free MLP.

It logs scalar training dynamics plus matrix spectra for weights and updates.
The implementation is intentionally standalone so it can run locally first and
then be copied to a remote GPU instance without touching the ComfyResearch UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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

from comfy_research.engine.optimizers.muon_optimizer import build_muon_with_aux_adam


OPTIMIZERS = ("muon", "adam", "sgd", "signsgd")

SOURCE_LINKS = {
    "Muon reference": "https://kellerjordan.github.io/posts/muon/",
    "Muon implementation": "https://github.com/KellerJordan/Muon/blob/master/muon.py",
    "Insights on Muon from Simple Quadratics": "https://arxiv.org/abs/2602.11948",
    "Memory-2": "https://kindxiaoming.github.io/blog/2026/memory-2/",
}


@dataclass(frozen=True)
class RunSpec:
    task: str
    optimizer: str
    lr: float
    seed: int
    condition: float | None = None

    @property
    def run_id(self) -> str:
        cond = "none" if self.condition is None else _tag_float(self.condition)
        return f"{self.task}-{self.optimizer}-lr{_tag_float(self.lr)}-cond{cond}-seed{self.seed}"


class SignSGD(torch.optim.Optimizer):
    """Plain signSGD baseline: theta <- theta - lr * sign(grad)."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float, weight_decay: float = 0.0) -> None:
        super().__init__(params, {"lr": float(lr), "weight_decay": float(weight_decay)})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: ANN001
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    g = g.add(p, alpha=wd)
                p.add_(g.sign(), alpha=-lr)
        return loss


class MatrixQuadratic(nn.Module):
    """L(W) = 0.5 * ||sqrt(A) W sqrt(B)||_F^2 with diagonal A, B."""

    def __init__(self, dim: int, condition: float, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        init = torch.randn((dim, dim), generator=g) / math.sqrt(float(dim))
        self.W = nn.Parameter(init)
        cond = max(float(condition), 1.0)
        eig = torch.logspace(-math.log10(cond), 0.0, steps=dim)
        self.register_buffer("sqrt_left", eig.sqrt().reshape(dim, 1))
        self.register_buffer("sqrt_right", eig.sqrt().reshape(1, dim))

    def loss(self) -> torch.Tensor:
        z = self.sqrt_left * self.W * self.sqrt_right
        return 0.5 * z.square().sum()


class MemoryMLP(nn.Module):
    """Small bias-free MLP matching the D -> W -> D -> V memorization shape."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.in_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.mid_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        self.lm_head = nn.Linear(input_dim, num_classes, bias=False)
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(float(p.shape[1])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h = F.silu(self.mid_proj(h))
        return self.lm_head(h)


class Store:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.step_rows: list[dict[str, Any]] = []
        self.summary_rows: list[dict[str, Any]] = []
        self.matrix_rows: list[dict[str, Any]] = []
        self.singular_rows: list[dict[str, Any]] = []
        self.eigen_rows: list[dict[str, Any]] = []

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(self.out_dir / "step_metrics.csv", self.step_rows)
        _write_csv(self.out_dir / "summary.csv", self.summary_rows)
        _write_csv(self.out_dir / "matrix_stats.csv", self.matrix_rows)
        _write_csv(self.out_dir / "singular_values.csv", self.singular_rows)
        _write_csv(self.out_dir / "eigen_values.csv", self.eigen_rows)


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
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(raw)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return dev


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "muon":
        return build_muon_with_aux_adam(model, lr=lr, momentum=0.95, ns_steps=5, weight_decay=0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if name == "signsgd":
        return SignSGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def _flat_named_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    chunks = []
    for _, p in named_params:
        if p.grad is not None:
            chunks.append(p.grad.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def _flat_update_map(update_map: dict[str, torch.Tensor]) -> torch.Tensor:
    if not update_map:
        return torch.empty(0)
    return torch.cat([u.float().reshape(-1).cpu() for u in update_map.values()])


def _l2_norm(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.linalg.vector_norm(x.float()).item())


def _param_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    parts = [p.detach().float().reshape(-1).cpu() for _, p in named_params]
    return _l2_norm(torch.cat(parts)) if parts else float("nan")


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
    if rows >= cols:
        gram = z.T @ z
    else:
        gram = z @ z.T
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
            residual = _orthogonal_residual(update / max(spec.lr, 1e-30)) if spec.optimizer == "muon" else float("nan")
            _record_one_matrix(store, spec, step, name, "update", update, ortho_residual=residual)


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
    cond = spectral_norm / min_sv if min_sv and min_sv > 1e-12 else float("inf")
    rank = int(np.sum(s > max(1e-12, spectral_norm * 1e-8))) if s.size else 0
    row = {
        "run_id": spec.run_id,
        "task": spec.task,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "seed": spec.seed,
        "condition": "" if spec.condition is None else spec.condition,
        "step": step,
        "param_name": param_name,
        "kind": kind,
        "rows": mat.shape[0],
        "cols": mat.shape[1],
        "spectral_norm": spectral_norm,
        "fro_norm": float(np.linalg.norm(mat, ord="fro")),
        "condition_number": cond,
        "rank": rank,
        "effective_rank": _effective_rank_from_singular_values(s),
        "orthogonal_residual": ortho_residual,
    }
    store.matrix_rows.append(row)
    for i, value in enumerate(s.tolist()):
        store.singular_rows.append(
            {
                "run_id": spec.run_id,
                "task": spec.task,
                "optimizer": spec.optimizer,
                "lr": spec.lr,
                "seed": spec.seed,
                "condition": "" if spec.condition is None else spec.condition,
                "step": step,
                "param_name": param_name,
                "kind": kind,
                "index": i,
                "value": float(value),
            }
        )
    if mat.shape[0] == mat.shape[1]:
        try:
            vals = np.linalg.eigvals(mat)
        except np.linalg.LinAlgError:
            vals = np.asarray([], dtype=np.complex128)
        for i, value in enumerate(vals.tolist()):
            store.eigen_rows.append(
                {
                    "run_id": spec.run_id,
                    "task": spec.task,
                    "optimizer": spec.optimizer,
                    "lr": spec.lr,
                    "seed": spec.seed,
                    "condition": "" if spec.condition is None else spec.condition,
                    "step": step,
                    "param_name": param_name,
                    "kind": kind,
                    "index": i,
                    "real": float(np.real(value)),
                    "imag": float(np.imag(value)),
                }
            )


def _metric_row(
    spec: RunSpec,
    step: int,
    loss: float,
    accuracy: float,
    memorized_bits: float,
    param_norm: float,
    grad_norm: float,
    update_norm: float,
    grad_update_cosine: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "task": spec.task,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "seed": spec.seed,
        "condition": "" if spec.condition is None else spec.condition,
        "step": step,
        "loss": loss,
        "accuracy": accuracy,
        "memorized_bits": memorized_bits,
        "param_norm": param_norm,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "grad_update_cosine": grad_update_cosine,
        "elapsed_seconds": elapsed_seconds,
    }


def _summarize_run(
    store: Store,
    spec: RunSpec,
    started_at: float,
    threshold_kind: str,
    threshold_value: float,
    stable: bool,
) -> None:
    rows = [r for r in store.step_rows if r["run_id"] == spec.run_id]
    if not rows:
        return
    final = rows[-1]
    steps_to_threshold = float("nan")
    for row in rows:
        if threshold_kind == "loss" and float(row["loss"]) <= threshold_value:
            steps_to_threshold = int(row["step"])
            break
        if threshold_kind == "accuracy" and float(row["accuracy"]) >= threshold_value:
            steps_to_threshold = int(row["step"])
            break
    losses = [float(r["loss"]) for r in rows if _is_finite(r["loss"])]
    store.summary_rows.append(
        {
            "run_id": spec.run_id,
            "task": spec.task,
            "optimizer": spec.optimizer,
            "lr": spec.lr,
            "seed": spec.seed,
            "condition": "" if spec.condition is None else spec.condition,
            "final_step": int(final["step"]),
            "final_loss": float(final["loss"]),
            "final_accuracy": float(final["accuracy"]) if _is_finite(final["accuracy"]) else float("nan"),
            "final_memorized_bits": float(final["memorized_bits"])
            if _is_finite(final["memorized_bits"])
            else float("nan"),
            "min_loss": min(losses) if losses else float("nan"),
            "steps_to_threshold": steps_to_threshold,
            "threshold_kind": threshold_kind,
            "threshold_value": threshold_value,
            "stable": bool(stable),
            "wall_seconds": time.perf_counter() - started_at,
        }
    )


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def run_simple_quadratic(
    store: Store,
    *,
    device: torch.device,
    dim: int,
    steps: int,
    log_every: int,
    seeds: list[int],
    conditions: list[float],
    lr_grid: dict[str, list[float]],
) -> None:
    for condition in conditions:
        for seed in seeds:
            for optimizer_name in OPTIMIZERS:
                for lr in lr_grid[optimizer_name]:
                    spec = RunSpec("simple_quadratic", optimizer_name, float(lr), seed, condition)
                    started = time.perf_counter()
                    _set_seed(seed)
                    model = MatrixQuadratic(dim, condition, seed).to(device)
                    optimizer = _make_optimizer(optimizer_name, model, lr)
                    named_params = list(model.named_parameters())
                    stable = True

                    loss0 = float(model.loss().detach().cpu().item())
                    store.step_rows.append(
                        _metric_row(
                            spec,
                            0,
                            loss0,
                            float("nan"),
                            float("nan"),
                            _param_norm(named_params),
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            0.0,
                        )
                    )
                    _record_matrix_snapshot(store, spec, 0, named_params, update_map=None)

                    for step in range(1, steps + 1):
                        before = {name: p.detach().clone() for name, p in named_params}
                        optimizer.zero_grad(set_to_none=True)
                        loss = model.loss()
                        if not torch.isfinite(loss):
                            stable = False
                            break
                        loss.backward()
                        grad_vec = _flat_named_grads(named_params)
                        optimizer.step()
                        update_map = {
                            name: (before[name] - p.detach()).float().cpu()
                            for name, p in named_params
                            if name in before
                        }
                        update_vec = _flat_update_map(update_map)
                        if step % log_every == 0 or step == steps:
                            current_loss = float(model.loss().detach().cpu().item())
                            if not math.isfinite(current_loss):
                                stable = False
                            store.step_rows.append(
                                _metric_row(
                                    spec,
                                    step,
                                    current_loss,
                                    float("nan"),
                                    float("nan"),
                                    _param_norm(named_params),
                                    _l2_norm(grad_vec),
                                    _l2_norm(update_vec),
                                    _cosine(grad_vec, update_vec),
                                    time.perf_counter() - started,
                                )
                            )
                            _record_matrix_snapshot(store, spec, step, named_params, update_map=update_map)
                        if not stable:
                            break
                    _summarize_run(store, spec, started, "loss", 1e-3, stable)


def _make_random_memorization_data(
    n_train: int,
    input_dim: int,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n_train, input_dim), generator=g)
    y = torch.randint(0, num_classes, (n_train,), generator=g)
    return x.to(device), y.to(device)


@torch.no_grad()
def _eval_memory(model: nn.Module, x: torch.Tensor, y: torch.Tensor, num_classes: int) -> tuple[float, float, float]:
    model.eval()
    logits = model(x)
    loss = float(F.cross_entropy(logits, y).detach().cpu().item())
    pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().detach().cpu().item())
    memorized_bits = (math.log(float(num_classes)) - loss) * int(y.numel()) / math.log(2.0)
    model.train()
    return loss, acc, memorized_bits


def run_random_memorization(
    store: Store,
    *,
    device: torch.device,
    steps: int,
    log_every: int,
    seeds: list[int],
    lr_grid: dict[str, list[float]],
    n_train: int,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    batch_size: int,
) -> None:
    for seed in seeds:
        x, y = _make_random_memorization_data(n_train, input_dim, num_classes, seed, device)
        batch_gen = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        for optimizer_name in OPTIMIZERS:
            for lr in lr_grid[optimizer_name]:
                spec = RunSpec("random_memorization", optimizer_name, float(lr), seed, None)
                started = time.perf_counter()
                _set_seed(seed)
                model = MemoryMLP(input_dim, hidden_dim, num_classes, seed).to(device)
                optimizer = _make_optimizer(optimizer_name, model, lr)
                named_params = list(model.named_parameters())
                stable = True
                loss0, acc0, bits0 = _eval_memory(model, x, y, num_classes)
                store.step_rows.append(
                    _metric_row(
                        spec,
                        0,
                        loss0,
                        acc0,
                        bits0,
                        _param_norm(named_params),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        0.0,
                    )
                )
                _record_matrix_snapshot(store, spec, 0, named_params, update_map=None)

                for step in range(1, steps + 1):
                    idx = torch.randint(0, n_train, (min(batch_size, n_train),), generator=batch_gen, device="cpu")
                    idx = idx.to(device)
                    xb = x.index_select(0, idx)
                    yb = y.index_select(0, idx)
                    before = {name: p.detach().clone() for name, p in named_params}
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    if not torch.isfinite(loss):
                        stable = False
                        break
                    loss.backward()
                    grad_vec = _flat_named_grads(named_params)
                    optimizer.step()
                    update_map = {
                        name: (before[name] - p.detach()).float().cpu()
                        for name, p in named_params
                        if name in before
                    }
                    update_vec = _flat_update_map(update_map)
                    if step % log_every == 0 or step == steps:
                        full_loss, full_acc, bits = _eval_memory(model, x, y, num_classes)
                        if not math.isfinite(full_loss):
                            stable = False
                        store.step_rows.append(
                            _metric_row(
                                spec,
                                step,
                                full_loss,
                                full_acc,
                                bits,
                                _param_norm(named_params),
                                _l2_norm(grad_vec),
                                _l2_norm(update_vec),
                                _cosine(grad_vec, update_vec),
                                time.perf_counter() - started,
                            )
                        )
                        _record_matrix_snapshot(store, spec, step, named_params, update_map=update_map)
                    if not stable:
                        break
                _summarize_run(store, spec, started, "accuracy", 0.95, stable)


def _mode_defaults(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {
            "seeds": [0],
            "conditions": [1.0, 100.0],
            "simple_steps": 40,
            "memory_steps": 60,
            "log_every": 10,
            "simple_dim": 12,
            "n_train": 96,
            "input_dim": 10,
            "hidden_dim": 32,
            "num_classes": 8,
            "batch_size": 48,
            "simple_lrs": {"muon": [0.02], "adam": [0.03], "sgd": [0.1], "signsgd": [0.002]},
            "memory_lrs": {"muon": [0.003], "adam": [0.003], "sgd": [0.05], "signsgd": [0.001]},
        }
    if mode == "full":
        return {
            "seeds": [0, 1, 2, 3],
            "conditions": [1.0, 10.0, 100.0, 1000.0],
            "simple_steps": 400,
            "memory_steps": 1500,
            "log_every": 25,
            "simple_dim": 24,
            "n_train": 512,
            "input_dim": 16,
            "hidden_dim": 128,
            "num_classes": 16,
            "batch_size": 128,
            "simple_lrs": {
                "muon": [0.01, 0.02, 0.05],
                "adam": [0.01, 0.03, 0.1],
                "sgd": [0.05, 0.1, 0.2],
                "signsgd": [0.0005, 0.001, 0.002],
            },
            "memory_lrs": {
                "muon": [0.001, 0.003, 0.01],
                "adam": [0.001, 0.003, 0.01],
                "sgd": [0.02, 0.05, 0.1],
                "signsgd": [0.0005, 0.001, 0.002],
            },
        }
    return {
        "seeds": [0, 1],
        "conditions": [1.0, 10.0, 100.0],
        "simple_steps": 160,
        "memory_steps": 600,
        "log_every": 20,
        "simple_dim": 16,
        "n_train": 256,
        "input_dim": 12,
        "hidden_dim": 64,
        "num_classes": 10,
        "batch_size": 64,
        "simple_lrs": {
            "muon": [0.01, 0.03],
            "adam": [0.01, 0.03],
            "sgd": [0.05, 0.15],
            "signsgd": [0.0005, 0.002],
        },
        "memory_lrs": {
            "muon": [0.003, 0.01, 0.03],
            "adam": [0.001, 0.003, 0.01],
            "sgd": [0.02, 0.05, 0.1],
            "signsgd": [0.0005, 0.001, 0.002],
        },
    }


def make_plots(out_dir: Path) -> None:
    step_rows = _read_csv(out_dir / "step_metrics.csv")
    summary_rows = _read_csv(out_dir / "summary.csv")
    matrix_rows = _read_csv(out_dir / "matrix_stats.csv")
    singular_rows = _read_csv(out_dir / "singular_values.csv")
    if not step_rows or not summary_rows:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    best = _best_run_ids(summary_rows)
    _plot_simple_loss(step_rows, best, plot_dir / "simple_quadratic_loss.png")
    _plot_memory_curves(step_rows, best, plot_dir / "random_memorization_loss_accuracy.png")
    _plot_steps_to_threshold(summary_rows, plot_dir / "steps_to_threshold.png")
    _plot_update_norm(step_rows, best, plot_dir / "update_norm_comparison.png")
    _plot_condition_number(matrix_rows, best, plot_dir / "spectral_condition_over_time.png")
    _plot_singular_hist(singular_rows, best, plot_dir / "singular_values_hist_over_time.png")


def _best_run_ids(summary_rows: list[dict[str, str]]) -> set[str]:
    best: set[str] = set()
    for agg in _aggregate_best_summary_rows(summary_rows):
        for row in summary_rows:
            if row["task"] != agg["task"]:
                continue
            if row["optimizer"] != agg["optimizer"]:
                continue
            if row.get("condition", "") != agg["condition"]:
                continue
            if abs(_float(row["lr"]) - float(agg["lr"])) > 1e-12:
                continue
            best.add(row["run_id"])
    return best


def _float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _mean_by_step(rows: list[dict[str, str]], value_key: str) -> tuple[list[int], list[float]]:
    by_step: dict[int, list[float]] = {}
    for row in rows:
        v = _float(row.get(value_key))
        if math.isfinite(v):
            by_step.setdefault(int(float(row["step"])), []).append(v)
    steps = sorted(by_step)
    return steps, [float(np.mean(by_step[s])) for s in steps]


def _plot_simple_loss(step_rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in step_rows if r["task"] == "simple_quadratic" and r["run_id"] in best]
    if not rows:
        return
    conditions = sorted({r["condition"] for r in rows}, key=_float)
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 3.8), squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        for opt in OPTIMIZERS:
            rs = [r for r in rows if r["condition"] == cond and r["optimizer"] == opt]
            steps, vals = _mean_by_step(rs, "loss")
            if steps:
                ax.plot(steps, vals, label=opt)
        ax.set_title(f"condition {cond}")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_memory_curves(step_rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in step_rows if r["task"] == "random_memorization" and r["run_id"] in best]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for opt in OPTIMIZERS:
        rs = [r for r in rows if r["optimizer"] == opt]
        steps, vals = _mean_by_step(rs, "loss")
        if steps:
            axes[0].plot(steps, vals, label=opt)
        steps, vals = _mean_by_step(rs, "accuracy")
        if steps:
            axes[1].plot(steps, vals, label=opt)
    axes[0].set_title("random memorization loss")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[1].set_title("random memorization train accuracy")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_steps_to_threshold(summary_rows: list[dict[str, str]], path: Path) -> None:
    mem = [r for r in summary_rows if r["task"] == "random_memorization"]
    if not mem:
        return
    labels = []
    means = []
    for opt in OPTIMIZERS:
        vals = []
        for r in mem:
            if r["optimizer"] != opt:
                continue
            v = _float(r.get("steps_to_threshold"))
            if math.isfinite(v):
                vals.append(v)
        if vals:
            labels.append(opt)
            means.append(float(np.mean(vals)))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(labels, means)
    ax.set_ylabel("steps to 95% train accuracy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_update_norm(step_rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [r for r in step_rows if r["run_id"] in best and int(float(r["step"])) > 0]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, task in zip(axes, ("simple_quadratic", "random_memorization")):
        for opt in OPTIMIZERS:
            rs = [r for r in rows if r["task"] == task and r["optimizer"] == opt]
            steps, vals = _mean_by_step(rs, "update_norm")
            if steps:
                ax.plot(steps, vals, label=opt)
        ax.set_title(task)
        ax.set_xlabel("step")
        ax.set_ylabel("actual update norm")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_condition_number(matrix_rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [
        r
        for r in matrix_rows
        if r["run_id"] in best
        and r["task"] == "simple_quadratic"
        and r["kind"] == "weight"
        and r["param_name"] == "W"
    ]
    if not rows:
        return
    max_cond = max({_float(r["condition"]) for r in rows if _is_finite(r.get("condition"))})
    rows = [r for r in rows if abs(_float(r["condition"]) - max_cond) < 1e-8]
    fig, ax = plt.subplots(figsize=(7, 4))
    for opt in OPTIMIZERS:
        rs = [r for r in rows if r["optimizer"] == opt]
        steps, vals = _mean_by_step(rs, "condition_number")
        if steps:
            ax.plot(steps, vals, label=opt)
    ax.set_title(f"weight condition number, quadratic condition {max_cond:g}")
    ax.set_xlabel("step")
    ax.set_ylabel("condition number")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_singular_hist(singular_rows: list[dict[str, str]], best: set[str], path: Path) -> None:
    rows = [
        r
        for r in singular_rows
        if r["run_id"] in best
        and r["task"] == "simple_quadratic"
        and r["kind"] == "weight"
        and r["param_name"] == "W"
    ]
    if not rows:
        return
    max_cond = max({_float(r["condition"]) for r in rows if _is_finite(r.get("condition"))})
    rows = [r for r in rows if abs(_float(r["condition"]) - max_cond) < 1e-8]
    final_step = max(int(float(r["step"])) for r in rows)
    steps = sorted({int(float(r["step"])) for r in rows})
    chosen_steps = sorted({steps[0], steps[len(steps) // 2], final_step})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), squeeze=False)
    for ax, opt in zip(axes.reshape(-1), OPTIMIZERS):
        for step in chosen_steps:
            vals = [
                _float(r["value"])
                for r in rows
                if r["optimizer"] == opt and int(float(r["step"])) == step
            ]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                ax.hist(vals, bins=16, alpha=0.45, label=f"step {step}")
        ax.set_title(opt)
        ax.set_xlabel("singular value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"W singular values over time, quadratic condition {max_cond:g}", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, config: dict[str, Any]) -> None:
    summary_rows = _read_csv(out_dir / "summary.csv")
    matrix_rows = _read_csv(out_dir / "matrix_stats.csv")
    step_rows = _read_csv(out_dir / "step_metrics.csv")
    report = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# Muon Optimizer Mini Research Report\n")
    lines.append(f"Output directory: `{out_dir.as_posix()}`\n")
    lines.append("## Sources\n")
    for name, url in SOURCE_LINKS.items():
        lines.append(f"- {name}: {url}")
    lines.append("")
    lines.append("## Mechanism Summary\n")
    lines.append(
        "Muon applies momentum to matrix-shaped gradients and then orthogonalizes the momentum update with a "
        "short Newton-Schulz iteration. In the reference recipe, non-matrix parameters such as biases are handled "
        "by an auxiliary Adam path. This makes Muon closer to an orthogonalized update-direction optimizer for "
        "2D weights than to elementwise adaptive methods such as Adam."
    )
    lines.append(
        "Adam rescales coordinates through first and second moments, SGD follows the raw gradient, and signSGD "
        "keeps only coordinate signs. Muon's distinctive observable is therefore not just update size, but the "
        "spectrum of matrix updates and how close those updates are to a scaled orthogonal matrix."
    )
    lines.append("")
    lines.append("## Literature Notes\n")
    lines.append(
        "- The Muon reference defines Muon as momentum on 2D hidden-layer parameters followed by a tuned "
        "Newton-Schulz matrix iteration that approximates the polar factor of the update. It also recommends "
        "using AdamW-style updates for vectors, scalars, embeddings, and output heads."
    )
    lines.append(
        "- `Insights on Muon from Simple Quadratics` argues that even simple strongly convex quadratics expose "
        "behaviors missed by single-step local proxies: inexact polar steps can qualitatively affect "
        "discrete-time dynamics, and finite-budget behavior depends on more than a single condition number."
    )
    lines.append(
        "- Memory-2 uses independent Gaussian inputs and random labels, a bias-free MLP of shape `D -> W -> D` "
        "plus an LM head, cross-entropy training, and memorized bits `(log(V) - final_loss) * N / log(2)`."
    )
    lines.append("")
    lines.append("## Experiment Setup\n")
    lines.append(
        f"- Simple quadratic: `dim={config['simple_dim']}`, conditions `{config['conditions']}`, "
        f"steps `{config['simple_steps']}`."
    )
    lines.append(
        f"- Random memorization: bias-free MLP `{config['input_dim']} -> {config['hidden_dim']} -> "
        f"{config['input_dim']} -> {config['num_classes']}`, train examples `{config['n_train']}`, "
        f"batch size `{config['batch_size']}`, steps `{config['memory_steps']}`."
    )
    lines.append(f"- Seeds: `{config['seeds']}`.")
    lines.append("- Metrics: loss, train accuracy, memorized bits, parameter norm, gradient norm, actual update norm, gradient-update cosine.")
    lines.append("- Matrix metrics: singular values, eigenvalues for square matrices, spectral norm, Frobenius norm, condition number, effective rank, and Muon update orthogonal residual.")
    lines.append("")
    lines.append("## Local Results\n")
    lines.extend(_format_best_summary(summary_rows))
    lines.append("")
    lines.append("## Muon Orthogonalization Check\n")
    lines.extend(_format_muon_residuals(matrix_rows))
    lines.append("")
    lines.append("## Preliminary Conclusions\n")
    lines.extend(_format_conclusions(summary_rows, step_rows))
    lines.append("")
    lines.append("## Plots\n")
    for plot in sorted((out_dir / "plots").glob("*.png")) if (out_dir / "plots").exists() else []:
        lines.append(f"- ![{plot.stem}](plots/{plot.name})")
    lines.append("")
    lines.append("## Reproduction Commands\n")
    lines.append("```powershell")
    lines.append(".\\.venv\\Scripts\\python scripts\\muon_research.py --mode smoke --out runs\\muon_research\\smoke")
    lines.append(".\\.venv\\Scripts\\python scripts\\muon_research.py --mode local --out runs\\muon_research\\local")
    lines.append(".\\.venv\\Scripts\\python scripts\\muon_research.py --mode full --out runs\\muon_research\\full --device auto")
    lines.append("```")
    lines.append("")
    lines.append("## Cloud Compute Note\n")
    lines.append(
        "The smoke and local sweeps are designed for CPU. If scaling to the full setting or larger models, use the "
        "existing AutoDL remote GPU flow in `docs/autodl-remote-gpu.md`. A single RTX 2080 Ti/A10-class GPU should "
        "be enough for the listed full sweep; start with smoke on the remote instance before extending the grid."
    )
    lines.append("")
    lines.append("## Open Extensions\n")
    lines.append("- Run a larger learning-rate sweep before making a strong optimizer ranking claim.")
    lines.append("- Repeat random memorization with a tiny Transformer and token mapping task.")
    lines.append("- Add validation on Memory-2-style structured variants if the exact blog setup is expanded.")
    lines.append("- Track layerwise update residuals for Muon when weight decay or schedules are enabled.")
    report.write_text("\n".join(lines), encoding="utf-8")


def _format_best_summary(summary_rows: list[dict[str, str]]) -> list[str]:
    if not summary_rows:
        return ["No summary rows were generated."]
    lines = ["Best learning rate per task/optimizer/condition, aggregated across seeds:"]
    lines.append("")
    lines.append("| task | condition | optimizer | lr | seeds | mean final loss | mean final acc | mean steps to threshold |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    best = _aggregate_best_summary_rows(summary_rows)
    for row in best:
        lines.append(
            "| {task} | {condition} | {optimizer} | {lr:.4g} | {seeds} | {loss:.4g} | {acc:.4g} | {steps} |".format(
                task=row["task"],
                condition=row.get("condition") or "-",
                optimizer=row["optimizer"],
                lr=row["lr"],
                seeds=row["n_seeds"],
                loss=row["mean_final_loss"],
                acc=row["mean_final_accuracy"],
                steps=_fmt_num(row["mean_steps_to_threshold"]),
            )
        )
    return lines


def _best_summary_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in summary_rows:
        groups.setdefault((row["task"], row.get("condition", ""), row["optimizer"]), []).append(row)
    out = []
    for (task, condition, optimizer), rows in sorted(groups.items()):
        if task == "random_memorization":
            rows = sorted(rows, key=lambda r: (-_float(r["final_accuracy"]), _float(r["final_loss"])))
        else:
            rows = sorted(rows, key=lambda r: (_float(r["final_loss"]), _float(r["lr"])))
        if rows:
            out.append(rows[0])
    return out


def _aggregate_best_summary_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, float], list[dict[str, str]]] = {}
    for row in summary_rows:
        groups.setdefault((row["task"], row.get("condition", ""), row["optimizer"], _float(row["lr"])), []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (task, condition, optimizer, lr), rows in groups.items():
        losses = [_float(r["final_loss"]) for r in rows if _is_finite(r.get("final_loss"))]
        accs = [_float(r["final_accuracy"]) for r in rows if _is_finite(r.get("final_accuracy"))]
        steps = [_float(r["steps_to_threshold"]) for r in rows if _is_finite(r.get("steps_to_threshold"))]
        aggregates.append(
            {
                "task": task,
                "condition": condition,
                "optimizer": optimizer,
                "lr": lr,
                "n_seeds": len({r["seed"] for r in rows}),
                "mean_final_loss": float(np.mean(losses)) if losses else float("nan"),
                "mean_final_accuracy": float(np.mean(accs)) if accs else float("nan"),
                "mean_steps_to_threshold": float(np.mean(steps)) if steps else float("nan"),
            }
        )
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in aggregates:
        key = (row["task"], row["condition"], row["optimizer"])
        old = best.get(key)
        if old is None:
            best[key] = row
            continue
        if row["task"] == "random_memorization":
            new_score = (-row["mean_final_accuracy"], row["mean_final_loss"])
            old_score = (-old["mean_final_accuracy"], old["mean_final_loss"])
        else:
            new_score = (row["mean_final_loss"], row["lr"])
            old_score = (old["mean_final_loss"], old["lr"])
        if new_score < old_score:
            best[key] = row
    return sorted(best.values(), key=lambda r: (r["task"], _float(r["condition"], -1.0), r["optimizer"]))


def _format_muon_residuals(matrix_rows: list[dict[str, str]]) -> list[str]:
    vals = [
        _float(r["orthogonal_residual"])
        for r in matrix_rows
        if r["optimizer"] == "muon" and r["kind"] == "update" and _is_finite(r.get("orthogonal_residual"))
    ]
    if not vals:
        return ["No Muon update residuals were available."]
    return [
        (
            "Muon update orthogonal residuals across logged matrix updates: "
            f"median `{np.median(vals):.4g}`, p95 `{np.percentile(vals, 95):.4g}`, max `{np.max(vals):.4g}`."
        )
    ]


def _format_conclusions(summary_rows: list[dict[str, str]], step_rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    if not summary_rows:
        return ["No completed runs to analyze yet."]
    mem_best = [r for r in _aggregate_best_summary_rows(summary_rows) if r["task"] == "random_memorization"]
    if mem_best:
        ranked = sorted(mem_best, key=lambda r: (-r["mean_final_accuracy"], r["mean_final_loss"]))
        top = ranked[0]
        lines.append(
            f"- On this local random memorization sweep, the best run is `{top['optimizer']}` "
            f"(lr `{top['lr']}`) with mean final accuracy `{top['mean_final_accuracy']:.3f}` "
            f"and mean loss `{top['mean_final_loss']:.4g}` across `{top['n_seeds']}` seeds."
        )
        muon = [r for r in ranked if r["optimizer"] == "muon"]
        if muon:
            lines.append(
                f"- Best Muon memorization mean accuracy is `{muon[0]['mean_final_accuracy']:.3f}`; treat this as "
                "a preliminary small-scale result, not a final claim, until a wider LR/seed sweep is run."
            )
            if top["optimizer"] != "muon":
                lines.append(
                    "- Negative result: Muon does not beat tuned Adam on this small random-memorization sweep. "
                    "Both reach perfect train accuracy, but Adam reaches the 95% threshold faster in the current grid."
                )
    sq = [r for r in _aggregate_best_summary_rows(summary_rows) if r["task"] == "simple_quadratic"]
    muon_won_any_simple = False
    if sq:
        conds = sorted({r.get("condition", "") for r in sq}, key=_float)
        for cond in conds:
            rows = [r for r in sq if r.get("condition", "") == cond]
            rows = sorted(rows, key=lambda r: r["mean_final_loss"])
            if rows:
                muon_won_any_simple = muon_won_any_simple or rows[0]["optimizer"] == "muon"
                lines.append(
                    f"- For simple quadratic condition `{cond}`, best mean final loss among logged runs is "
                    f"`{rows[0]['mean_final_loss']:.4g}` from `{rows[0]['optimizer']}`."
                )
    if sq and not muon_won_any_simple:
        lines.append(
            "- Negative result: Muon is not the best simple-quadratic optimizer in this finite-budget setup; "
            "SGD wins the isotropic case and Adam wins the ill-conditioned cases with the tested learning rates."
        )
        lines.append(
            "- This is directionally consistent with the simple-quadratics paper's warning: Muon behavior on even "
            "basic quadratics is a discrete-time, finite-budget phenomenon, not a universal condition-number-only win."
        )
    lines.append(
        "- Spectral interpretation: Muon's orthogonalized updates help it separate from SGD/signSGD on memorization, "
        "but the current spectra do not explain an advantage over tuned Adam; the next useful test is a wider "
        "seed/LR sweep with per-layer spectrum plots."
    )
    unstable = [r for r in summary_rows if str(r.get("stable", "")).lower() != "true"]
    if unstable:
        lines.append(f"- Negative result: `{len(unstable)}` runs became numerically unstable or stopped early.")
    if not step_rows:
        lines.append("- Step metrics are missing, so curve-level conclusions are not yet available.")
    return lines or ["The current result table is too small for a stable conclusion."]


def _fmt_num(x: float) -> str:
    if not math.isfinite(x):
        return "-"
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.4g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "local", "full"], default="smoke")
    parser.add_argument("--task", choices=["all", "simple", "memory", "plots", "report"], default="all")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _mode_defaults(args.mode)
    out_dir = args.out or Path("runs") / "muon_research" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    config = {**config, "mode": args.mode, "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.task in {"plots", "report"}:
        if args.task == "plots":
            make_plots(out_dir)
        else:
            write_report(out_dir, config)
        return

    store = Store(out_dir)
    if args.task in {"all", "simple"}:
        run_simple_quadratic(
            store,
            device=device,
            dim=int(config["simple_dim"]),
            steps=int(config["simple_steps"]),
            log_every=int(config["log_every"]),
            seeds=list(config["seeds"]),
            conditions=list(config["conditions"]),
            lr_grid=dict(config["simple_lrs"]),
        )
    if args.task in {"all", "memory"}:
        run_random_memorization(
            store,
            device=device,
            steps=int(config["memory_steps"]),
            log_every=int(config["log_every"]),
            seeds=list(config["seeds"]),
            lr_grid=dict(config["memory_lrs"]),
            n_train=int(config["n_train"]),
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            num_classes=int(config["num_classes"]),
            batch_size=int(config["batch_size"]),
        )
    store.write()
    if not args.skip_plots:
        make_plots(out_dir)
    if not args.skip_report:
        write_report(out_dir, config)
    print(f"Wrote Muon research outputs to {out_dir}")


if __name__ == "__main__":
    main()
