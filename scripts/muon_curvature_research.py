"""Compact reproduction probes for "Why Muon Outperforms Adam: A Curvature Perspective".

This script is intentionally smaller than the paper's 124M NanoGPT/FineWeb setup.
It reproduces the paper's core measurable mechanism on local hardware:

1. One-step Taylor decomposition for Adam vs. Muon updates on a tiny token
   Transformer trained on Zipf-PCFG-lite synthetic language data.
2. NDS/update-norm decomposition and the effect of stronger Zipf imbalance.
3. A structured matrix-block quadratic sanity check mirroring the paper's
   Section 5/Figure 7.

Outputs are CSV files, PNG plots, and a short Markdown report.
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
from typing import Any, Iterable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comfy_research.engine.optimizers.muon_optimizer import SingleDeviceMuonWithAuxAdam
from comfy_research.engine.models.token_transformer_model import TokenTransformerModel


OPTIMIZERS = ("muon", "adam")
SOURCE_LINKS = {
    "Paper": "https://arxiv.org/abs/2606.04662",
    "Muon reference": "https://kellerjordan.github.io/posts/muon/",
    "ComfyResearch Muon implementation": "comfy_research/engine/optimizers/muon_optimizer.py",
}


@dataclass(frozen=True)
class CurvatureRunSpec:
    optimizer: str
    zipf_s: float
    seed: int
    lr: float
    aux_lr: float

    @property
    def run_id(self) -> str:
        return f"curvature-{self.optimizer}-s{_tag_float(self.zipf_s)}-lr{_tag_float(self.lr)}-seed{self.seed}"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    model_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int


class Store:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.curvature_rows: list[dict[str, Any]] = []
        self.curvature_summary_rows: list[dict[str, Any]] = []
        self.quadratic_rows: list[dict[str, Any]] = []
        self.quadratic_summary_rows: list[dict[str, Any]] = []

    def write(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(self.out_dir / "curvature_metrics.csv", self.curvature_rows)
        _write_csv(self.out_dir / "curvature_summary.csv", self.curvature_summary_rows)
        _write_csv(self.out_dir / "structured_quadratic.csv", self.quadratic_rows)
        _write_csv(self.out_dir / "structured_quadratic_summary.csv", self.quadratic_summary_rows)


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


def _float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _make_model(cfg: ModelConfig, seed: int, device: torch.device) -> TokenTransformerModel:
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


def _make_optimizer(
    name: str,
    model: torch.nn.Module,
    *,
    lr: float,
    aux_lr: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=aux_lr, betas=(0.8, 0.95), eps=1e-8)
    if name == "muon":
        muon_params: list[torch.nn.Parameter] = []
        aux_params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for pname, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            is_embedding_or_head = pname.startswith("embedding.") or pname.startswith("lm_head.")
            if p.ndim >= 2 and not is_embedding_or_head:
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
                    "weight_decay": 0.0,
                    "use_muon": True,
                    "ns_steps": 5,
                }
            )
        if aux_params:
            groups.append(
                {
                    "params": aux_params,
                    "lr": aux_lr,
                    "betas": (0.8, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "use_muon": False,
                }
            )
        return SingleDeviceMuonWithAuxAdam(groups)
    raise ValueError(f"unknown optimizer: {name!r}")


def _loss_tensor(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    if logits.ndim == 3:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).long())
    return F.cross_entropy(logits, y.long())


@torch.no_grad()
def _eval_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 128) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for start in range(0, int(x.shape[0]), int(batch_size)):
            xb = x[start : start + batch_size]
            yb = y[start : start + batch_size]
            losses.append(float(_loss_tensor(model, xb, yb).detach().cpu().item()))
    finally:
        model.train(was_training)
    return float(np.mean(losses)) if losses else float("nan")


def _params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def _clone_params(params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


@torch.no_grad()
def _copy_params(params: list[torch.nn.Parameter], values: list[torch.Tensor]) -> None:
    for p, v in zip(params, values):
        p.copy_(v)


def _dot_tensors(xs: list[torch.Tensor], ys: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=xs[0].device if xs else "cpu")
    for x, y in zip(xs, ys):
        total = total + (x * y).sum()
    return total


def _norm_sq_tensors(xs: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=xs[0].device if xs else "cpu")
    for x in xs:
        total = total + x.square().sum()
    return total


def _hessian_quadratic_form(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    direction: list[torch.Tensor],
) -> float:
    params = _params(model)
    model.zero_grad(set_to_none=True)
    loss = _loss_tensor(model, x, y)
    grads_raw = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_raw, params)]
    directional_grad = _dot_tensors(grads, direction)
    hvp_raw = torch.autograd.grad(directional_grad, params, retain_graph=False, allow_unused=True)
    hvp = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_raw, params)]
    q = _dot_tensors(direction, hvp)
    return float(q.detach().cpu().item())


def _train_step_with_curvature_metrics(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    params = _params(model)
    before = _clone_params(params)

    optimizer.zero_grad(set_to_none=True)
    loss = _loss_tensor(model, x, y)
    loss_before = float(loss.detach().cpu().item())
    loss.backward()
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
    optimizer.step()

    after = _clone_params(params)
    updates = [b - a for b, a in zip(before, after)]
    with torch.no_grad():
        loss_after = float(_loss_tensor(model, x, y).detach().cpu().item())

    _copy_params(params, before)
    hquad = _hessian_quadratic_form(model, x, y, updates)
    _copy_params(params, after)

    first_order = float(_dot_tensors(grads, updates).detach().cpu().item())
    update_norm_sq = float(_norm_sq_tensors(updates).detach().cpu().item())
    update_norm = math.sqrt(max(update_norm_sq, 0.0))
    curvature_penalty = 0.5 * hquad
    nds = hquad / update_norm_sq if update_norm_sq > 0 else float("nan")
    return {
        "train_batch_loss": loss_before,
        "same_batch_loss_after_update": loss_after,
        "realized_loss_decrease": loss_before - loss_after,
        "first_order_gain": first_order,
        "hessian_quadratic": hquad,
        "curvature_penalty": curvature_penalty,
        "predicted_loss_decrease": first_order - curvature_penalty,
        "update_norm": update_norm,
        "update_norm_sq": update_norm_sq,
        "nds": nds,
    }


def _class_ranges(vocab_size: int, num_classes: int) -> list[np.ndarray]:
    ids = np.arange(max(2, vocab_size), dtype=np.int64)
    chunks = np.array_split(ids, max(1, num_classes))
    return [c for c in chunks if c.size > 0]


def _zipf_pcfg_lite_arrays(
    *,
    n_samples: int,
    context_length: int,
    vocab_size: int,
    zipf_s: float,
    seed: int,
    num_topics: int,
    num_classes: int,
    topic_stay: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """Small topic-conditioned PCFG-ish LM data with controllable Zipf imbalance.

    This is not the paper's full 4412-token, 20-class generator. It preserves the
    key control variable: token emission is reweighted by rank^{-s} inside each
    grammar class and latent topic.
    """
    rng = np.random.default_rng(seed)
    ranges = _class_ranges(vocab_size, num_classes)
    c = len(ranges)
    k = max(1, int(num_topics))
    alpha_phi = 0.3
    topic_class_probs: list[list[np.ndarray]] = []
    for _topic in range(k):
        per_class: list[np.ndarray] = []
        for ids in ranges:
            base = rng.gamma(alpha_phi, 1.0, size=int(ids.size)).astype(np.float64)
            base = base / max(float(base.sum()), 1e-12)
            order = np.argsort(-base)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, int(ids.size) + 1, dtype=np.float64)
            probs = base * np.power(ranks, -float(zipf_s))
            probs = probs / max(float(probs.sum()), 1e-12)
            per_class.append(probs)
        topic_class_probs.append(per_class)

    # Production templates over grammar classes. They are deliberately small but
    # include noun/verb/object/function-like positions with topic persistence.
    templates = [
        (0, 4, 1, 3, 5, 2),
        (6, 0, 3, 6, 1, 7),
        (0, 8, 3, 1, 9, 2),
        (6, 1, 2, 10, 0, 3, 1),
        (0, 3, 1, 11),
    ]
    templates = [tuple(x % c for x in t) for t in templates]
    prior = rng.dirichlet(np.ones(k, dtype=np.float64))
    target_len = int(context_length) + 1
    rows = np.empty((int(n_samples), target_len), dtype=np.int64)

    for i in range(int(n_samples)):
        topic = int(rng.choice(k, p=prior))
        out: list[int] = []
        while len(out) < target_len:
            if rng.random() > topic_stay:
                topic = int(rng.integers(0, k))
            tmpl = templates[int(rng.integers(0, len(templates)))]
            for cls in tmpl:
                ids = ranges[cls]
                probs = topic_class_probs[topic][cls]
                out.append(int(rng.choice(ids, p=probs)))
                if len(out) >= target_len:
                    break
        rows[i] = np.asarray(out[:target_len], dtype=np.int64)
    return rows[:, :-1], rows[:, 1:]


def _to_device_arrays(x: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


def run_curvature_probe(
    store: Store,
    *,
    device: torch.device,
    model_cfg: ModelConfig,
    seeds: list[int],
    zipf_exponents: list[float],
    steps: int,
    log_every: int,
    nds_every: int,
    train_n: int,
    val_n: int,
    batch_size: int,
    num_topics: int,
    num_classes: int,
    lr_muon: float,
    lr_muon_aux: float,
    lr_adam: float,
) -> None:
    for zipf_s in zipf_exponents:
        for seed in seeds:
            x_train_np, y_train_np = _zipf_pcfg_lite_arrays(
                n_samples=train_n,
                context_length=model_cfg.context_length,
                vocab_size=model_cfg.vocab_size,
                zipf_s=float(zipf_s),
                seed=seed,
                num_topics=num_topics,
                num_classes=num_classes,
            )
            # The paper fixes validation at s=0 while varying training imbalance.
            x_val_np, y_val_np = _zipf_pcfg_lite_arrays(
                n_samples=val_n,
                context_length=model_cfg.context_length,
                vocab_size=model_cfg.vocab_size,
                zipf_s=0.0,
                seed=100_000 + seed,
                num_topics=num_topics,
                num_classes=num_classes,
            )
            x_train, y_train = _to_device_arrays(x_train_np, y_train_np, device)
            x_val, y_val = _to_device_arrays(x_val_np, y_val_np, device)
            batch_gen = torch.Generator(device="cpu").manual_seed(20_000 + seed)

            for optimizer_name in OPTIMIZERS:
                lr = lr_muon if optimizer_name == "muon" else lr_adam
                aux_lr = lr_muon_aux if optimizer_name == "muon" else lr_adam
                spec = CurvatureRunSpec(optimizer_name, float(zipf_s), seed, lr, aux_lr)
                model = _make_model(model_cfg, seed, device)
                optimizer = _make_optimizer(optimizer_name, model, lr=lr, aux_lr=aux_lr)
                started = time.perf_counter()
                last_metrics: dict[str, float] | None = None

                for step in range(1, int(steps) + 1):
                    idx = torch.randint(0, int(x_train.shape[0]), (int(batch_size),), generator=batch_gen, device="cpu")
                    xb = x_train.index_select(0, idx.to(device))
                    yb = y_train.index_select(0, idx.to(device))
                    should_measure = step % int(nds_every) == 0 or step == 1 or step == int(steps)
                    if should_measure:
                        metrics = _train_step_with_curvature_metrics(model, optimizer, xb, yb)
                        last_metrics = metrics
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        loss = _loss_tensor(model, xb, yb)
                        loss.backward()
                        optimizer.step()
                        metrics = {}

                    if step % int(log_every) == 0 or should_measure or step == int(steps):
                        val_loss = _eval_loss(model, x_val, y_val)
                        row = {
                            "run_id": spec.run_id,
                            "optimizer": spec.optimizer,
                            "zipf_s": spec.zipf_s,
                            "seed": spec.seed,
                            "lr": spec.lr,
                            "aux_lr": spec.aux_lr,
                            "step": step,
                            "val_loss": val_loss,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                        if metrics:
                            row.update(metrics)
                        else:
                            row.update(
                                {
                                    "train_batch_loss": float(loss.detach().cpu().item()),
                                    "same_batch_loss_after_update": float("nan"),
                                    "realized_loss_decrease": float("nan"),
                                    "first_order_gain": float("nan"),
                                    "hessian_quadratic": float("nan"),
                                    "curvature_penalty": float("nan"),
                                    "predicted_loss_decrease": float("nan"),
                                    "update_norm": float("nan"),
                                    "update_norm_sq": float("nan"),
                                    "nds": float("nan"),
                                }
                            )
                        store.curvature_rows.append(row)

                measured = [r for r in store.curvature_rows if r["run_id"] == spec.run_id and math.isfinite(_float(r.get("nds")))]
                nds_vals = [_float(r["nds"]) for r in measured if math.isfinite(_float(r.get("nds")))]
                curv_vals = [_float(r["curvature_penalty"]) for r in measured if math.isfinite(_float(r.get("curvature_penalty")))]
                update_vals = [_float(r["update_norm_sq"]) for r in measured if math.isfinite(_float(r.get("update_norm_sq")))]
                final_val = _eval_loss(model, x_val, y_val)
                store.curvature_summary_rows.append(
                    {
                        "run_id": spec.run_id,
                        "optimizer": spec.optimizer,
                        "zipf_s": spec.zipf_s,
                        "seed": spec.seed,
                        "lr": spec.lr,
                        "aux_lr": spec.aux_lr,
                        "steps": steps,
                        "final_val_loss": final_val,
                        "mean_nds": float(np.mean(nds_vals)) if nds_vals else float("nan"),
                        "mean_curvature_penalty": float(np.mean(curv_vals)) if curv_vals else float("nan"),
                        "mean_update_norm_sq": float(np.mean(update_vals)) if update_vals else float("nan"),
                        "last_nds": float(last_metrics["nds"]) if last_metrics else float("nan"),
                    }
                )


def _random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def _direction_metrics_for_structured_quadratic(
    d1: int,
    d2: int,
    q: int,
    alpha_w: float,
    alpha_sigma: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    u = _random_orthogonal(rng, d1)
    v = _random_orthogonal(rng, d2)
    q_eff = max(1, min(int(q), int(d1), int(d2)))
    idx = np.arange(1, q_eff + 1, dtype=np.float64)
    w = np.power(idx, -float(alpha_w))
    w = w / float(w.max())
    sigma = np.power(w, float(alpha_sigma))
    g = np.zeros((d1, d2), dtype=np.float64)
    h_modes: list[np.ndarray] = []
    for i in range(q_eff):
        m = np.outer(u[:, i], v[:, i])
        h_modes.append(m)
        g += sigma[i] * m

    def hquad(direction: np.ndarray) -> float:
        coeffs = np.asarray([float(np.sum(direction * m)) for m in h_modes], dtype=np.float64)
        return float(np.sum(w * coeffs * coeffs))

    def loss_decrease(direction: np.ndarray) -> float:
        first = float(np.sum(g * direction))
        second = hquad(direction)
        if second <= 1e-20:
            return float("nan")
        return 0.5 * first * first / second

    directions = {
        "gd": g,
        "muon": sum(h_modes),
        "adam": np.sign(g),
    }
    out: dict[str, dict[str, float]] = {}
    for name, direction in directions.items():
        norm_sq = float(np.sum(direction * direction))
        quad = hquad(direction)
        out[name] = {
            "nds": quad / norm_sq if norm_sq > 0 else float("nan"),
            "loss_decrease": loss_decrease(direction),
            "direction_norm_sq": norm_sq,
            "hessian_quadratic": quad,
        }
    return out


def run_structured_quadratic(
    store: Store,
    *,
    seeds: list[int],
    d1: int,
    d2: int,
    q: int,
    alpha_w: float,
    alpha_sigma: float,
) -> None:
    for seed in seeds:
        metrics = _direction_metrics_for_structured_quadratic(d1, d2, q, alpha_w, alpha_sigma, seed)
        gd_nds = metrics["gd"]["nds"]
        gd_dec = metrics["gd"]["loss_decrease"]
        for opt, vals in metrics.items():
            row = {
                "seed": seed,
                "optimizer": opt,
                "d1": d1,
                "d2": d2,
                "q": q,
                "alpha_w": alpha_w,
                "alpha_sigma": alpha_sigma,
                **vals,
                "nds_ratio_to_gd": vals["nds"] / gd_nds if gd_nds else float("nan"),
                "loss_decrease_ratio_to_gd": vals["loss_decrease"] / gd_dec if gd_dec else float("nan"),
            }
            store.quadratic_rows.append(row)

    for opt in ("gd", "adam", "muon"):
        rows = [r for r in store.quadratic_rows if r["optimizer"] == opt]
        if not rows:
            continue
        store.quadratic_summary_rows.append(
            {
                "optimizer": opt,
                "seeds": len(rows),
                "mean_nds": float(np.mean([_float(r["nds"]) for r in rows])),
                "mean_loss_decrease": float(np.mean([_float(r["loss_decrease"]) for r in rows])),
                "mean_nds_ratio_to_gd": float(np.mean([_float(r["nds_ratio_to_gd"]) for r in rows])),
                "mean_loss_decrease_ratio_to_gd": float(
                    np.mean([_float(r["loss_decrease_ratio_to_gd"]) for r in rows])
                ),
            }
        )


def make_plots(out_dir: Path) -> None:
    curv = _read_csv(out_dir / "curvature_metrics.csv")
    curv_sum = _read_csv(out_dir / "curvature_summary.csv")
    quad_sum = _read_csv(out_dir / "structured_quadratic_summary.csv")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if curv:
        _plot_nds_over_steps(curv, plot_dir / "nds_over_steps.png")
        _plot_decomposition_ratios(curv, plot_dir / "adam_muon_decomposition_ratios.png")
    if curv_sum:
        _plot_imbalance_gap(curv_sum, plot_dir / "imbalance_nds_gap.png")
    if quad_sum:
        _plot_quadratic_summary(quad_sum, plot_dir / "structured_quadratic_ratios.png")


def _group_mean(rows: list[dict[str, str]], key: str, group_key: str = "step") -> tuple[list[float], list[float]]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        v = _float(row.get(key))
        g = _float(row.get(group_key))
        if math.isfinite(v) and math.isfinite(g):
            groups.setdefault(g, []).append(v)
    xs = sorted(groups)
    return xs, [float(np.mean(groups[x])) for x in xs]


def _plot_nds_over_steps(rows: list[dict[str, str]], path: Path) -> None:
    measured = [r for r in rows if math.isfinite(_float(r.get("nds")))]
    if not measured:
        return
    zipfs = sorted({_float(r["zipf_s"]) for r in measured})
    fig, axes = plt.subplots(1, len(zipfs), figsize=(5 * len(zipfs), 3.6), squeeze=False)
    for ax, s in zip(axes[0], zipfs):
        for opt in OPTIMIZERS:
            rs = [r for r in measured if abs(_float(r["zipf_s"]) - s) < 1e-12 and r["optimizer"] == opt]
            xs, ys = _group_mean(rs, "nds")
            if xs:
                ax.plot(xs, ys, marker="o", label=opt)
        ax.set_title(f"Zipf s={s:g}")
        ax.set_xlabel("step")
        ax.set_ylabel("NDS")
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _paired_ratio_rows(rows: list[dict[str, str]], metric: str) -> dict[float, dict[int, list[float]]]:
    by_key: dict[tuple[float, int, int], dict[str, float]] = {}
    for r in rows:
        v = _float(r.get(metric))
        if not math.isfinite(v):
            continue
        key = (_float(r["zipf_s"]), int(_float(r["seed"])), int(_float(r["step"])))
        by_key.setdefault(key, {})[r["optimizer"]] = v
    out: dict[float, dict[int, list[float]]] = {}
    for (s, _seed, step), vals in by_key.items():
        m = vals.get("muon")
        a = vals.get("adam")
        if m is None or a is None or abs(m) < 1e-30:
            continue
        out.setdefault(s, {}).setdefault(step, []).append(a / m)
    return out


def _plot_decomposition_ratios(rows: list[dict[str, str]], path: Path) -> None:
    measured = [r for r in rows if math.isfinite(_float(r.get("nds")))]
    if not measured:
        return
    metrics = [
        ("curvature_penalty", "curvature penalty"),
        ("nds", "NDS"),
        ("update_norm_sq", "update scale"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8), squeeze=False)
    for ax, (metric, label) in zip(axes[0], metrics):
        ratios = _paired_ratio_rows(measured, metric)
        for s, by_step in sorted(ratios.items()):
            steps = sorted(by_step)
            vals = [float(np.mean(by_step[step])) for step in steps]
            ax.plot(steps, vals, marker="o", label=f"s={s:g}")
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Adam/Muon {label}")
        ax.set_xlabel("step")
        ax.set_ylabel("ratio")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_imbalance_gap(rows: list[dict[str, str]], path: Path) -> None:
    groups: dict[tuple[float, str], list[float]] = {}
    for r in rows:
        v = _float(r.get("mean_nds"))
        if math.isfinite(v):
            groups.setdefault((_float(r["zipf_s"]), r["optimizer"]), []).append(v)
    if not groups:
        return
    s_vals = sorted({k[0] for k in groups})
    muon0_vals = groups.get((0.0, "muon"), [])
    norm = float(np.mean(muon0_vals)) if muon0_vals else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for opt in OPTIMIZERS:
        vals = [float(np.mean(groups.get((s, opt), [float("nan")]))) / norm for s in s_vals]
        axes[0].plot(s_vals, vals, marker="o", label=opt)
    gaps = []
    for s in s_vals:
        adam = float(np.mean(groups.get((s, "adam"), [float("nan")]))) / norm
        muon = float(np.mean(groups.get((s, "muon"), [float("nan")]))) / norm
        gaps.append(adam - muon)
    axes[1].plot(s_vals, gaps, marker="o", color="tab:red")
    axes[0].set_title("trajectory-averaged NDS")
    axes[0].set_xlabel("Zipf exponent s")
    axes[0].set_ylabel("normalized by Muon s=0")
    axes[0].legend()
    axes[1].set_title("Adam - Muon NDS gap")
    axes[1].set_xlabel("Zipf exponent s")
    axes[1].set_ylabel("normalized gap")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_quadratic_summary(rows: list[dict[str, str]], path: Path) -> None:
    opts = ["gd", "adam", "muon"]
    nds = [_float(next((r for r in rows if r["optimizer"] == opt), {}).get("mean_nds_ratio_to_gd")) for opt in opts]
    dec = [
        _float(next((r for r in rows if r["optimizer"] == opt), {}).get("mean_loss_decrease_ratio_to_gd"))
        for opt in opts
    ]
    x = np.arange(len(opts))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(x, nds)
    axes[0].set_title("NDS ratio to GD")
    axes[1].bar(x, dec)
    axes[1].set_title("loss-decrease ratio to GD")
    for ax in axes:
        ax.set_xticks(x, opts)
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, config: dict[str, Any]) -> None:
    curv_sum = _read_csv(out_dir / "curvature_summary.csv")
    quad_sum = _read_csv(out_dir / "structured_quadratic_summary.csv")
    lines: list[str] = []
    lines.append("# Muon Curvature Reproduction Report\n")
    lines.append(f"Output directory: `{out_dir.as_posix()}`\n")
    lines.append("## Sources\n")
    for name, url in SOURCE_LINKS.items():
        lines.append(f"- {name}: {url}")
    lines.append("")
    lines.append("## What This Script Reproduces\n")
    lines.append("- Figure 1 mechanism: one-step Taylor decomposition into first-order gain and curvature penalty.")
    lines.append("- Figure 2 mechanism: curvature penalty factorized into update norm and NDS.")
    lines.append("- Figure 3 mechanism: stronger Zipf imbalance can widen the Adam-Muon NDS gap.")
    lines.append("- Figure 7-style sanity check: structured quadratic directions for GD, Adam, and Muon.")
    lines.append("")
    lines.append("## Scale Caveat\n")
    lines.append(
        "The paper's main runs use 124M NanoGPT on FineWeb and HVPs every 500 steps on 4xA100-80GB. "
        "This script uses a small ComfyResearch token Transformer and a Zipf-PCFG-lite generator so it can run locally."
    )
    lines.append("")
    lines.append("## Config\n")
    lines.append("```json")
    lines.append(json.dumps(config, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Curvature Summary\n")
    if not curv_sum:
        lines.append("No curvature rows were generated.")
    else:
        lines.append("| zipf s | optimizer | seeds | mean NDS | mean curvature penalty | mean update norm^2 | mean final val loss |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        grouped: dict[tuple[float, str], list[dict[str, str]]] = {}
        for r in curv_sum:
            grouped.setdefault((_float(r["zipf_s"]), r["optimizer"]), []).append(r)
        for (s, opt), rows in sorted(grouped.items()):
            lines.append(
                "| {s:g} | {opt} | {n} | {nds:.4g} | {curv:.4g} | {upd:.4g} | {vl:.4g} |".format(
                    s=s,
                    opt=opt,
                    n=len(rows),
                    nds=float(np.mean([_float(r["mean_nds"]) for r in rows])),
                    curv=float(np.mean([_float(r["mean_curvature_penalty"]) for r in rows])),
                    upd=float(np.mean([_float(r["mean_update_norm_sq"]) for r in rows])),
                    vl=float(np.mean([_float(r["final_val_loss"]) for r in rows])),
                )
            )
    lines.append("")
    lines.append("## Structured Quadratic Summary\n")
    if not quad_sum:
        lines.append("No structured quadratic rows were generated.")
    else:
        lines.append("| optimizer | mean NDS/GD | mean loss-decrease/GD |")
        lines.append("|---|---:|---:|")
        for r in quad_sum:
            lines.append(
                f"| {r['optimizer']} | {_float(r['mean_nds_ratio_to_gd']):.4g} | "
                f"{_float(r['mean_loss_decrease_ratio_to_gd']):.4g} |"
            )
    lines.append("")
    lines.append("## Plots\n")
    for plot in sorted((out_dir / "plots").glob("*.png")) if (out_dir / "plots").exists() else []:
        lines.append(f"- ![{plot.stem}](plots/{plot.name})")
    lines.append("")
    lines.append("## Reproduction Commands\n")
    lines.append("```powershell")
    lines.append(".\\.venv\\Scripts\\python scripts\\muon_curvature_research.py --mode smoke --out runs\\muon_curvature\\smoke")
    lines.append(".\\.venv\\Scripts\\python scripts\\muon_curvature_research.py --mode local --out runs\\muon_curvature\\local --device auto")
    lines.append("```")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _mode_defaults(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {
            "seeds": [0],
            "zipf_exponents": [0.0, 1.0],
            "steps": 12,
            "log_every": 6,
            "nds_every": 6,
            "train_n": 128,
            "val_n": 64,
            "batch_size": 16,
            "num_topics": 6,
            "num_classes": 8,
            "model": {
                "vocab_size": 64,
                "context_length": 12,
                "model_dim": 16,
                "num_heads": 2,
                "num_layers": 1,
                "ff_dim": 32,
            },
            "lr_muon": 0.003,
            "lr_muon_aux": 0.006,
            "lr_adam": 0.006,
            "quadratic": {"seeds": [0, 1, 2], "d1": 64, "d2": 64, "q": 24, "alpha_w": 1.3, "alpha_sigma": 0.5},
        }
    if mode == "full":
        return {
            "seeds": [0, 1, 2],
            "zipf_exponents": [0.0, 0.5, 1.0],
            "steps": 240,
            "log_every": 20,
            "nds_every": 20,
            "train_n": 2048,
            "val_n": 512,
            "batch_size": 64,
            "num_topics": 16,
            "num_classes": 12,
            "model": {
                "vocab_size": 256,
                "context_length": 24,
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "ff_dim": 128,
            },
            "lr_muon": 0.003,
            "lr_muon_aux": 0.006,
            "lr_adam": 0.006,
            "quadratic": {"seeds": list(range(20)), "d1": 256, "d2": 256, "q": 88, "alpha_w": 1.3, "alpha_sigma": 0.5},
        }
    return {
        "seeds": [0, 1],
        "zipf_exponents": [0.0, 0.5, 1.0],
        "steps": 80,
        "log_every": 10,
        "nds_every": 10,
        "train_n": 768,
        "val_n": 256,
        "batch_size": 32,
        "num_topics": 10,
        "num_classes": 10,
        "model": {
            "vocab_size": 128,
            "context_length": 16,
            "model_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "ff_dim": 64,
        },
        "lr_muon": 0.003,
        "lr_muon_aux": 0.006,
        "lr_adam": 0.006,
        "quadratic": {"seeds": list(range(10)), "d1": 128, "d2": 128, "q": 48, "alpha_w": 1.3, "alpha_sigma": 0.5},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "local", "full"], default="smoke")
    parser.add_argument("--task", choices=["all", "curvature", "quadratic", "plots", "report"], default="all")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _mode_defaults(args.mode)
    out_dir = args.out or Path("runs") / "muon_curvature" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    cfg = {**cfg, "mode": args.mode, "device": str(device)}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.task in {"plots", "report"}:
        if args.task == "plots":
            make_plots(out_dir)
        else:
            write_report(out_dir, cfg)
        return

    store = Store(out_dir)
    if args.task in {"all", "curvature"}:
        m = cfg["model"]
        model_cfg = ModelConfig(
            vocab_size=int(m["vocab_size"]),
            context_length=int(m["context_length"]),
            model_dim=int(m["model_dim"]),
            num_heads=int(m["num_heads"]),
            num_layers=int(m["num_layers"]),
            ff_dim=int(m["ff_dim"]),
        )
        run_curvature_probe(
            store,
            device=device,
            model_cfg=model_cfg,
            seeds=list(cfg["seeds"]),
            zipf_exponents=list(cfg["zipf_exponents"]),
            steps=int(cfg["steps"]),
            log_every=int(cfg["log_every"]),
            nds_every=int(cfg["nds_every"]),
            train_n=int(cfg["train_n"]),
            val_n=int(cfg["val_n"]),
            batch_size=int(cfg["batch_size"]),
            num_topics=int(cfg["num_topics"]),
            num_classes=int(cfg["num_classes"]),
            lr_muon=float(cfg["lr_muon"]),
            lr_muon_aux=float(cfg["lr_muon_aux"]),
            lr_adam=float(cfg["lr_adam"]),
        )
    if args.task in {"all", "quadratic"}:
        qcfg = cfg["quadratic"]
        run_structured_quadratic(
            store,
            seeds=list(qcfg["seeds"]),
            d1=int(qcfg["d1"]),
            d2=int(qcfg["d2"]),
            q=int(qcfg["q"]),
            alpha_w=float(qcfg["alpha_w"]),
            alpha_sigma=float(qcfg["alpha_sigma"]),
        )
    store.write()
    if not args.skip_plots:
        make_plots(out_dir)
    if not args.skip_report:
        write_report(out_dir, cfg)
    print(f"Wrote Muon curvature research outputs to {out_dir}")


if __name__ == "__main__":
    main()
