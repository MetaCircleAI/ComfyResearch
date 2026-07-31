"""Interactive sampling and comparison endpoints for the diffusion reproducibility graph.

Model checkpoints stay in the canvas node. This module persists only generated
images, so reopening a project can restore a preview without committing model
weights or experiment artifacts to the repository.
"""
from __future__ import annotations

import base64
import io
import json
import os
import uuid
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.responses import StreamingResponse

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comfy_research.engine.datasets.vision_datasets_runtime import build_cifar10_arrays, denormalize_cifar10_images
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node
from comfy_research.engine.trainer.checkpoint import load_model_weights_from_checkpoint_b64
from comfy_research.schemas.graph import Edge, Node, NodeKind

router = APIRouter(prefix="/api", tags=["diffusion"])
_SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "runtime" / "diffusion_samples"


class _GraphRequest(BaseModel):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class SamplerRequest(_GraphRequest):
    sampler_node_id: str


class ObservableRequest(_GraphRequest):
    observable_node_id: str


def _graph(body: _GraphRequest) -> tuple[dict[str, Node], list[Edge]]:
    nodes = [Node.model_validate(raw) for raw in body.nodes]
    return {node.id: node for node in nodes}, [Edge.model_validate(raw) for raw in body.edges]


def _incoming(edges: list[Edge], nodes: dict[str, Node], target: str, handle: str) -> Node | None:
    for edge in reversed(edges):
        if edge.target == target and edge.targetHandle == handle:
            return nodes.get(edge.source)
    return None


def _required(edges: list[Edge], nodes: dict[str, Node], target: str, handle: str) -> Node:
    node = _incoming(edges, nodes, target, handle)
    if node is None:
        raise HTTPException(status_code=400, detail=f"Missing connection to {handle}.")
    return node


def _active_checkpoint(data: dict[str, Any]) -> str:
    source = str(data.get("checkpointSource") or "memory")
    if source == "file":
        return str(data.get("checkpoint_b64") or "")
    return str(data.get("memoryCheckpoint_b64") or data.get("checkpoint_b64") or "")


def _runtime_device() -> torch.device:
    """Prefer CUDA for interactive sampling and image-comparison workloads."""
    # Must be set before the first CUDA matmul when deterministic algorithms are enabled.
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_unet(
    sampler: Node,
    nodes: dict[str, Node],
    edges: list[Edge],
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    checkpoint = _required(edges, nodes, sampler.id, "checkpoint")
    if checkpoint.type != NodeKind.model_checkpoint:
        raise HTTPException(status_code=400, detail="Sampler checkpoint input must be model_checkpoint.")
    trainer = _required(edges, nodes, checkpoint.id, "model_checkpoint")
    model_node = _required(edges, nodes, trainer.id, "model")
    dataset_node = _required(edges, nodes, trainer.id, "dataset")
    if model_node.type != NodeKind.unet_ddpm_model or dataset_node.type != NodeKind.cifar10_dataset:
        raise HTTPException(status_code=400, detail="Sampler requires a CIFAR-10 UNet-DDPM trainer checkpoint.")
    checkpoint_b64 = _active_checkpoint(dict(checkpoint.data or {}))
    if not checkpoint_b64:
        raise HTTPException(status_code=400, detail="Load or train a checkpoint before sampling.")
    model = build_model_for_node(model_node, ModelBuildContext(input_channels=3, image_size=32))
    load_model_weights_from_checkpoint_b64(model, checkpoint_b64)
    return model, dict(dataset_node.data or {}), model_node.id


def _ddim_sample_steps(
    model: torch.nn.Module,
    noise: torch.Tensor,
    *,
    steps: int,
) -> Iterator[tuple[int, int, torch.Tensor | None]]:
    """Run deterministic DDIM sampling and expose a progress event per denoise step."""
    total = int(getattr(model, "max_timesteps", 1000))
    schedule = torch.linspace(total - 1, 0, min(max(2, steps), total), device=noise.device).long()
    betas = torch.linspace(1e-4, 0.02, total, device=noise.device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    x = noise
    model.eval()
    with torch.no_grad():
        for index, t in enumerate(schedule):
            tt = torch.full((x.shape[0],), int(t.item()), device=x.device, dtype=torch.long)
            a_t = alpha_bar[t]
            eps = model(x, tt)
            x0 = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
            if index + 1 == len(schedule):
                x = x0
            else:
                a_next = alpha_bar[schedule[index + 1]]
                x = torch.sqrt(a_next) * x0 + torch.sqrt(1.0 - a_next) * eps
            is_final = index + 1 == len(schedule)
            yield index + 1, len(schedule), x.clamp(-1.0, 1.0) if is_final else None


def _ddim_sample(model: torch.nn.Module, noise: torch.Tensor, *, steps: int) -> torch.Tensor:
    """Compatibility wrapper for non-streaming callers and unit tests."""
    result: torch.Tensor | None = None
    for _, _, final in _ddim_sample_steps(model, noise, steps=steps):
        if final is not None:
            result = final
    if result is None:  # Defensive: schedules always have at least two steps.
        raise RuntimeError("DDIM sampler produced no final image batch.")
    return result


def _paths(run_id: str) -> tuple[Path, Path]:
    safe = "".join(char for char in run_id if char.isalnum() or char in "-_")
    return _SAMPLE_DIR / f"{safe}.npz", _SAMPLE_DIR / f"{safe}.json"


def _save_run(images: np.ndarray, metadata: dict[str, Any]) -> str:
    _SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"diffusion-{uuid.uuid4().hex[:12]}"
    data_path, meta_path = _paths(run_id)
    np.savez_compressed(data_path, images=np.asarray(images, dtype=np.float32))
    meta_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return run_id


def _load_run(run_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    data_path, meta_path = _paths(str(run_id))
    if not data_path.is_file() or not meta_path.is_file():
        raise HTTPException(status_code=404, detail="Generated sample run not found; sample again.")
    with np.load(data_path) as payload:
        images = np.asarray(payload["images"], dtype=np.float32)
    return images, json.loads(meta_path.read_text(encoding="utf-8"))


def _image_grid(images: np.ndarray, *, columns: int = 4) -> str:
    show = np.asarray(images[:16], dtype=np.float32)
    rows = max(1, int(np.ceil(len(show) / columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    for axis, image in zip(np.asarray(axes).reshape(-1), show):
        axis.imshow(np.transpose(image, (1, 2, 0)))
        axis.axis("off")
    for axis in np.asarray(axes).reshape(-1)[len(show):]:
        axis.axis("off")
    buffer = io.BytesIO()
    fig.tight_layout(pad=0.1)
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _figure_data_url(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.tight_layout(pad=0.25)
    figure.savefig(buffer, format="png", dpi=120)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pair_grid(left: np.ndarray, right: np.ndarray, *, title: str) -> str:
    """Render corresponding image pairs as evidence, not just a scalar metric."""
    count = min(8, len(left), len(right))
    figure, axes = plt.subplots(max(1, count), 2, figsize=(4, max(1, count) * 2))
    flat_axes = np.asarray(axes).reshape(max(1, count), 2)
    for index in range(count):
        flat_axes[index, 0].imshow(np.transpose(left[index], (1, 2, 0)))
        flat_axes[index, 1].imshow(np.transpose(right[index], (1, 2, 0)))
        flat_axes[index, 0].set_ylabel(str(index), rotation=0, labelpad=10, va="center")
    for axis in flat_axes.reshape(-1):
        axis.axis("off")
    flat_axes[0, 0].set_title(title.split(" / ")[0])
    flat_axes[0, 1].set_title(title.split(" / ")[-1])
    return _figure_data_url(figure)


def _histogram(values: np.ndarray, *, title: str, xlabel: str) -> str:
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        finite = np.asarray([0.0], dtype=np.float32)
    lo, hi = float(finite.min()), float(finite.max())
    if np.isclose(lo, hi):
        padding = max(abs(lo) * 0.01, 1e-3)
        value_range: tuple[float, float] | None = (lo - padding, hi + padding)
    else:
        value_range = None
    figure, axis = plt.subplots(figsize=(4.2, 2.6))
    axis.hist(finite, bins=min(20, max(5, len(finite))), range=value_range, color="#657fcf", edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    return _figure_data_url(figure)


def _sample_context(body: SamplerRequest) -> tuple[torch.nn.Module, dict[str, Any], str, int, int, int, torch.device, torch.Tensor]:
    nodes, edges = _graph(body)
    sampler = nodes.get(body.sampler_node_id)
    if sampler is None or sampler.type != NodeKind.deterministic_diffusion_sampler:
        raise HTTPException(status_code=404, detail="Deterministic diffusion sampler not found.")
    model, dataset_data, model_id = _resolve_unet(sampler, nodes, edges)
    data = dict(sampler.data or {})
    count = max(1, min(int(data.get("sampleCount") or 64), 512))
    seed = int(data.get("noiseSeed") or 0)
    steps = max(2, int(data.get("numSteps") or 50))
    device = _runtime_device()
    model.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (count, int(model.in_channels), int(model.image_size), int(model.image_size)),
        generator=generator,
        device=device,
    )
    return model, dataset_data, model_id, count, seed, steps, device, noise


def _sample_result(
    model: torch.nn.Module,
    dataset_data: dict[str, Any],
    model_id: str,
    count: int,
    seed: int,
    steps: int,
    device: torch.device,
    generated: torch.Tensor,
) -> dict[str, Any]:
    images = denormalize_cifar10_images(generated.cpu().numpy(), "minus_one_to_one")
    metadata = {
        "modelId": model_id,
        "sampleCount": count,
        "noiseSeed": seed,
        "numSteps": steps,
        "device": str(device),
        "dataset": dataset_data,
    }
    run_id = _save_run(images, metadata)
    return {"runId": run_id, "metadata": metadata, "previewGrid": _image_grid(images)}


@router.post("/diffusion/sampler")
def sample(body: SamplerRequest) -> dict[str, Any]:
    context = _sample_context(body)
    model, dataset_data, model_id, count, seed, steps, device, noise = context
    return _sample_result(
        model,
        dataset_data,
        model_id,
        count,
        seed,
        steps,
        device,
        _ddim_sample(model, noise, steps=steps),
    )


@router.post("/diffusion/sampler/stream")
def sample_stream(body: SamplerRequest) -> StreamingResponse:
    """Stream DDIM sampling progress before returning the persisted sample result."""

    def events() -> Iterator[str]:
        try:
            model, dataset_data, model_id, count, seed, steps, device, noise = _sample_context(body)
            yield json.dumps({"type": "progress", "step": 0, "total": steps}) + "\n"
            for step, total, generated in _ddim_sample_steps(model, noise, steps=steps):
                if generated is None:
                    yield json.dumps({"type": "progress", "step": step, "total": total}) + "\n"
                    continue
                result = _sample_result(
                    model,
                    dataset_data,
                    model_id,
                    count,
                    seed,
                    steps,
                    device,
                    generated,
                )
                yield json.dumps({"type": "complete", **result}) + "\n"
        except HTTPException as exc:
            yield json.dumps({"type": "error", "detail": str(exc.detail)}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


@router.get("/diffusion/preview")
def preview(run_id: str) -> dict[str, Any]:
    images, metadata = _load_run(run_id)
    return {"runId": run_id, "metadata": metadata, "previewGrid": _image_grid(images)}


@router.post("/diffusion/paired-similarity")
def paired_similarity(body: ObservableRequest) -> dict[str, Any]:
    nodes, edges = _graph(body)
    observable = nodes.get(body.observable_node_id)
    if observable is None:
        raise HTTPException(status_code=404, detail="Paired similarity node not found.")
    run_a, _ = _load_run(str((_required(edges, nodes, observable.id, "sampler_a").data or {}).get("runId") or ""))
    run_b, _ = _load_run(str((_required(edges, nodes, observable.id, "sampler_b").data or {}).get("runId") or ""))
    if run_a.shape != run_b.shape:
        raise HTTPException(status_code=400, detail="Sampler outputs are not aligned.")
    device = _runtime_device()
    a = torch.from_numpy(run_a).to(device)
    b = torch.from_numpy(run_b).to(device)
    mae = (a - b).abs().mean(dim=(1, 2, 3)).cpu().numpy()
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
    return {
        "meanMae": float(mae.mean()),
        "meanMse": float(mse.mean()),
        "mae": mae.tolist(),
        "mse": mse.tolist(),
        "imageGrid": _pair_grid(run_a, run_b, title="Seed 0 / Seed 1"),
        "histogramPng": _histogram(mae, title="Paired MAE histogram", xlabel="MAE"),
        "device": str(device),
    }


@router.post("/diffusion/rp-score")
def rp_score(body: ObservableRequest) -> dict[str, Any]:
    nodes, edges = _graph(body)
    observable = nodes.get(body.observable_node_id)
    if observable is None:
        raise HTTPException(status_code=404, detail="RP score node not found.")
    run_a, _ = _load_run(str((_required(edges, nodes, observable.id, "sampler_a").data or {}).get("runId") or ""))
    run_b, _ = _load_run(str((_required(edges, nodes, observable.id, "sampler_b").data or {}).get("runId") or ""))
    device = _runtime_device()
    a = F.normalize(torch.from_numpy(run_a.reshape(len(run_a), -1)).to(device), dim=1)
    b = F.normalize(torch.from_numpy(run_b.reshape(len(run_b), -1)).to(device), dim=1)
    similarities = (a * b).sum(dim=1).cpu().numpy()
    threshold = float((observable.data or {}).get("threshold") or 0.95)
    return {
        "rp": float(np.mean(similarities >= threshold)),
        "meanSimilarity": float(similarities.mean()),
        "similarities": similarities.tolist(),
        "threshold": threshold,
        "backend": "pixel_cosine",
        "histogramPng": _histogram(similarities, title="Paired similarity histogram", xlabel="pixel cosine similarity"),
        "device": str(device),
    }


@router.post("/diffusion/nearest-train")
def nearest_train(body: ObservableRequest) -> dict[str, Any]:
    nodes, edges = _graph(body)
    observable = nodes.get(body.observable_node_id)
    if observable is None:
        raise HTTPException(status_code=404, detail="Nearest-train node not found.")
    generated = _required(edges, nodes, observable.id, "generated")
    dataset = _required(edges, nodes, observable.id, "train_dataset")
    if dataset.type != NodeKind.cifar10_dataset:
        raise HTTPException(status_code=400, detail="Nearest-train currently supports CIFAR-10 only.")
    images, _ = _load_run(str((generated.data or {}).get("runId") or ""))
    data = dict(dataset.data or {})
    train_size = int(data.get("trainSize") or 2048)
    train, _, _, _ = build_cifar10_arrays(data, train_size, 0, np.random.default_rng(int(data.get("subsetSeed") or 0)))
    train = denormalize_cifar10_images(train, str(data.get("normalize") or "zero_one"))
    device = _runtime_device()
    generated_features = F.normalize(torch.from_numpy(images.reshape(len(images), -1)).to(device), dim=1)
    train_features = F.normalize(torch.from_numpy(train.reshape(len(train), -1)).to(device), dim=1)
    best = torch.full((len(images),), -float("inf"), device=device)
    best_index = torch.zeros((len(images),), dtype=torch.long, device=device)
    for start in range(0, len(train_features), 2048):
        values, indices = (generated_features @ train_features[start:start + 2048].T).max(dim=1)
        mask = values > best
        best[mask] = values[mask]
        best_index[mask] = indices[mask] + start
    threshold = float((observable.data or {}).get("glThreshold") or 0.95)
    return {
        "glScore": float((best < threshold).float().mean()),
        "nearestSimilarity": best.cpu().tolist(),
        "nearestIndex": best_index.cpu().tolist(),
        "threshold": threshold,
        "backend": "pixel_cosine_exact",
        "imageGrid": _pair_grid(images, train[best_index.cpu().numpy()], title="Generated / Nearest train"),
        "histogramPng": _histogram(best.cpu().numpy(), title="Nearest-train similarity", xlabel="pixel cosine similarity"),
        "device": str(device),
    }
