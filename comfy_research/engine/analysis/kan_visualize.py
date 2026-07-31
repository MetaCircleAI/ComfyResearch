"""Render pykan ``KAN.plot()`` as PNG for the visualize_kan canvas node."""

from __future__ import annotations

import base64
import io
import tempfile
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import HTTPException

from comfy_research.engine.analysis.activation_collect import resolve_model_upstream_of_model_output
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.models.kan_model_build import build_kan_for_plot
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.runs.trainer_run import (
    _apply_linear_map,
    _incoming,
    _memorization_b_xy_numpy,
    _memorization_b_vocab_size,
    _node_map,
    _random_linear_weights,
    _resolve_linear_dataset_mlp_dims,
    _sample_teacher_input_tensor,
    _scalar_float,
    _scalar_int,
    _scalar_str,
    load_model_weights_from_checkpoint_b64,
)
from comfy_research.schemas.graph import Edge, Node, NodeKind


def _pick1(x: Any, default: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else default
    return x if x is not None else default


def _checkpoint_b64_from_node(ckpt: Node) -> str:
    dd: dict[str, Any] = ckpt.data or {}
    b64 = str(dd.get("checkpoint_b64") or "").strip()
    if not b64:
        b64 = str(dd.get("memoryCheckpoint_b64") or "").strip()
    return b64


def _incoming_first_edge(
    edges: list[Edge], nmap: dict[str, Node], target_id: str, target_handle: str
) -> tuple[Edge, Node] | None:
    th = target_handle.strip()
    for e in edges:
        if e.target != target_id or (e.targetHandle or "").strip() != th:
            continue
        n = nmap.get(e.source)
        if n is not None:
            return e, n
    return None


def _clamp_plot_batch_size(n: int) -> int:
    if n < 8:
        return 8
    if n > 4096:
        return 4096
    return n


def _sample_rows_from_dataset(
    edges: list[Edge],
    nmap: dict[str, Node],
    dataset_node: Node,
    dataset_edge: Edge,
    kan_node: Node,
    rng: np.random.Generator,
    *,
    prefer_test_split: bool = False,
) -> np.ndarray:
    """Draw ``x`` with shape ``(N, input_dim)`` matching the KAN, using the same rules as MSE training."""
    md_k: dict[str, Any] = kan_node.data or {}
    dd: dict[str, Any] = dataset_node.data or {}
    sh = (dataset_edge.sourceHandle or "").strip()
    if sh == "dataset":
        use_test = bool(prefer_test_split)
    elif sh == "test_dataset":
        use_test = True
    elif sh == "train_dataset":
        use_test = False
    else:
        raise HTTPException(
            status_code=400,
            detail="visualize_kan dataset edge must use the dataset node's unified `dataset` output "
            "(or legacy train_dataset / test_dataset).",
        )
    raw_n = _scalar_int(dd.get("testSize" if use_test else "trainSize"), 0 if use_test else 800)
    if raw_n < 1:
        split = "test" if use_test else "train"
        raise HTTPException(
            status_code=400,
            detail=f"Dataset {split} split has no samples (set {'testSize' if use_test else 'trainSize'} >= 1).",
        )
    n = _clamp_plot_batch_size(raw_n)

    if dataset_node.type == NodeKind.memorization_b_dataset:
        input_dim, _output_dim = _resolve_linear_dataset_mlp_dims(dd, md_k, dataset_node.type, model_kind=kan_node.type)
        vocab = _memorization_b_vocab_size(dd)
        out_dist = _scalar_str(dd.get("outputDistribution"), "uniform_class_probs")
        mem_alpha = _scalar_float(dd.get("alpha"), 1.0)
        x_np, _ = _memorization_b_xy_numpy(rng, n, vocab, out_dist, mem_alpha)
        return x_np.astype(np.float32)
    if dataset_node.type in (NodeKind.linear_dataset, NodeKind.memorization_a_dataset):
        input_dim, output_dim = _resolve_linear_dataset_mlp_dims(dd, md_k, dataset_node.type, model_kind=kan_node.type)
        input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
        out_dist = _scalar_str(dd.get("outputDistribution"), "additive_gaussian")
        noise_level = _scalar_float(dd.get("noiseLevel"), 0.25)
        additive = out_dist == "additive_gaussian"
        sigma = noise_level if additive else 0.0
        x_np = sample_inputs(rng, n, input_dim, input_dist)
        w = _random_linear_weights(rng, input_dim, output_dim)
        _ = _apply_linear_map(x_np, w, sigma, additive, rng)
        return x_np.astype(np.float32)

    if dataset_node.type == NodeKind.symbolic_func_dataset:
        input_dim, _ = _resolve_linear_dataset_mlp_dims(dd, md_k, dataset_node.type, model_kind=kan_node.type)
        input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
        x_np = sample_inputs(rng, n, input_dim, input_dist)
        y_fn = build_y_numpy_fn(dd)
        _ = y_fn(x_np)
        return x_np.astype(np.float32)

    if dataset_node.type == NodeKind.teacher_dataset:
        teacher_src = _incoming(edges, nmap, dataset_node.id, "model")
        if teacher_src is None or teacher_src.type != NodeKind.mlp_model:
            raise HTTPException(
                status_code=400,
                detail="teacher_dataset for visualize_kan must have an mlp_model on its model socket.",
            )
        tmd: dict[str, Any] = teacher_src.data or {}
        teacher_in = _scalar_int(tmd.get("inputDim"), 10)
        kan_in = _scalar_int(md_k.get("inputDim"), 10)
        if teacher_in != kan_in:
            raise HTTPException(
                status_code=400,
                detail=f"Teacher MLP input dim ({teacher_in}) must match KAN input dim ({kan_in}) for dataset plot.",
            )
        handle = "test_input" if use_test else "train_input"
        x_np, _ = _sample_teacher_input_tensor(edges, nmap, dataset_node, handle, None)
        if x_np.shape[0] > n:
            x_np = x_np[:n]
        return x_np.astype(np.float32)

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported dataset type for visualize_kan: {dataset_node.type}.",
    )


def _build_kan_from_kan_node(kan_node: Node) -> Any:
    md: dict[str, Any] = kan_node.data or {}
    input_dim = _scalar_int(md.get("inputDim"), 10)
    output_dim = _scalar_int(md.get("outputDim"), 1)
    depth = int(_pick1(md.get("depth"), 2))
    width = int(_pick1(md.get("width"), 5))
    grid = int(_pick1(md.get("grid"), 3))
    spline_k = int(_pick1(md.get("k"), 3))
    base_fun = str(_pick1(md.get("baseFun"), "silu")).strip().lower()
    seed = _scalar_int(md.get("seed"), 0)
    return build_kan_for_plot(
        input_dim,
        output_dim,
        depth,
        width,
        grid,
        spline_k,
        seed,
        base_fun,
    )


def run_kan_plot(
    nodes: list[Node],
    edges: list[Edge],
    visualize_kan_node_id: str,
    sample_count: int = 256,
    plot_scale: float = 0.35,
    plot_metric: Literal["backward", "forward_n", "forward_u"] = "backward",
    dpi: int = 120,
) -> dict[str, Any]:
    """Build or load a KAN, run a forward for activations, call ``plot``, return PNG base64."""
    nmap = _node_map(nodes)
    viz_id = visualize_kan_node_id.strip()
    viz = nmap.get(viz_id)
    if viz is None:
        raise HTTPException(status_code=404, detail="visualize_kan node not found.")
    if viz.type != NodeKind.visualize_kan:
        raise HTTPException(status_code=400, detail="Target is not a visualize_kan node.")

    src = _incoming(edges, nmap, viz_id, "model")
    if src is None:
        raise HTTPException(status_code=400, detail="Connect a kan_model or model_checkpoint (from a trained KAN).")

    kan_node: Node | None = None
    ckpt_b64: str | None = None

    if src.type == NodeKind.kan_model:
        kan_node = src
    elif src.type == NodeKind.model_checkpoint:
        upstream = resolve_model_upstream_of_model_output(nmap, edges, src.id)
        if upstream is None or upstream.type != NodeKind.kan_model:
            raise HTTPException(
                status_code=400,
                detail="Model checkpoint must trace back to a kan_model (trainer → KAN → checkpoint).",
            )
        kan_node = upstream
        b64 = _checkpoint_b64_from_node(src)
        if not b64:
            raise HTTPException(
                status_code=400,
                detail="Model checkpoint has no weights yet. Train, load from memory, or load from file first.",
            )
        ckpt_b64 = b64
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model input for visualize_kan: {src.type}. Use kan_model or model_checkpoint.",
        )

    assert kan_node is not None
    ds_pair = _incoming_first_edge(edges, nmap, viz_id, "dataset")
    uses_dataset = ds_pair is not None

    if not uses_dataset:
        if sample_count < 8 or sample_count > 4096:
            raise HTTPException(status_code=400, detail="sample_count must be between 8 and 4096.")
    if plot_scale <= 0 or plot_scale > 2.0:
        raise HTTPException(status_code=400, detail="plot_scale must be in (0, 2].")
    if dpi < 72 or dpi > 300:
        raise HTTPException(status_code=400, detail="dpi must be between 72 and 300.")

    model = _build_kan_from_kan_node(kan_node)
    if ckpt_b64:
        load_model_weights_from_checkpoint_b64(model, ckpt_b64)

    model.eval()
    md_k: dict[str, Any] = kan_node.data or {}
    in_dim = _scalar_int(md_k.get("inputDim"), 10)

    if uses_dataset:
        ds_edge, ds_node = ds_pair
        viz_dd: dict[str, Any] = viz.data or {}
        split_raw = str(viz_dd.get("datasetSampleSplit") or "train").strip().lower()
        prefer_test = split_raw == "test"
        if not has_capability(ds_node.type, "activation_sample_dataset"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "visualize_kan dataset must be linear_dataset, memorization_a_dataset, memorization_b_dataset, "
                    "symbolic_func_dataset, or teacher_dataset."
                ),
            )
        dd_ds: dict[str, Any] = ds_node.data or {}
        seed = _scalar_int(dd_ds.get("seed"), 0)
        rng = np.random.default_rng(seed)
        x_np = _sample_rows_from_dataset(
            edges, nmap, ds_node, ds_edge, kan_node, rng, prefer_test_split=prefer_test
        )
        if x_np.ndim != 2 or x_np.shape[1] != in_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Sampled inputs have width {x_np.shape[1] if x_np.ndim == 2 else '?'}"
                f", but KAN expects input_dim={in_dim}.",
            )
        x = torch.from_numpy(x_np)
    else:
        x = torch.randn(sample_count, in_dim)

    plt.close("all")
    try:
        with torch.enable_grad():
            model(x)
        with tempfile.TemporaryDirectory(prefix="cr_kan_plot_") as tmp:
            model.plot(
                folder=tmp,
                scale=float(plot_scale),
                metric=str(plot_metric),
                tick=False,
                sample=False,
            )
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=int(dpi), facecolor=fig.get_facecolor())
            plt.close(fig)
        raw = buf.getvalue()
        return {"plot_png_base64": base64.standard_b64encode(raw).decode("ascii")}
    except HTTPException:
        raise
    except Exception as e:
        plt.close("all")
        raise HTTPException(
            status_code=400,
            detail=f"KAN plot failed (pykan / matplotlib): {e}",
        ) from e
