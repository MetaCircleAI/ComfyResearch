"""Dataset runtime: per-verb registries for dataset dispatch (Step 2a).

Migrates two deterministic verbs off trainer_run's node.type chains:
toy-language LM materialization and declared split-size resolution. Per-verb
registries (not a protocol class). MUST NOT import trainer_run (one-way edge).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from fastapi import HTTPException

from comfy_research.schemas.graph import Edge, Node, NodeKind
from comfy_research.engine.datasets.toy_language_dyck_runtime import build_dyck_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_external_runtime import (
    build_cogs_arrays_from_seed,
    build_listops_arrays_from_seed,
    build_scan_arrays_from_seed,
    phi_style_lm_from_seed,
    tiny_shakespeare_lm_from_seed,
    tiny_stories_lm_from_seed,
)
from comfy_research.engine.datasets.toy_language_formal_runtime import build_formal_suite_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_ngram_runtime import build_ngram_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_pcfg_runtime import build_pcfg_lm_arrays_from_seed
from comfy_research.engine.datasets.toy_language_physics_lm_runtime import (
    build_biography_lm_arrays_from_seed,
    build_multi_hop_fact_chain_lm_arrays_from_seed,
    build_relation_tuple_lm_arrays_from_seed,
    build_synthetic_playground_lm_arrays_from_seed,
)
from comfy_research.engine.datasets.synthetic_dataset_builders import (
    _build_bigram_low_rank_arrays,
    _build_circular_motion_arrays,
    _build_kepler_2d_arrays,
    _build_uniform_linear_motion_arrays,
    _linear_like_targets_from_x,
    _memorization_b_vocab_size,
    _memorization_b_xy_numpy,
    _memorization_class_probs,
    _sample_memorization_labels,
    _unigram_token_probs,
)
from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs


def _scalar_int(x: Any, default: int = 0) -> int:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return int(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _scalar_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, list):
        if not x:
            return default
        try:
            return float(x[0])
        except (TypeError, ValueError):
            return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


def _incoming_node(edges: list[Edge], nmap: dict[str, Node], target_id: str, handle: str) -> Node | None:
    want = handle or ""
    matches: list[Node] = []
    for e in edges:
        if e.target != target_id:
            continue
        th = e.targetHandle or ""
        if th != want:
            continue
        n = nmap.get(e.source)
        if n is not None:
            matches.append(n)
    if len(matches) > 1:
        kinds = ", ".join(str(n.type) for n in matches)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple edges target node {target_id!r} handle {handle!r} ({kinds}). "
                "Disconnect duplicate wires so exactly one upstream node feeds this socket."
            ),
        )
    return matches[0] if matches else None


# --- Toy-language LM materialization ---
TOY_LM_BUILDERS: dict[
    NodeKind, Callable[[dict[str, Any], int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
] = {
    NodeKind.pcfg_dataset: build_pcfg_lm_arrays_from_seed,
    NodeKind.dyck_dataset: build_dyck_lm_arrays_from_seed,
    NodeKind.ngram_language_dataset: build_ngram_lm_arrays_from_seed,
    NodeKind.formal_language_suite_dataset: build_formal_suite_lm_arrays_from_seed,
    NodeKind.scan_dataset: build_scan_arrays_from_seed,
    NodeKind.cogs_dataset: build_cogs_arrays_from_seed,
    NodeKind.listops_dataset: build_listops_arrays_from_seed,
    NodeKind.tinystories_dataset: tiny_stories_lm_from_seed,
    NodeKind.tinyshakespeare_lm_dataset: tiny_shakespeare_lm_from_seed,
    NodeKind.phi1_style_dataset: phi_style_lm_from_seed,
    NodeKind.biography_lm_dataset: build_biography_lm_arrays_from_seed,
    NodeKind.relation_tuple_dataset: build_relation_tuple_lm_arrays_from_seed,
    NodeKind.synthetic_playground_dataset: build_synthetic_playground_lm_arrays_from_seed,
    NodeKind.multi_hop_fact_chain_dataset: build_multi_hop_fact_chain_lm_arrays_from_seed,
}


def _materialize_toy_language_lm_numpy(
    ds_kind: NodeKind,
    dd_train: dict[str, Any],
    train_n: int,
    test_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        builder = TOY_LM_BUILDERS[ds_kind]
    except KeyError:
        raise HTTPException(status_code=500, detail=f"Internal: unsupported toy language dataset {ds_kind}")
    return builder(dd_train, train_n, test_n)


# --- Declared split-size resolution ---
def _declared_dataset_mixer_split_sizes(
    node: Node, edges: list[Edge], nmap: dict[str, Node], *, _visited: set[str] | None = None
) -> tuple[int, int]:
    dd: dict[str, Any] = node.data or {}
    raw_tr = dd.get("trainTotalSamples")
    raw_te = dd.get("testTotalSamples")
    train_size = _scalar_int(raw_tr, 800) if raw_tr is not None else _scalar_int(dd.get("totalSamples"), 800)
    test_size = _scalar_int(raw_te, 0) if raw_te is not None else _scalar_int(dd.get("totalSamples"), 0)
    return train_size, test_size


def _declared_dataset_mixer_b_split_sizes(
    node: Node, edges: list[Edge], nmap: dict[str, Node], *, _visited: set[str] | None = None
) -> tuple[int, int]:
    if _visited is None:
        _visited = set()
    if node.id in _visited:
        raise HTTPException(status_code=400, detail="dataset_mixer_b graph contains a cycle.")
    _visited.add(node.id)
    ds_a = _incoming_node(edges, nmap, node.id, "dataset_a")
    ds_b = _incoming_node(edges, nmap, node.id, "dataset_b")
    if ds_a is None or ds_b is None:
        raise HTTPException(
            status_code=400, detail="dataset_mixer_b requires both dataset_a and dataset_b connections."
        )
    tr_a, te_a = _declared_dataset_split_sizes(ds_a, edges, nmap, _visited=set(_visited))
    tr_b, te_b = _declared_dataset_split_sizes(ds_b, edges, nmap, _visited=set(_visited))
    if tr_a != tr_b or te_a != te_b:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer_b requires dataset_a and dataset_b to have matching train/test sizes.",
        )
    return tr_a, te_a


def _declared_modular_addition_split_sizes(
    node: Node, edges: list[Edge], nmap: dict[str, Node], *, _visited: set[str] | None = None
) -> tuple[int, int]:
    dd: dict[str, Any] = node.data or {}
    p = max(2, _scalar_int(dd.get("modulus"), 59))
    frac = _scalar_float(dd.get("trainFraction"), 0.3)
    total = int(p) * int(p)
    n_train = int(round(frac * total))
    n_train = min(max(n_train, 1), total)
    return n_train, total - n_train


def _default_declared_dataset_split_sizes(node: Node) -> tuple[int, int]:
    dd: dict[str, Any] = node.data or {}
    return _scalar_int(dd.get("trainSize"), 800), _scalar_int(dd.get("testSize"), 0)


def _declared_teacher_dataset_split_sizes(
    node: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
    *,
    _visited: set[str] | None = None,
) -> tuple[int, int]:
    """Use the wired input samplers as the source of truth for teacher splits.

    ``teacher_dataset`` does not expose its own sample-count fields: its train
    and test tensors are generated by ``input_sampler`` nodes. Falling back
    to the generic dataset default made the trainer's minibatch sampler think
    a fixed teacher dataset contained 800 rows even when its input sampler
    generated a different number.
    """
    dd: dict[str, Any] = node.data or {}

    def _size_for(handle: str, fallback_key: str, default: int) -> int:
        sampler_edge = next(
            (
                edge
                for edge in edges
                if edge.target == node.id and edge.targetHandle == handle
            ),
            None,
        )
        sampler = nmap.get(sampler_edge.source) if sampler_edge is not None else None
        if sampler is not None and sampler.type == NodeKind.input_sampler:
            return _scalar_int((sampler.data or {}).get("numSamples"), default)
        return _scalar_int(dd.get(fallback_key), default)

    return (
        _size_for("train_input", "trainSize", 800),
        _size_for("test_input", "testSize", 0),
    )


SPLIT_SIZE_HANDLERS: dict[NodeKind, Callable[..., tuple[int, int]]] = {
    NodeKind.dataset_mixer: _declared_dataset_mixer_split_sizes,
    NodeKind.dataset_mixer_b: _declared_dataset_mixer_b_split_sizes,
    NodeKind.modular_addition_dataset: _declared_modular_addition_split_sizes,
    NodeKind.teacher_dataset: _declared_teacher_dataset_split_sizes,
}


def _declared_dataset_split_sizes(
    node: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
    *,
    _visited: set[str] | None = None,
) -> tuple[int, int]:
    handler = SPLIT_SIZE_HANDLERS.get(node.type)
    if handler is not None:
        return handler(node, edges, nmap, _visited=_visited)
    return _default_declared_dataset_split_sizes(node)


# --- Non-vision leaf dataset sampling ---
def _vector_scalar_config(node: Node) -> tuple[int, int, str, str, float, float, bool, float]:
    """Pure parse of the shared vector scalar block (trainer_run 2457-2464). No validation, no raise."""
    dd: dict[str, Any] = node.data or {}
    input_dim = _scalar_int(dd.get("inputDim"), 1)
    output_dim = _scalar_int(dd.get("outputDim"), 1)
    input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
    out_dist = _scalar_str(dd.get("outputDistribution"), "additive_gaussian")
    alpha = _scalar_float(dd.get("alpha"), 1.0)
    noise_level = _scalar_float(dd.get("noiseLevel"), 0.25)
    additive = out_dist == "additive_gaussian"
    sigma = noise_level if additive else 0.0
    return input_dim, output_dim, input_dist, out_dist, alpha, noise_level, additive, sigma


_ELSE_RAISE_DETAIL = (
    "dataset_mixer sources must be linear_dataset, random_noise_dataset, memorization_b_dataset, "
    "symbolic_func_dataset, uniform_linear_motion_dataset, kepler_2d_dataset, diffusion_pde_dataset, reaction_diffusion_dataset, "
    "advection_dataset, dataset_mixer, or dataset_mixer_b. "
    "memorization_a_dataset is only allowed when the trainer uses cross_entropy_loss (not MSE)."
)


def _leaf_uniform_linear_motion(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    x_np, y_np, _, _ = _build_uniform_linear_motion_arrays(rng, dd, dd, sample_count, 0)
    return x_np, y_np


def _leaf_kepler(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    x_np, y_np, _, _ = _build_kepler_2d_arrays(rng, dd, dd, sample_count, 0)
    return x_np, y_np


def _leaf_pde(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    x_np, y_np, _, _ = build_pde_field_arrays(node.type, rng, dd, dd, sample_count, 0)
    return x_np, y_np

def _leaf_memorization_a(node, sample_count, rng, *, trainer_task, eff_task):
    if eff_task != "cross_entropy_dense":
        raise HTTPException(
            status_code=400,
            detail=(
                "memorization_a_dataset is only valid with cross_entropy_loss (random discrete labels). "
                "Remove it from dataset_mixer for MSE or diffusion training, or switch the trainer loss to "
                "cross_entropy_loss."
            ),
        )
    input_dim, output_dim, input_dist, out_dist, alpha, _, _, _ = _vector_scalar_config(node)
    x_np = sample_inputs(rng, sample_count, input_dim, input_dist)
    y_np = _sample_memorization_labels(rng, sample_count, output_dim, out_dist, alpha)
    return x_np, y_np


def _leaf_memorization_b(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    _, _, _, out_dist, alpha, _, _, _ = _vector_scalar_config(node)
    if eff_task == "cross_entropy_dense":
        vocab = _memorization_b_vocab_size(dd)
        return _memorization_b_xy_numpy(rng, sample_count, vocab, out_dist, alpha)
    raise HTTPException(status_code=400, detail=_ELSE_RAISE_DETAIL)


def _leaf_random_noise(node, sample_count, rng, *, trainer_task, eff_task):
    input_dim, output_dim, input_dist, _, _, _, _, _ = _vector_scalar_config(node)
    x_np = sample_inputs(rng, sample_count, input_dim, input_dist)
    y_np = sample_inputs(rng, sample_count, output_dim, input_dist)
    return x_np, y_np


def _leaf_linear_like(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    input_dim, output_dim, input_dist, _, _, _, additive, sigma = _vector_scalar_config(node)
    x_np = sample_inputs(rng, sample_count, input_dim, input_dist)
    y_np = _linear_like_targets_from_x(dd, x_np, input_dim, output_dim, sigma=sigma, additive=additive, rng=rng)
    return x_np, y_np


def _leaf_symbolic(node, sample_count, rng, *, trainer_task, eff_task):
    dd = node.data or {}
    input_dim, _, input_dist, _, _, _, additive, sigma = _vector_scalar_config(node)
    x_np = sample_inputs(rng, sample_count, input_dim, input_dist)
    try:
        y_fn = build_y_numpy_fn(dd)
    except HTTPException as e:
        detail = str(getattr(e, "detail", "") or "")
        if "equationLatex is required" in detail:
            instance_title = _scalar_str(dd.get("instanceTitle"), "").strip() or node.id
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{detail} Triggered by symbolic_func_dataset '{instance_title}'. "
                    "If you are not intentionally using a symbolic dataset, inspect the dataset_mixer inputs and replace any "
                    "accidentally-added symbolic_func_dataset node."
                ),
            ) from e
        raise
    y_np = y_fn(x_np)
    if additive and sigma > 0:
        y_np = y_np + sigma * rng.standard_normal((y_np.shape[0], y_np.shape[1])).astype(np.float32)
    return x_np, y_np


LEAF_SAMPLE_HANDLERS: dict[NodeKind, Callable] = {
    NodeKind.uniform_linear_motion_dataset: _leaf_uniform_linear_motion,
    NodeKind.kepler_2d_dataset: _leaf_kepler,
    NodeKind.diffusion_pde_dataset: _leaf_pde,
    NodeKind.reaction_diffusion_dataset: _leaf_pde,
    NodeKind.advection_dataset: _leaf_pde,
    NodeKind.memorization_a_dataset: _leaf_memorization_a,
    NodeKind.memorization_b_dataset: _leaf_memorization_b,
    NodeKind.random_noise_dataset: _leaf_random_noise,
    NodeKind.linear_dataset: _leaf_linear_like,
    NodeKind.symbolic_func_dataset: _leaf_symbolic,
}


# --- dataset_mixer_b input drawers (private scoped map) ---
def _mib_uniform_motion(node, n, rng, *, eff_task):
    dd = node.data or {}
    x_np, _, _, _ = _build_uniform_linear_motion_arrays(rng, dd, dd, n, 0)
    return x_np


def _mib_kepler(node, n, rng, *, eff_task):
    dd = node.data or {}
    x_np, _, _, _ = _build_kepler_2d_arrays(rng, dd, dd, n, 0)
    return x_np


def _mib_pde(node, n, rng, *, eff_task):
    dd = node.data or {}
    x_np, _, _, _ = build_pde_field_arrays(node.type, rng, dd, dd, n, 0)
    return x_np

def _mib_memorization_b(node, n, rng, *, eff_task):
    dd = node.data or {}
    if eff_task == "cross_entropy_dense":
        vocab = _memorization_b_vocab_size(dd)
        out_dist = _scalar_str(dd.get("outputDistribution"), "additive_gaussian")
        alpha = _scalar_float(dd.get("alpha"), 1.0)
        x_np, _ = _memorization_b_xy_numpy(rng, n, vocab, out_dist, alpha)
        return x_np
    input_dim = _scalar_int(dd.get("inputDim"), 1)
    input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
    return sample_inputs(rng, n, input_dim, input_dist)


def _mib_memorization_a(node, n, rng, *, eff_task):
    dd = node.data or {}
    if eff_task == "cross_entropy_dense":
        input_dim = _scalar_int(dd.get("inputDim"), 1)
        input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
        return sample_inputs(rng, n, input_dim, input_dist)
    raise HTTPException(
        status_code=400,
        detail=(
            "memorization_a_dataset is only valid with cross_entropy_loss. "
            "Remove it from this dataset_mixer_b branch or switch the trainer to cross_entropy_loss."
        ),
    )


_MIXER_B_INPUT_DRAWERS: dict[NodeKind, Any] = {
    NodeKind.uniform_linear_motion_dataset: _mib_uniform_motion,
    NodeKind.kepler_2d_dataset: _mib_kepler,
    NodeKind.diffusion_pde_dataset: _mib_pde,
    NodeKind.reaction_diffusion_dataset: _mib_pde,
    NodeKind.advection_dataset: _mib_pde,
    NodeKind.memorization_b_dataset: _mib_memorization_b,
    NodeKind.memorization_a_dataset: _mib_memorization_a,
}


def _regression_dataset_draw_inputs(node, n, rng, trainer_task, edges, nmap, *, _draw_visited=None):
    """Draw ``x`` of shape ``(n, input_dim)`` for regression-style datasets (dataset_mixer_b sync driver = port A)."""
    dd: dict[str, Any] = node.data or {}
    eff_task = "mse_regression" if trainer_task == "diffusion_noise" else trainer_task
    if node.type == NodeKind.dataset_mixer:
        raise HTTPException(
            status_code=400,
            detail=(
                "dataset_mixer_b cannot use dataset_mixer (concat batch) on dataset_a or dataset_b: "
                "interpolated mixing needs one synchronized feature batch x per row."
            ),
        )
    if node.type == NodeKind.dataset_mixer_b:
        if _draw_visited is None:
            _draw_visited = set()
        if node.id in _draw_visited:
            raise HTTPException(status_code=400, detail="dataset_mixer_b graph contains a cycle.")
        _draw_visited.add(node.id)
        inner = _incoming_node(edges, nmap, node.id, "dataset_a")
        if inner is None:
            raise HTTPException(status_code=400, detail="dataset_mixer_b is missing dataset_a.")
        return _regression_dataset_draw_inputs(inner, n, rng, trainer_task, edges, nmap, _draw_visited=_draw_visited)
    handler = _MIXER_B_INPUT_DRAWERS.get(node.type)
    if handler is not None:
        return handler(node, n, rng, eff_task=eff_task)
    input_dim = _scalar_int(dd.get("inputDim"), 1)
    input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
    return sample_inputs(rng, n, input_dim, input_dist)


# --- dataset_mixer_b output samplers on synchronized inputs (private scoped map) ---
_MIXER_B_OUTPUT_ELSE_DETAIL = (
    "dataset_mixer_b sources must be linear_dataset, random_noise_dataset, memorization_a_dataset, "
    "memorization_b_dataset, symbolic_func_dataset, or dataset_mixer_b."
)


def _mob_memorization_b(node, x_shared, rng, *, trainer_task, out_dist, alpha, additive, sigma):
    dd = node.data or {}
    if trainer_task == "cross_entropy_dense":
        vocab = _memorization_b_vocab_size(dd)
        _, y_np = _memorization_b_xy_numpy(rng, int(x_shared.shape[0]), vocab, out_dist, alpha)
        return y_np
    raise HTTPException(status_code=400, detail=_MIXER_B_OUTPUT_ELSE_DETAIL)


def _mob_memorization_a(node, x_shared, rng, *, trainer_task, out_dist, alpha, additive, sigma):
    dd = node.data or {}
    output_dim = _scalar_int(dd.get("outputDim"), 1)
    return _sample_memorization_labels(rng, int(x_shared.shape[0]), output_dim, out_dist, alpha)


def _mob_random_noise(node, x_shared, rng, *, trainer_task, out_dist, alpha, additive, sigma):
    dd = node.data or {}
    n = int(x_shared.shape[0])
    output_dim = _scalar_int(dd.get("outputDim"), 1)
    input_dist = _scalar_str(dd.get("inputDistribution"), "standard_normal")
    y_np = sample_inputs(rng, n, output_dim, input_dist)
    if additive and sigma > 0:
        y_np = y_np + sigma * rng.standard_normal((y_np.shape[0], y_np.shape[1])).astype(np.float32)
    return y_np


def _mob_linear_like(node, x_shared, rng, *, trainer_task, out_dist, alpha, additive, sigma):
    dd = node.data or {}
    input_dim = int(x_shared.shape[1]) if x_shared.ndim >= 2 else _scalar_int(dd.get("inputDim"), 1)
    output_dim = _scalar_int(dd.get("outputDim"), 1)
    return _linear_like_targets_from_x(dd, x_shared, input_dim, output_dim, sigma=sigma, additive=additive, rng=rng)

def _mob_symbolic(node, x_shared, rng, *, trainer_task, out_dist, alpha, additive, sigma):
    dd = node.data or {}
    try:
        y_fn = build_y_numpy_fn(dd)
    except HTTPException as e:
        detail = str(getattr(e, "detail", "") or "")
        if "equationLatex is required" in detail:
            instance_title = _scalar_str(dd.get("instanceTitle"), "").strip() or node.id
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{detail} Triggered by symbolic_func_dataset '{instance_title}'. "
                    "If you are not intentionally using a symbolic dataset, inspect the dataset_mixer inputs and replace any "
                    "accidentally-added symbolic_func_dataset node."
                ),
            ) from e
        raise
    y_np = y_fn(x_shared)
    if additive and sigma > 0:
        y_np = y_np + sigma * rng.standard_normal((y_np.shape[0], y_np.shape[1])).astype(np.float32)
    return y_np


_MIXER_B_OUTPUT_SAMPLERS: dict[NodeKind, Any] = {
    NodeKind.memorization_b_dataset: _mob_memorization_b,
    NodeKind.memorization_a_dataset: _mob_memorization_a,
    NodeKind.random_noise_dataset: _mob_random_noise,
    NodeKind.linear_dataset: _mob_linear_like,
    NodeKind.symbolic_func_dataset: _mob_symbolic,
}


def _sample_dataset_b_outputs_for_synced_inputs(
    node, x_shared, edges, nmap, trainer_task, *, rng, leaf_sampler, _visited=None
):
    dd: dict[str, Any] = node.data or {}
    if node.type == NodeKind.dataset_mixer_b:
        x_mix, y_mix = _sample_interpolated_dataset_arrays(
            node,
            int(x_shared.shape[0]),
            edges,
            nmap,
            trainer_task,
            _visited=set(_visited or set()),
            rng=rng,
            x_override=x_shared,
            leaf_sampler=leaf_sampler,
        )
        if tuple(x_mix.shape) != tuple(x_shared.shape):
            raise HTTPException(status_code=400, detail="dataset_mixer_b failed to preserve shared input shape.")
        return y_mix
    if node.type == NodeKind.dataset_mixer:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer_b dataset_b input cannot be dataset_mixer A (cannot synchronize inputs).",
        )
    eff_task = "mse_regression" if trainer_task == "diffusion_noise" else trainer_task
    if node.type == NodeKind.memorization_a_dataset and eff_task != "cross_entropy_dense":
        raise HTTPException(
            status_code=400,
            detail=(
                "memorization_a_dataset is only valid with cross_entropy_loss. "
                "Remove it from dataset_mixer_b or switch the trainer to cross_entropy_loss."
            ),
        )
    out_dist = _scalar_str(dd.get("outputDistribution"), "additive_gaussian")
    alpha = _scalar_float(dd.get("alpha"), 1.0)
    noise_level = _scalar_float(dd.get("noiseLevel"), 0.25)
    additive = out_dist == "additive_gaussian"
    sigma = noise_level if additive else 0.0
    handler = _MIXER_B_OUTPUT_SAMPLERS.get(node.type)
    if handler is not None:
        return handler(node, x_shared, rng, trainer_task=trainer_task, out_dist=out_dist, alpha=alpha, additive=additive, sigma=sigma)
    raise HTTPException(status_code=400, detail=_MIXER_B_OUTPUT_ELSE_DETAIL)


# --- dataset_mixer_b interpolation with an injected leaf sampler ---
def _sample_interpolated_dataset_arrays(
    mixer_node, sample_count, edges, nmap, trainer_task, *, _visited=None, rng=None, x_override=None, leaf_sampler=None
):
    if _visited is None:
        _visited = set()
    if mixer_node.id in _visited:
        raise HTTPException(status_code=400, detail="dataset_mixer_b graph contains a cycle.")
    _visited.add(mixer_node.id)
    md: dict[str, Any] = mixer_node.data or {}
    total = int(sample_count)
    if total < 0:
        raise HTTPException(status_code=400, detail="dataset_mixer_b sample count must be >= 0.")
    lam = _scalar_float(md.get("interpolationLambda"), 0.5)
    if lam < 0.0 or lam > 1.0:
        raise HTTPException(status_code=400, detail="dataset_mixer_b interpolationLambda must be in [0, 1].")
    if rng is None:
        rng = np.random.default_rng(0)
    ds_a = _incoming_node(edges, nmap, mixer_node.id, "dataset_a")
    ds_b = _incoming_node(edges, nmap, mixer_node.id, "dataset_b")
    if ds_a is None or ds_b is None:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer_b requires both dataset_a and dataset_b connections.",
        )
    # Same tensors and RNG stream as wiring dataset A directly (no branch-B draws; avoids bogus test loss).
    if x_override is None and lam >= 1.0:
        return _sample_dataset_arrays_for_mixer(
            ds_a, sample_count, edges, nmap, trainer_task, _visited=set(_visited), rng=rng, leaf_sampler=leaf_sampler
        )
    if x_override is not None:
        x_shared = x_override
    else:
        x_shared = _regression_dataset_draw_inputs(ds_a, total, rng, trainer_task, edges, nmap)
    dd_b: dict[str, Any] = ds_b.data or {}
    want_in_b = _scalar_int(dd_b.get("inputDim"), int(x_shared.shape[1]) if x_shared.ndim >= 2 else 1)
    if x_shared.ndim >= 2 and int(x_shared.shape[1]) != want_in_b:
        raise HTTPException(
            status_code=400,
            detail=(
                f"dataset_mixer_b: synchronized x has input dim {int(x_shared.shape[1])} but dataset B "
                f"declares inputDim={want_in_b}."
            ),
        )
    if x_override is not None and tuple(x_shared.shape) != tuple(x_override.shape):
        raise HTTPException(status_code=400, detail="dataset_mixer_b x_override shape mismatch.")
    if lam >= 1.0:
        y_a = _sample_dataset_b_outputs_for_synced_inputs(
            ds_a, x_shared, edges, nmap, trainer_task, rng=rng, _visited=set(_visited), leaf_sampler=leaf_sampler
        )
        y_mix = y_a.astype(np.float32, copy=False)
    elif lam <= 0.0:
        y_b = _sample_dataset_b_outputs_for_synced_inputs(
            ds_b, x_shared, edges, nmap, trainer_task, rng=rng, _visited=set(_visited), leaf_sampler=leaf_sampler
        )
        y_mix = y_b.astype(np.float32, copy=False)
    else:
        y_a = _sample_dataset_b_outputs_for_synced_inputs(
            ds_a, x_shared, edges, nmap, trainer_task, rng=rng, _visited=set(_visited), leaf_sampler=leaf_sampler
        )
        y_b = _sample_dataset_b_outputs_for_synced_inputs(
            ds_b, x_shared, edges, nmap, trainer_task, rng=rng, _visited=set(_visited), leaf_sampler=leaf_sampler
        )
        if tuple(y_a.shape) != tuple(y_b.shape):
            raise HTTPException(
                status_code=400,
                detail=(
                    "dataset_mixer_b: dataset A and B outputs must have the same shape after synchronizing on x "
                    f"(got {tuple(y_a.shape)} vs {tuple(y_b.shape)})."
                ),
            )
        y_mix = lam * y_a.astype(np.float32, copy=False) + (1.0 - lam) * y_b.astype(np.float32, copy=False)
    return x_shared.astype(np.float32, copy=False), y_mix.astype(np.float32, copy=False)


# --- dataset_mixer concatenation with an injected leaf sampler ---
def _sample_mixed_dataset_arrays(
    mixer_node, sample_count, edges, nmap, trainer_task, *, _visited=None, rng=None, leaf_sampler=None
):
    if _visited is None:
        _visited = set()
    if mixer_node.id in _visited:
        raise HTTPException(status_code=400, detail="dataset_mixer graph contains a cycle.")
    _visited.add(mixer_node.id)
    md: dict[str, Any] = mixer_node.data or {}
    total = int(sample_count)
    if total < 0:
        raise HTTPException(status_code=400, detail="dataset_mixer sample count must be >= 0.")
    p_a = _scalar_float(md.get("proportionA"), 0.5)
    if p_a < 0.0 or p_a > 1.0:
        raise HTTPException(status_code=400, detail="dataset_mixer proportionA must be in [0, 1].")
    if rng is None:
        seed = _scalar_int(md.get("initSeed"), 0)
        rng = np.random.default_rng(seed)
    n_a = int(round(total * p_a))
    n_a = max(0, min(total, n_a))
    n_b = total - n_a

    ds_a = _incoming_node(edges, nmap, mixer_node.id, "dataset_a")
    ds_b = _incoming_node(edges, nmap, mixer_node.id, "dataset_b")
    if ds_a is None or ds_b is None:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer requires both dataset_a and dataset_b connections.",
        )

    def _sample_from(node, n):
        return _sample_dataset_arrays_for_mixer(
            node, n, edges, nmap, trainer_task, _visited=set(_visited), rng=rng, leaf_sampler=leaf_sampler
        )

    x_a, y_a = _sample_from(ds_a, n_a)
    x_b, y_b = _sample_from(ds_b, n_b)
    if x_a.ndim != x_b.ndim or y_a.ndim != y_b.ndim:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer inputs must have matching x/y rank before concatenation.",
        )
    if tuple(x_a.shape[1:]) != tuple(x_b.shape[1:]) or tuple(y_a.shape[1:]) != tuple(y_b.shape[1:]):
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer inputs must have matching input/output dimensions.",
        )
    x_mix = np.concatenate([x_a, x_b], axis=0) if total > 0 else np.zeros((0,), dtype=np.float32)
    y_mix = np.concatenate([y_a, y_b], axis=0) if total > 0 else np.zeros((0,), dtype=np.float32)
    if x_mix.shape[0] != y_mix.shape[0]:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer produced mismatched x/y sample counts.",
        )
    if x_mix.shape[0] > 1:
        perm = rng.permutation(x_mix.shape[0])
        x_mix = x_mix[perm]
        y_mix = y_mix[perm]
    return x_mix, y_mix


# --- Mixer router with an injected sampler for leaf datasets ---
def _sample_dataset_arrays_for_mixer(
    node, sample_count, edges, nmap, trainer_task, *, _visited=None, rng=None, leaf_sampler=None
):
    if node.type == NodeKind.dataset_mixer:
        return _sample_mixed_dataset_arrays(
            node, sample_count, edges, nmap, trainer_task, _visited=set(_visited or set()), rng=rng, leaf_sampler=leaf_sampler
        )
    if node.type == NodeKind.dataset_mixer_b:
        return _sample_interpolated_dataset_arrays(
            node, sample_count, edges, nmap, trainer_task, _visited=set(_visited or set()), rng=rng, leaf_sampler=leaf_sampler
        )
    if rng is None:
        rng = np.random.default_rng(0)
    assert leaf_sampler is not None, "leaf_sampler must be injected for the leaf path"
    return leaf_sampler(node, sample_count, rng, trainer_task)


# Token-arm streaming samplers use NumPy; tensor conversion stays in trainer_run.
def _st_memorization_b(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ds = _memorization_b_vocab_size(dd_train)
    alpha = _scalar_float(dd_train.get("alpha"), 1.0)
    out_dist_tr = _scalar_str(dd_train.get("outputDistribution"), "uniform_class_probs").strip()
    probs = _memorization_class_probs(vocab_ds, out_dist_tr, alpha)
    x_tok = rng.choice(vocab_ds, size=(sample_count,), p=probs).astype(np.int64)
    y_np = rng.choice(vocab_ds, size=(sample_count,), p=probs).astype(np.int64)
    return x_tok[:, None], y_np


def _st_unigram(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 4)
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    alpha = _scalar_float(dd_train.get("alpha"), 1.0)
    out_dist_tr = _scalar_str(dd_train.get("outputDistribution"), "power_law_class_probs").strip()
    probs = _unigram_token_probs(vocab_ds, out_dist_tr, alpha)
    tokens_np = rng.choice(vocab_ds, size=(sample_count, ctx_len_ds), p=probs).astype(np.int64)
    y_np = rng.choice(vocab_ds, size=(sample_count,), p=probs).astype(np.int64)
    return tokens_np, y_np


def _st_token_prediction(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 4)
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 4)
    retrieval_mode = _scalar_str(dd_train.get("retrievalMode"), "position").strip().lower()
    which_token = _scalar_int(dd_train.get("whichToken"), -1)
    tokens_np = rng.integers(0, vocab_ds, size=(sample_count, ctx_len_ds), dtype=np.int64)
    if retrieval_mode == "content":
        current = tokens_np[:, -1][:, None]
        previous = tokens_np[:, :-1]
        distances = np.abs(previous - current)
        nearest_idx = np.argmin(distances, axis=1)
        y_np = previous[np.arange(previous.shape[0]), nearest_idx].copy()
    else:
        y_np = tokens_np[:, which_token].copy()
    return tokens_np, y_np


def _st_bigram(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    x_np, y_np, _, _, _ = _build_bigram_low_rank_arrays(dd_train, dd_test_eff, sample_count, 0, train_pair_rng=rng)
    return x_np, y_np


def _st_circle_random_walk(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 4)
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    p_right = _scalar_float(dd_train.get("rightStepProb"), 0.5)

    def _sample_circle_walk(n, g):
        if n <= 0:
            return np.zeros((0, ctx_len_ds), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        seq = np.empty((n, ctx_len_ds + 1), dtype=np.int64)
        seq[:, 0] = g.integers(0, vocab_ds, size=(n,), dtype=np.int64)
        steps = np.where(g.random((n, ctx_len_ds)) < p_right, 1, -1).astype(np.int64)
        for t in range(ctx_len_ds):
            seq[:, t + 1] = (seq[:, t] + steps[:, t]) % vocab_ds
        return seq[:, :ctx_len_ds], seq[:, ctx_len_ds]

    return _sample_circle_walk(sample_count, rng)


def _st_circular_motion(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    x_np, y_np, _, _, _, _ = _build_circular_motion_arrays(dd_train, dd_test_eff, sample_count, 0, train_sample_rng=rng)
    return x_np, y_np


def _st_associative_recall(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ic = _scalar_int(dd_train.get("vocabSize"), 64)
    num_pairs = _scalar_int(dd_train.get("numPairs"), 32)
    in_context_repeat = _scalar_int(dd_train.get("inContextRepeat"), 1)
    cross_repeat_prob = _scalar_float(dd_train.get("crossSampleRepeatProb"), 0.0)
    repeated_token_count = _scalar_int(dd_train.get("repeatedTokenCount"), 2)
    ctx_len_ds = 2 * num_pairs + 1
    repeated_pool = np.arange(repeated_token_count, dtype=np.int64)

    def _sample_associative_batch(n, g):
        if n <= 0:
            return np.zeros((0, ctx_len_ds), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        x = np.empty((n, ctx_len_ds), dtype=np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            keys = g.permutation(vocab_ic)[:num_pairs].astype(np.int64)
            vals = g.permutation(vocab_ic)[:num_pairs].astype(np.int64)
            query_slot = int(g.integers(0, num_pairs))
            query = int(keys[query_slot])
            if cross_repeat_prob > 0 and repeated_pool.size > 0 and float(g.random()) < cross_repeat_prob:
                query = int(repeated_pool[int(g.integers(0, repeated_pool.size))])
                hit = np.where(keys == query)[0]
                if hit.size == 0:
                    query_slot = int(g.integers(0, num_pairs))
                    keys[query_slot] = query
                else:
                    query_slot = int(hit[0])
            if in_context_repeat > 1:
                candidates = np.arange(num_pairs, dtype=np.int64)
                candidates = candidates[candidates != query_slot]
                if candidates.size > 0:
                    g.shuffle(candidates)
                    extra = min(in_context_repeat - 1, int(candidates.size))
                    keys[candidates[:extra]] = query
            seq = np.empty((ctx_len_ds,), dtype=np.int64)
            seq[0::2][:num_pairs] = keys
            seq[1::2][:num_pairs] = vals
            seq[-1] = query
            hit_after = np.where(keys == query)[0]
            if hit_after.size == 0:
                target = int(vals[0])
            else:
                target = int(vals[int(hit_after[0])])
            x[i] = seq
            y[i] = target
        return x, y

    return _sample_associative_batch(sample_count, rng)


def _st_modular_addition(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_train = node.data or {}
    vocab_ds_m = _scalar_int(dd_train.get("modulus"), 59)
    train_fraction = _scalar_float(dd_train.get("trainFraction"), 0.3)
    a, b = np.meshgrid(
        np.arange(vocab_ds_m, dtype=np.int64), np.arange(vocab_ds_m, dtype=np.int64), indexing="ij"
    )
    x_all = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
    y_all = ((x_all[:, 0] + x_all[:, 1]) % vocab_ds_m).astype(np.int64)
    perm = rng.permutation(x_all.shape[0])
    x_all = x_all[perm]
    y_all = y_all[perm]
    n_train = int(round(train_fraction * x_all.shape[0]))
    n_train = min(max(n_train, 1), x_all.shape[0])
    if sample_size_override is not None:
        n_want = int(sample_size_override)
        n_remain = int(x_all.shape[0]) - n_train
        n_take = min(max(n_want, 0), n_remain)
        if n_take <= 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        return x_all[n_train : n_train + n_take].copy(), y_all[n_train : n_train + n_take].copy()
    return x_all[:n_train].copy(), y_all[:n_train].copy()


def _st_toy_language(node, sample_count, rng, *, dd_test_eff, stream_scalar_seed, sample_size_override=None):
    dd_eff = dict(node.data or {})
    dd_eff["seed"] = [stream_scalar_seed]
    dd_eff["initSeed"] = [stream_scalar_seed]
    x_np, y_np, _, _ = _materialize_toy_language_lm_numpy(node.type, dd_eff, sample_count, 0)
    return x_np, y_np


def sample_default_circle_walk_tokens(node, sample_count, rng):
    dd_train = node.data or {}
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 4)
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    p_right = _scalar_float(dd_train.get("rightStepProb"), 0.5)
    seq = np.empty((sample_count, ctx_len_ds + 1), dtype=np.int64)
    seq[:, 0] = rng.integers(0, vocab_ds, size=(sample_count,), dtype=np.int64)
    steps = np.where(rng.random((sample_count, ctx_len_ds)) < p_right, 1, -1).astype(np.int64)
    for t in range(ctx_len_ds):
        seq[:, t + 1] = (seq[:, t] + steps[:, t]) % vocab_ds
    return seq[:, :ctx_len_ds].copy(), seq[:, ctx_len_ds].copy()


STREAMING_TOKEN_SAMPLE_HANDLERS: dict[NodeKind, Any] = {
    NodeKind.memorization_b_dataset: _st_memorization_b,
    NodeKind.unigram_dataset: _st_unigram,
    NodeKind.token_prediction_dataset: _st_token_prediction,
    NodeKind.bigram_low_rank_dataset: _st_bigram,
    NodeKind.circle_random_walk_dataset: _st_circle_random_walk,
    NodeKind.circular_motion_dataset: _st_circular_motion,
    NodeKind.in_context_associative_recall_dataset: _st_associative_recall,
    NodeKind.modular_addition_dataset: _st_modular_addition,
    **{kind: _st_toy_language for kind in TOY_LM_BUILDERS},
}
