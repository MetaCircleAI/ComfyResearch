"""Per-step full-tensor redraw for streaming trainers.

Trainer-side glue layer of the dataset 3-layer boundary:
  dataset_runtime (numpy, no torch)  ->  THIS (seed*torch*device*vision*teacher)  ->  trainer_run (assembly)

Must never import trainer_run. The three trainer_run-private helpers it needs
(leaf sampler, teacher input sampler, linear-dim resolver) are injected as callbacks.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.schemas.graph import Edge, Node, NodeKind
from comfy_research.engine.losses.loss_builders import TrainerTask
from comfy_research.engine.datasets.streaming_seed import streaming_train_step_seed
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.datasets.teacher_dataset_runtime import teacher_labels_numpy
from comfy_research.engine.datasets.pde_field_dataset_runtime import build_pde_field_arrays
from comfy_research.engine.datasets.vision_datasets_runtime import (
    build_vision_numpy_arrays,
    _vision_flatten_feature_matrix,
)
from comfy_research.engine.datasets.synthetic_dataset_builders import (
    _build_kepler_2d_arrays,
    _build_uniform_linear_motion_arrays,
    _linear_like_targets_from_x,
    _memorization_b_vocab_size,
    _memorization_b_xy_numpy,
    _sample_memorization_labels,
)
from comfy_research.engine.datasets.dataset_runtime import (
    STREAMING_TOKEN_SAMPLE_HANDLERS,
    sample_default_circle_walk_tokens,
    _declared_dataset_split_sizes,
    _sample_dataset_arrays_for_mixer,
)

# Injection contract for the three private trainer helpers used below.
# Protocols (not bare Callable[..., ...]) so a drift in any callback's signature
# is caught at the type layer, matching the boundary contract the spec promised.
class LeafSampler(Protocol):
    def __call__(
        self,
        node: Node,
        sample_count: int,
        rng: np.random.Generator,
        trainer_task: Literal[
            "mse_regression", "cross_entropy_dense", "cross_entropy_tokens", "diffusion_noise"
        ],
    ) -> tuple[np.ndarray, np.ndarray]: ...


class TeacherInputSampler(Protocol):
    def __call__(
        self,
        edges: list[Edge],
        nmap: dict[str, Node],
        teacher_ds: Node,
        handle: str,
        fallback_sampler: Node | None,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, Node]: ...


class ResolveMlpDims(Protocol):
    def __call__(
        self,
        dd_train: dict[str, Any],
        md: dict[str, Any],
        ds_type: NodeKind | None = None,
        *,
        model_kind: NodeKind | None = None,
        memorization_a_ce_slot_groups: int = 1,
    ) -> tuple[int, int]: ...

# Derived constants -- recomputed from node capabilities, identical to trainer_run's,
# so they cannot drift. NOT hand-written kind lists.
_DATASET_MIXER_TYPES = frozenset(NodeKind(t) for t in node_types_with_capability("dataset_mixer"))
_VISION_DATASET_TYPES = frozenset(NodeKind(t) for t in node_types_with_capability("vision_dataset"))
_LINEAR_LIKE_DATASET_TYPES = frozenset(NodeKind(t) for t in node_types_with_capability("linear_like_dataset"))
_PDE_FIELD_DATASET_TYPES = frozenset(NodeKind(t) for t in node_types_with_capability("pde_field_dataset"))


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


def rematerialize_train_tensors_for_step(
    *,
    step: int,
    dataset_base_seed: int,
    trainer_task: TrainerTask,
    ds_train: Node,
    edges: list[Edge],
    nmap: dict[str, Node],
    model_node: Node,
    ds_test_raw: Node | None,
    legacy_optional_test_wire: bool,
    teacher_module: nn.Module | None,
    device: torch.device,
    sample_size_override: int | None = None,
    memorization_a_ce_slot_groups: int = 1,
    leaf_sampler: LeafSampler,
    teacher_input_sampler: TeacherInputSampler,
    resolve_mlp_dims: ResolveMlpDims,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a fresh full training tensor pair for this global train step (streaming mode only). Returns on-device tensors."""
    ss = int(streaming_train_step_seed(dataset_base_seed, step))
    rng = np.random.default_rng(ss)
    dd_train = ds_train.data or {}
    if sample_size_override is not None:
        train_size = int(sample_size_override)
    elif ds_train.type == NodeKind.dataset_mixer:
        raw_tr = dd_train.get("trainTotalSamples")
        train_size = _scalar_int(raw_tr, 800) if raw_tr is not None else _scalar_int(dd_train.get("totalSamples"), 800)
    elif ds_train.type == NodeKind.dataset_mixer_b:
        train_size, _ = _declared_dataset_split_sizes(ds_train, edges, nmap)
    elif ds_train.type == NodeKind.modular_addition_dataset:
        train_size, _ = _declared_dataset_split_sizes(ds_train, edges, nmap)
    else:
        train_size = _scalar_int(dd_train.get("trainSize"), 800)
    if train_size < 1:
        raise HTTPException(status_code=500, detail="Internal: train size invalid during streaming rematerialize.")

    dd_test_eff: dict[str, Any]
    if legacy_optional_test_wire and ds_test_raw is None:
        dd_test_eff = dd_train
    elif ds_test_raw is not None:
        dd_test_eff = ds_test_raw.data or {}
    else:
        dd_test_eff = dd_train

    md: dict[str, Any] = model_node.data or {}

    if trainer_task in ("mse_regression", "cross_entropy_dense", "diffusion_noise", "vision_classification"):
        if trainer_task == "vision_classification":
            x_np, y_np, _, _ = build_vision_numpy_arrays(ds_train.type, dd_train, train_size, 0, rng)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).long().to(device)
        if ds_train.type == NodeKind.teacher_dataset:
            if teacher_module is None:
                raise HTTPException(status_code=500, detail="Internal: teacher module missing for streaming rematerialize.")
            x_np, _ = teacher_input_sampler(
                edges, nmap, ds_train, "train_input", None, rng=rng
            )
            y_np = teacher_labels_numpy(teacher_module, x_np)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)

        input_dist = _scalar_str(dd_train.get("inputDistribution"), "standard_normal")
        out_dist = _scalar_str(dd_train.get("outputDistribution"), "additive_gaussian")
        mem_alpha = _scalar_float(dd_train.get("alpha"), 1.0)
        noise_level = _scalar_float(dd_train.get("noiseLevel"), 0.25)
        additive = out_dist == "additive_gaussian"
        sigma = noise_level if additive else 0.0

        if ds_train.type in _DATASET_MIXER_TYPES:
            x_np, y_np = _sample_dataset_arrays_for_mixer(ds_train, train_size, edges, nmap, trainer_task, rng=rng, leaf_sampler=leaf_sampler)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type == NodeKind.uniform_linear_motion_dataset:
            x_np, y_np, _, _ = _build_uniform_linear_motion_arrays(
                rng, dd_train, dd_test_eff, train_size, 0
            )
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type == NodeKind.kepler_2d_dataset:
            x_np, y_np, _, _ = _build_kepler_2d_arrays(rng, dd_train, dd_test_eff, train_size, 0)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type in _PDE_FIELD_DATASET_TYPES:
            x_np, y_np, _, _ = build_pde_field_arrays(
                ds_train.type, rng, dd_train, dd_test_eff, train_size, 0
            )
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)

        input_dim, output_dim = resolve_mlp_dims(
            dd_train,
            md,
            ds_train.type,
            model_kind=model_node.type,
            memorization_a_ce_slot_groups=memorization_a_ce_slot_groups,
        )
        if ds_train.type == NodeKind.memorization_b_dataset and trainer_task == "cross_entropy_dense":
            vocab = _memorization_b_vocab_size(dd_train)
            x_np, y_np = _memorization_b_xy_numpy(
                rng, train_size, vocab, out_dist, mem_alpha
            )
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type == NodeKind.memorization_a_dataset and trainer_task == "cross_entropy_dense":
            x_np = sample_inputs(rng, train_size, input_dim, input_dist)
            y_np = _sample_memorization_labels(rng, train_size, output_dim, out_dist, mem_alpha)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type in _VISION_DATASET_TYPES and trainer_task == "cross_entropy_dense":
            x4, y_np, _, _ = build_vision_numpy_arrays(ds_train.type, dd_train, train_size, 0, rng)
            x_np = _vision_flatten_feature_matrix(x4)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).long().to(device)
        if ds_train.type in _LINEAR_LIKE_DATASET_TYPES:
            x_np = sample_inputs(rng, train_size, input_dim, input_dist)
            y_np = _linear_like_targets_from_x(
                dd_train, x_np, input_dim, output_dim, sigma=sigma, additive=additive, rng=rng
            )
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        if ds_train.type == NodeKind.symbolic_func_dataset:
            x_np = sample_inputs(rng, train_size, input_dim, input_dist)
            y_fn = build_y_numpy_fn(dd_train)
            y_np = y_fn(x_np)
            if additive and sigma > 0:
                y_np = y_np + sigma * rng.standard_normal((y_np.shape[0], y_np.shape[1])).astype(np.float32)
            return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Streaming rematerialize does not support dataset type {ds_train.type} for this trainer task; "
                "use linear_dataset, random_noise_dataset, memorization_b_dataset, memorization_a_dataset with "
                "cross_entropy_loss only, dataset_mixer, dataset_mixer_b, uniform_linear_motion_dataset, kepler_2d_dataset, "
                "diffusion_pde_dataset, reaction_diffusion_dataset, advection_dataset, teacher_dataset, or "
                "symbolic_func_dataset."
            ),
        )

    # token_classification
    handler = STREAMING_TOKEN_SAMPLE_HANDLERS.get(ds_train.type)
    if handler is not None:
        x_np, y_np = handler(ds_train, train_size, rng, dd_test_eff=dd_test_eff, stream_scalar_seed=ss, sample_size_override=sample_size_override)
    else:
        x_np, y_np = sample_default_circle_walk_tokens(ds_train, train_size, rng)
    return torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
