"""Dataset dims/sampling-mode/minibatch helpers and dataset-family kind-sets (extracted from trainer_run)."""

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import HTTPException

from comfy_research.engine.datasets.dataset_runtime import LEAF_SAMPLE_HANDLERS
from comfy_research.engine.datasets.sampling_orders import (
    seedsequence_epoch_permutation,
)
from comfy_research.engine.models.hyena_like_numeric_model import read_numeric_hyena_layout_from_md
from comfy_research.engine.models.numeric_transformer_model import read_numeric_transformer_layout_from_md
from comfy_research.engine.datasets.pde_field_dataset_runtime import pde_field_flat_dims
from comfy_research.engine.datasets.streaming_seed import streaming_train_step_seed
from comfy_research.engine.datasets.synthetic_dataset_builders import _memorization_b_vocab_size
from comfy_research.engine.trainer.scalar import _scalar_int
from comfy_research.engine.datasets.vision_datasets_runtime import (
    build_vision_numpy_arrays,
    vision_num_classes,
    _vision_flatten_feature_matrix,
)
from comfy_research.engine.models.vision_models import infer_vision_input_channels_height_width
from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.schemas.graph import Node, NodeKind

_PDE_FIELD_DATASET_TYPES = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("pde_field_dataset"))


_VISION_DATASET_TYPES = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("vision_dataset"))


def _vision_flatten_enabled(dd: dict[str, Any]) -> bool:
    v = dd.get("flattenOutput", dd.get("flatten"))
    if isinstance(v, list):
        v = v[0] if v else False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return False


_TOY_LANGUAGE_TOKEN_DATASETS = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("toy_language_token_dataset")
)


_TOKEN_CLASSIFICATION_DATASET_TYPES = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("token_classification_dataset")
)


_VECTOR_REGRESSION_DATASET_TYPES = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("vector_regression_dataset")
)


_DIFFUSION_NOISE_DATASET_TYPES = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("diffusion_noise_dataset")
)


_DATASET_MIXER_TYPES = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("dataset_mixer"))



_TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES = frozenset(
    NodeKind(node_type) for node_type in node_types_with_capability("text_heavy_toy_language_dataset")
)


_LINEAR_LIKE_DATASET_TYPES = frozenset(NodeKind(node_type) for node_type in node_types_with_capability("linear_like_dataset"))


def _resolve_linear_dataset_mlp_dims(
    dd_train: dict[str, Any],
    md: dict[str, Any],
    ds_type: NodeKind | None = None,
    *,
    model_kind: NodeKind | None = None,
    binary_single_logit: bool = False,
    memorization_a_ce_slot_groups: int = 1,
) -> tuple[int, int]:
    m_in = _scalar_int(md.get("inputDim"), 1)
    m_out = _scalar_int(md.get("outputDim"), 1)
    if binary_single_logit and ds_type == NodeKind.memorization_b_dataset:
        vocab = _memorization_b_vocab_size(dd_train)
        if vocab != 2 or m_in != vocab or m_out != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "binary_cross_entropy_with_logits_loss requires a binary memorization_b_dataset "
                    f"(vocabSize=2), model inputDim=2, and model outputDim=1; got "
                    f"vocabSize={vocab}, inputDim={m_in}, outputDim={m_out}."
                ),
            )
        return vocab, vocab
    if binary_single_logit and ds_type in {
        NodeKind.information_bottleneck_dataset,
        NodeKind.memorization_a_dataset,
    }:
        d_in = _scalar_int(dd_train.get("inputDim"), m_in)
        d_out = _scalar_int(dd_train.get("outputDim"), 2)
        if d_out != 2 or m_in != d_in or m_out != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "binary_cross_entropy_with_logits_loss requires a binary dataset "
                    f"(outputDim=2), matching model inputDim={d_in}, and model outputDim=1; got "
                    f"dataset outputDim={d_out}, model inputDim={m_in}, model outputDim={m_out}."
                ),
            )
        return d_in, d_out
    if ds_type == NodeKind.memorization_b_dataset:
        vocab = _memorization_b_vocab_size(dd_train)
        if m_in != vocab or m_out != vocab:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"memorization_b_dataset uses shared vocabSize={vocab}; "
                    f"model dimensions must be ({vocab}, {vocab}) but got ({m_in}, {m_out})."
                ),
            )
        return vocab, vocab
    if ds_type in _PDE_FIELD_DATASET_TYPES:
        t_ds, c_ds, g_ds, flat = pde_field_flat_dims(dd_train)
        if model_kind in (NodeKind.mpp_spatiotemporal_model, NodeKind.afno_lite_spatiotemporal_model):
            tm = max(1, _scalar_int(md.get("contextFrames"), t_ds))
            cm = max(1, _scalar_int(md.get("channels"), c_ds))
            gm = max(4, _scalar_int(md.get("gridSize"), g_ds))
            if (tm, cm, gm) != (t_ds, c_ds, g_ds):
                model_label = (
                    "mpp_spatiotemporal_model"
                    if model_kind == NodeKind.mpp_spatiotemporal_model
                    else "afno_lite_spatiotemporal_model"
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{model_label} contextFrames, channels, and gridSize must match the "
                        f"PDE dataset ({t_ds}, {c_ds}, {g_ds}); got ({tm}, {cm}, {gm})."
                    ),
                )
        if m_in != flat or m_out != flat:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"PDE field dataset flat size is {flat} (= contextFrames*channels*gridSize^2); "
                    f"model inputDim and outputDim must both equal {flat} (got {m_in}, {m_out})."
                ),
            )
        return flat, flat
    if ds_type in _VISION_DATASET_TYPES and _vision_flatten_enabled(dd_train):
        ch, h, w = infer_vision_input_channels_height_width(ds_type, dd_train)
        flat = int(ch) * int(h) * int(w)
        n_cls = vision_num_classes(ds_type, dd_train)
        if m_in != flat or m_out != n_cls:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Flattened vision: MLP inputDim must be C·H·W = {flat} (from CHW=({ch},{h},{w})), "
                    f"outputDim must be num_classes = {n_cls}; got inputDim={m_in}, outputDim={m_out}."
                ),
            )
        return flat, n_cls
    if ds_type == NodeKind.uniform_linear_motion_dataset:
        pos_dim = _scalar_int(dd_train.get("positionDim"), 1)
        ctx = max(1, _scalar_int(dd_train.get("contextLength"), 2))
        if ctx != 2:
            raise HTTPException(
                status_code=400,
                detail="Uniform linear motion dataset requires contextLength=2 for the built-in sampler.",
            )
        flat_in = ctx * pos_dim
        flat_out = ctx * pos_dim
        if model_kind == NodeKind.numeric_transformer_model:
            ctx_m, tok_in, tok_out = read_numeric_transformer_layout_from_md(md)
            if ctx_m != ctx:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Numeric transformer contextLength ({ctx_m}) must match the uniform motion dataset "
                        f"contextLength ({ctx})."
                    ),
                )
            if tok_in != pos_dim or tok_out != pos_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Uniform motion positionDim ({pos_dim}) must match numeric transformer "
                        f"inputDim and outputDim (per-position dimension); model has token dims ({tok_in}, {tok_out})."
                    ),
                )
            return flat_in, flat_out
        if model_kind == NodeKind.numeric_hyena_model:
            ctx_m, tok_in, tok_out = read_numeric_hyena_layout_from_md(md)
            if ctx_m != ctx:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Numeric Hyena contextLength ({ctx_m}) must match the uniform motion dataset "
                        f"contextLength ({ctx})."
                    ),
                )
            if tok_in != pos_dim or tok_out != pos_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Uniform motion positionDim ({pos_dim}) must match numeric Hyena "
                        f"inputDim and outputDim (per-position dimension); model has token dims ({tok_in}, {tok_out})."
                    ),
                )
            return flat_in, flat_out
        if flat_in != m_in or flat_out != m_out:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uniform linear motion flat I/O is ({flat_in}, {flat_out}) = contextLength * positionDim "
                    f"each way; model must use inputDim={flat_in} and outputDim={flat_out} for flat MLP-style models."
                ),
            )
            return flat_in, flat_out
    if ds_type == NodeKind.kepler_2d_dataset:
        ctx = max(1, _scalar_int(dd_train.get("contextLength"), 8))
        tok_dim = 2
        flat_in = ctx * tok_dim
        flat_out = ctx * tok_dim
        if model_kind == NodeKind.numeric_transformer_model:
            ctx_m, tok_in, tok_out = read_numeric_transformer_layout_from_md(md)
            if ctx_m != ctx:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Numeric transformer contextLength ({ctx_m}) must match kepler_2d_dataset contextLength ({ctx})."
                    ),
                )
            if tok_in != tok_dim or tok_out != tok_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "kepler_2d_dataset emits 2D position tokens; numeric transformer "
                        f"inputDim/outputDim must both be 2 (got {tok_in}, {tok_out})."
                    ),
                )
            return flat_in, flat_out
        if model_kind == NodeKind.numeric_hyena_model:
            ctx_m, tok_in, tok_out = read_numeric_hyena_layout_from_md(md)
            if ctx_m != ctx:
                raise HTTPException(
                    status_code=400,
                    detail=f"Numeric Hyena contextLength ({ctx_m}) must match kepler_2d_dataset contextLength ({ctx}).",
                )
            if tok_in != tok_dim or tok_out != tok_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "kepler_2d_dataset emits 2D position tokens; numeric Hyena "
                        f"inputDim/outputDim must both be 2 (got {tok_in}, {tok_out})."
                    ),
                )
            return flat_in, flat_out
        if flat_in != m_in or flat_out != m_out:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"kepler_2d_dataset flat I/O is ({flat_in}, {flat_out}) = contextLength * 2 each way; "
                    f"model must use inputDim={flat_in} and outputDim={flat_out} for flat MLP-style models."
                ),
            )
        return flat_in, flat_out
    if ds_type == NodeKind.memorization_a_dataset and memorization_a_ce_slot_groups > 1:
        t_slots = int(memorization_a_ce_slot_groups)
        if t_slots < 2:
            raise HTTPException(status_code=500, detail="Internal: memorization_a_ce_slot_groups invalid.")
        default_v = max(1, m_out // t_slots) if m_out >= t_slots else 1
        d_in = _scalar_int(dd_train.get("inputDim"), m_in)
        d_out = _scalar_int(dd_train.get("outputDim"), default_v)
        if d_in != m_in:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Linear dataset dimensions ({d_in}, *) must match "
                    f"the MLP model dimensions ({m_in}, *)."
                ),
            )
        if m_out != d_out * t_slots:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"memorization_a_dataset with cross-entropy slot masking (T={t_slots}) requires "
                    f"model outputDim = dataset outputDim * T ({d_out * t_slots}); "
                    f"got model outputDim={m_out} and dataset outputDim={d_out}."
                ),
            )
        if d_out < 2:
            raise HTTPException(
                status_code=400,
                detail="memorization_a_dataset outputDim (per-slot vocab V) must be >= 2 for cross-entropy.",
            )
        return d_in, d_out
    d_in = _scalar_int(dd_train.get("inputDim"), m_in)
    d_out = _scalar_int(dd_train.get("outputDim"), m_out)
    if d_in != m_in or d_out != m_out:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Linear dataset dimensions ({d_in}, {d_out}) must match "
                f"the MLP model dimensions ({m_in}, {m_out})."
            ),
        )
    return d_in, d_out


def _sample_base_vector_dataset_arrays(
    node: Node,
    sample_count: int,
    rng: np.random.Generator,
    trainer_task: Literal[
        "mse_regression", "cross_entropy_dense", "cross_entropy_tokens", "diffusion_noise"
    ],
) -> tuple[np.ndarray, np.ndarray]:
    dd: dict[str, Any] = node.data or {}
    eff_task: Literal["mse_regression", "cross_entropy_dense", "cross_entropy_tokens"] = (
        "mse_regression" if trainer_task == "diffusion_noise" else trainer_task
    )
    if sample_count < 0:
        raise HTTPException(status_code=400, detail="Dataset sample count must be >= 0.")
    if node.type in _VISION_DATASET_TYPES and eff_task == "cross_entropy_dense":
        if not _vision_flatten_enabled(dd):
            raise HTTPException(
                status_code=400,
                detail="Vision dataset sampling requires 'flatten to vector' for cross_entropy_dense.",
            )
        x4, y_np, _, _ = build_vision_numpy_arrays(node.type, dd, sample_count, 0, rng)
        return _vision_flatten_feature_matrix(x4), y_np
    handler = LEAF_SAMPLE_HANDLERS.get(node.type)
    if handler is not None:
        return handler(node, sample_count, rng, trainer_task=trainer_task, eff_task=eff_task)
    raise HTTPException(
        status_code=400,
        detail=(
            "dataset_mixer sources must be linear_dataset, random_noise_dataset, memorization_b_dataset, "
            "symbolic_func_dataset, uniform_linear_motion_dataset, kepler_2d_dataset, diffusion_pde_dataset, reaction_diffusion_dataset, "
            "advection_dataset, dataset_mixer, or dataset_mixer_b. "
            "memorization_a_dataset is only allowed when the trainer uses cross_entropy_loss (not MSE)."
        ),
    )


def _dataset_sampling_mode_raw(dd: dict[str, Any]) -> Literal["fixed", "streaming"]:
    raw = dd.get("samplingMode")
    if isinstance(raw, list):
        raw = raw[0] if raw else "fixed"
    s = str(raw or "fixed").strip().lower()
    if s == "streaming":
        return "streaming"
    return "fixed"


def _trainer_train_dataset_streaming(ds_train: Node, dd_train: dict[str, Any]) -> bool:
    return _dataset_sampling_mode_raw(dd_train) == "streaming"


def _take_train_minibatch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    step: int,
    perm_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    if batch_size < 0 or n <= 0 or batch_size >= n:
        return x, y
    dev = x.device
    g = torch.Generator(device=dev)
    g.manual_seed(int(streaming_train_step_seed(perm_seed, step)))
    perm = torch.randperm(n, generator=g, device=dev)
    idx = perm[: int(batch_size)]
    return x.index_select(0, idx), y.index_select(0, idx)


def _take_epoch_shuffled_minibatch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    epoch: int,
    step_in_epoch: int,
    run_seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact NumPy SeedSequence permutation per data epoch."""
    n = int(x.shape[0])
    if batch_size < 0 or batch_size >= n:
        return x, y
    permutation = torch.as_tensor(
        seedsequence_epoch_permutation(n, seed=run_seed, epoch=epoch),
        dtype=torch.long,
        device=x.device,
    )
    start = int(step_in_epoch) * int(batch_size)
    idx = permutation[start : min(start + int(batch_size), n)]
    return x.index_select(0, idx), y.index_select(0, idx)


def _cifar10_crop_flip_standardize(
    x: torch.Tensor,
    pixel_mean: torch.Tensor,
    global_std: float,
    *,
    generator: torch.Generator,
    padding: int = 4,
) -> torch.Tensor:
    """Random edge crop + horizontal flip, then the paper's standardization."""
    batch, _, height, width = x.shape
    padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    offsets = torch.randint(
        0,
        2 * padding + 1,
        (batch, 2),
        generator=generator,
        device=x.device,
    )
    patches = padded.unfold(2, height, 1).unfold(3, width, 1)
    rows = torch.arange(batch, device=x.device)
    cropped = patches[rows, :, offsets[:, 0], offsets[:, 1]].clone()
    flip = torch.rand(batch, generator=generator, device=x.device) < 0.5
    cropped[flip] = torch.flip(cropped[flip], dims=(3,))
    return (cropped - pixel_mean) / float(global_std)
