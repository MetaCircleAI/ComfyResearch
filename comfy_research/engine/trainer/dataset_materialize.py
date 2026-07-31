"""Unified  dataset materialization.

materialize_dataset_for_training(req) resolves the four-way build dispatch
(vision / atomic chain / vector model / token) through the branch-level
DATASET_PROVIDERS registry; providers receive a narrow
DatasetMaterializeRequest, never PrepareState) and each branch's family
dispatch EXACTLY — same conditions, same order, same shared-rng draw
sequence — returning a DatasetArrays and never writing state. The
per-family helpers moved here from the four prepare_build_* modules
sectioning); where the atomic/vector copies were byte-identical one
copy remains, and every genuine divergence is kept behind an explicit
``branch`` parameter, locked by tests/test_dataset_materialize_dispatch.py
(KNOWN_DIVERGENCES) and the RNG-sequence golden.

DatasetArrays conventions per branch:
- vision: input_dim=channels, output_dim=num_classes, extras={"image_size": h}
- atomic/vector: input_dim/output_dim = dataset dims; teacher carries
  extras={"teacher": module} (becomes streaming_teacher)
- token: input_dim=ctx_len_ds (token width), output_dim=vocab_ds (class count)
"""
from typing import Any, Literal

import numpy as np
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import collect_atomic_layer_chain_front_to_back
from comfy_research.engine.datasets.dataset_runtime import (
    _materialize_toy_language_lm_numpy,
    _sample_dataset_arrays_for_mixer,
)
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs
from comfy_research.engine.datasets.symbolic_dataset_compile import build_y_numpy_fn
from comfy_research.engine.datasets.synthetic_dataset_builders import (
    _apply_linear_map,
    _build_bigram_low_rank_arrays,
    _build_circular_motion_arrays,
    _build_kepler_2d_arrays,
    _build_uniform_linear_motion_arrays,
    _linear_like_targets_from_x,
    _linear_regression_weight_rng,
    _memorization_b_vocab_size,
    _memorization_b_xy_numpy,
    _memorization_class_probs,
    _random_linear_weights,
    _sample_memorization_labels,
    _unigram_token_probs,
)
from comfy_research.engine.datasets.teacher_dataset_runtime import teacher_labels_numpy
from comfy_research.engine.trainer.dataset_arrays import DatasetArrays
from comfy_research.engine.trainer.dataset_helpers import (
    _DATASET_MIXER_TYPES,
    _TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES,
    _TOY_LANGUAGE_TOKEN_DATASETS,
    _VISION_DATASET_TYPES,
    _resolve_linear_dataset_mlp_dims,
    _sample_base_vector_dataset_arrays,
)
from comfy_research.engine.trainer.graph import _incoming
from comfy_research.engine.trainer.model_helpers import _atomic_chain_dataset_dims
from comfy_research.engine.trainer.provider_types import (
    BuildBranch,
    DatasetMaterializeRequest,
    DatasetProvider,
)
from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int, _scalar_str
from comfy_research.engine.trainer.teacher_helpers import (
    _resolve_teacher_model_eval,
    _sample_teacher_input_tensor,
)
from comfy_research.engine.datasets.vision_datasets_runtime import (
    build_vision_numpy_arrays,
    vision_num_classes,
    _vision_flatten_feature_matrix,
)
from comfy_research.engine.models.vision_models import infer_vision_input_channels_height_width
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.nodes.provider_types import DatasetMaterializeContext
from comfy_research.nodes.registry import dataset_defs_materializers
from comfy_research.schemas.graph import NodeKind

_Branch = Literal["atomic", "vector"]


def materialize_dataset_for_training(req: DatasetMaterializeRequest) -> DatasetArrays:
    """Dispatch through the branch registry; the branch was computed once by the
    calling stage via provider_types._build_branch, reproducing
    build_training_objects_stage's four-way split verbatim. The caller
    (materialize_dataset_stage) owns tensor conversion and all state writes.
    """
    return DATASET_PROVIDERS[req.branch](req)


def _materialize_vision_branch(req: DatasetMaterializeRequest) -> DatasetArrays:
    return _materialize_vision(req.ds_train, req.dd_train, req.train_size, req.test_size, req.rng)


def _materialize_for_atomic(req: DatasetMaterializeRequest) -> DatasetArrays:
    """Atomic-chain branch: chain/dims first (no RNG; the atomic model
    provider re-resolves the same chain), then teacher or dense-family
    dispatch."""
    if req.combined_flat_chain is not None:
        chain = req.combined_flat_chain
    else:
        chain = collect_atomic_layer_chain_front_to_back(req.model_node, req.edges, req.nmap)
    if not chain:
        raise HTTPException(status_code=400, detail="Atomic model layer chain is empty.")
    md_atomic = _atomic_chain_dataset_dims(chain)
    if req.ds_train.type == NodeKind.teacher_dataset:
        return _materialize_teacher(
            req.ds_train, req.ds_test_raw, req.edges, req.nmap, req.nodes, req.seed, md_atomic, req.test_size,
            branch="atomic",
        )
    return _materialize_dense_families(req, md_atomic, branch="atomic")


def _materialize_for_vector(req: DatasetMaterializeRequest) -> DatasetArrays:
    md: dict[str, Any] = req.model_node.data or {}
    if req.ds_train.type == NodeKind.teacher_dataset:
        # Vector-only pre-check (KNOWN_DIVERGENCES: teacher/vector-diffusion-precheck;
        # unreachable through the full pipeline — graph validation intercepts
        # earlier with its own message — preserved verbatim as defense in depth).
        if has_capability(req.model_node.type, "diffusion_loss_model"):
            raise HTTPException(
                status_code=400,
                detail="diffusion_score_model cannot be trained with teacher_dataset.",
            )
        return _materialize_teacher(
            req.ds_train, req.ds_test_raw, req.edges, req.nmap, req.nodes, req.seed, md, req.test_size,
            branch="vector",
        )
    return _materialize_dense_families(req, md, branch="vector")


def _materialize_dense_families(req: DatasetMaterializeRequest, md: dict[str, Any], *, branch: _Branch) -> DatasetArrays:
    """Non-teacher family dispatch shared by the atomic and vector branches —
    identical conditions and order in both originals; ``branch`` selects the
    documented message divergences (mixer wording, terminal unsupported
    wording)."""
    dd_test = req.dd_test
    dd_train = req.dd_train
    ds_test_raw = req.ds_test_raw
    ds_train = req.ds_train
    edges = req.edges
    nmap = req.nmap
    rng = req.rng
    test_size = req.test_size
    train_size = req.train_size
    trainer_task = req.trainer_task

    input_dist = _scalar_str(dd_train.get("inputDistribution"), "standard_normal")
    out_dist = _scalar_str(dd_train.get("outputDistribution"), "additive_gaussian")
    mem_alpha = _scalar_float(dd_train.get("alpha"), 1.0)
    noise_level = _scalar_float(dd_train.get("noiseLevel"), 0.25)
    additive = out_dist == "additive_gaussian"
    sigma = noise_level if additive else 0.0

    input_dist_test = _scalar_str(dd_test.get("inputDistribution"), input_dist)
    out_dist_test = _scalar_str(dd_test.get("outputDistribution"), out_dist)
    mem_alpha_test = _scalar_float(dd_test.get("alpha"), mem_alpha)
    noise_test = _scalar_float(dd_test.get("noiseLevel"), noise_level)
    additive_test = out_dist_test == "additive_gaussian"
    sigma_test = noise_test if additive_test else 0.0

    if ds_train.type in _DATASET_MIXER_TYPES:
        return _materialize_mixer(
            ds_train, ds_test_raw, edges, nmap, trainer_task, train_size, test_size, md,
            branch=branch,
        )
    input_dim, output_dim = _resolve_linear_dataset_mlp_dims(
        dd_train,
        md,
        ds_train.type,
        model_kind=req.model_node.type,
        binary_single_logit=req.binary_single_logit,
        memorization_a_ce_slot_groups=req.ce_mem_a_slots,
    )
    # NodeDef 通道采用 generated-first；该分支位于 mixer early-return 之后、
    # dims 解析之后、family arms 之前)。provider 收到的 dims 就是上面算出的值——
    # "搬 arm,不重解释";空集恒 miss。
    # NodeKind 是 str-enum:成员与值串 hash/== 等价,直接作 dict 键命中 plain-str
    # 注册表;str(NodeKind.x) 会给 "NodeKind.x",不可用。
    _mat = dataset_defs_materializers().get(ds_train.type)
    if _mat is not None:
        return _mat(DatasetMaterializeContext(
            ds_type=ds_train.type,
            rng=rng,
            dd_train=dd_train,
            dd_test=dd_test,
            train_size=train_size,
            test_size=test_size,
            input_dim=input_dim,
            output_dim=output_dim,
            ds_test_raw=ds_test_raw,
        ))
    if ds_train.type == NodeKind.memorization_b_dataset and trainer_task == "cross_entropy_dense":
        return _materialize_memorization_b_dense(
            rng, dd_train, dd_test, ds_test_raw, train_size, test_size,
            out_dist, mem_alpha, out_dist_test, mem_alpha_test, input_dim, output_dim,
        )
    if ds_train.type == NodeKind.memorization_a_dataset and trainer_task == "cross_entropy_dense":
        return _materialize_memorization_a_dense(
            rng, train_size, test_size, input_dim, output_dim,
            input_dist, out_dist, mem_alpha, input_dist_test, out_dist_test, mem_alpha_test,
        )
    if ds_train.type in _VISION_DATASET_TYPES and trainer_task == "cross_entropy_dense":
        return _materialize_vision_flatten_dense(
            ds_train, dd_train, rng, train_size, test_size, input_dim, output_dim,
        )
    # Terminal wording diverges per branch (KNOWN_DIVERGENCES: unsupported_dataset;
    # unreachable through the full pipeline — graph validation intercepts earlier).
    unsupported_label = "atomic-chain MLP regression" if branch == "atomic" else "full-model MLP regression"
    raise HTTPException(
        status_code=400,
        detail=(
            f"Dataset type {ds_train.type} is not supported for {unsupported_label} "
            "with this loss configuration."
        ),
    )


def _materialize_for_token(req: DatasetMaterializeRequest) -> DatasetArrays:
    """Token family chain, verbatim from build_token_stage."""
    dd_test = req.dd_test
    dd_train = req.dd_train
    ds_test_raw = req.ds_test_raw
    ds_train = req.ds_train
    rng = req.rng
    test_size = req.test_size
    train_size = req.train_size
    if ds_train.type == NodeKind.memorization_b_dataset:
        return _materialize_memorization_b_token(
            rng, dd_train, dd_test, ds_test_raw, train_size, test_size,
        )
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 4)
    if ds_train.type == NodeKind.unigram_dataset:
        return _materialize_unigram(rng, dd_train, dd_test, train_size, test_size, vocab_ds)
    elif ds_train.type == NodeKind.token_prediction_dataset:
        return _materialize_token_prediction(rng, dd_train, dd_test, train_size, test_size, vocab_ds)
    elif ds_train.type == NodeKind.bigram_low_rank_dataset:
        return _materialize_bigram_low_rank(dd_train, dd_test, ds_test_raw, train_size, test_size)
    elif ds_train.type == NodeKind.circle_random_walk_dataset:
        return _materialize_circle_random_walk(rng, dd_train, dd_test, train_size, test_size)
    elif ds_train.type == NodeKind.circular_motion_dataset:
        return _materialize_circular_motion(dd_train, dd_test, ds_test_raw, train_size, test_size)
    elif ds_train.type == NodeKind.in_context_associative_recall_dataset:
        return _materialize_in_context_associative_recall(rng, dd_train, dd_test, train_size, test_size)
    elif ds_train.type == NodeKind.modular_addition_dataset:
        return _materialize_modular_addition(rng, dd_train, dd_test, train_size, test_size)
    elif ds_train.type in _TOY_LANGUAGE_TOKEN_DATASETS:
        return _materialize_toy_language(ds_train, dd_train, dd_test, train_size, test_size)
    else:
        return _materialize_fallback_circle_walk(
            rng, dd_train, dd_test, train_size, test_size, vocab_ds,
        )


# Branch-level providers own family-specific dataset dispatch.
DATASET_PROVIDERS: dict[BuildBranch, DatasetProvider] = {
    "vision": _materialize_vision_branch,
    "atomic": _materialize_for_atomic,
    "vector": _materialize_for_vector,
    "token": _materialize_for_token,
}


# ---------------------------------------------------------------------------
# vision (F1a)
# ---------------------------------------------------------------------------

def _materialize_vision(ds_train, dd_train, train_size, test_size, rng) -> DatasetArrays:
    ch, h, w = infer_vision_input_channels_height_width(ds_train.type, dd_train)
    x_np, y_np, x_te_np, y_te_np = build_vision_numpy_arrays(
        ds_train.type, dd_train, train_size, test_size, rng
    )
    if int(x_np.shape[1]) != ch or int(x_np.shape[2]) != h or int(x_np.shape[3]) != w:
        raise HTTPException(
            status_code=500,
            detail="Internal: vision dataset output shape does not match inferred (C,H,W).",
        )
    n_cls = vision_num_classes(ds_train.type, dd_train)
    extras: dict[str, Any] = {"image_size": h}
    recipe = _scalar_str(dd_train.get("trainingRecipe"), "standard").strip().lower()
    if ds_train.type == NodeKind.cifar10_dataset and recipe == "jastrzbski_fig1":
        from comfy_research.engine.datasets.vision_datasets_runtime import cifar10_pixel_mean_global_std

        pixel_mean, global_std = cifar10_pixel_mean_global_std(x_np)
        if x_te_np is not None:
            x_te_np = np.asarray((x_te_np - pixel_mean) / global_std, dtype=np.float32)
        extras.update(
            training_recipe=recipe,
            pixel_mean=np.asarray(pixel_mean, dtype=np.float32),
            global_std=global_std,
        )
    return DatasetArrays(
        x_np=x_np, y_np=y_np, x_test_np=x_te_np, y_test_np=y_te_np,
        input_dim=ch, output_dim=n_cls,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# atomic/vector dense families (F1b/F1c)
# ---------------------------------------------------------------------------

def _materialize_teacher(
    ds_train, ds_test_raw, edges, nmap, nodes, seed, md, test_size, *, branch: _Branch,
) -> DatasetArrays:
    """Teacher path: teacher torch-init, fresh-numpy input sampling, no shared
    rng. ``branch`` selects the dim-mismatch 400 wording (KNOWN_DIVERGENCES:
    teacher/atomic-vs-vector); bodies were otherwise identical."""
    dims_label = "Student chain I/O" if branch == "atomic" else "Student model dimensions"
    teacher_src = _incoming(edges, nmap, ds_train.id, "model")
    teacher, input_dim, output_dim = _resolve_teacher_model_eval(teacher_src, nodes, edges, seed)
    x_np, train_sampler = _sample_teacher_input_tensor(edges, nmap, ds_train, "train_input", None)
    if x_np.shape[1] != input_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Train input tensor dim ({x_np.shape[1]}) must match teacher input dim ({input_dim})."
            ),
        )
    m_in = _scalar_int(md.get("inputDim"), input_dim)
    m_out = _scalar_int(md.get("outputDim"), output_dim)
    if m_in != input_dim or m_out != output_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{dims_label} ({m_in}, {m_out}) must match the teacher ({input_dim}, {output_dim})."
            ),
        )
    y_np = teacher_labels_numpy(teacher, x_np)
    x_test_np: np.ndarray | None = None
    y_test_np: np.ndarray | None = None
    test_td = ds_test_raw if ds_test_raw is not None else ds_train
    test_sampler_wired = _incoming(edges, nmap, test_td.id, "test_input") is not None
    legacy_rid_wired = _incoming(edges, nmap, test_td.id, "input_distribution") is not None
    if test_sampler_wired or (legacy_rid_wired and test_size > 0):
        x_test_np, _ = _sample_teacher_input_tensor(edges, nmap, test_td, "test_input", train_sampler)
        if x_test_np.shape[1] != input_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Test input tensor dim must match the teacher "
                    f"input dim ({input_dim}); got {x_test_np.shape[1]}."
                ),
            )
        y_test_np = teacher_labels_numpy(teacher, x_test_np)
    return DatasetArrays(
        x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
        input_dim=input_dim, output_dim=output_dim,
        extras={"teacher": teacher},
    )


def _materialize_mixer(
    ds_train, ds_test_raw, edges, nmap, trainer_task, train_size, test_size, md, *, branch: _Branch,
) -> DatasetArrays:
    """``branch`` selects the dim-mismatch 400 wording (KNOWN_DIVERGENCES:
    mixer/atomic-vs-vector); bodies were otherwise identical."""
    dims_label = "student chain dimensions" if branch == "atomic" else "student model dimensions"
    x_np, y_np = _sample_dataset_arrays_for_mixer(ds_train, train_size, edges, nmap, trainer_task, leaf_sampler=_sample_base_vector_dataset_arrays)
    if x_np.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail="dataset_mixer A/B currently supports 2D input tensors [batch, input_dim].",
        )
    input_dim = int(x_np.shape[1])
    output_dim = int(y_np.shape[1]) if y_np.ndim == 2 else 1
    m_in = _scalar_int(md.get("inputDim"), input_dim)
    m_out = _scalar_int(md.get("outputDim"), output_dim)
    if m_in != input_dim or m_out != output_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"dataset_mixer A/B mixed dimensions ({input_dim}, {output_dim}) must match "
                f"{dims_label} ({m_in}, {m_out})."
            ),
        )
    x_test_np = y_test_np = None
    if test_size > 0:
        test_src = ds_test_raw if ds_test_raw is not None else ds_train
        x_test_np, y_test_np = _sample_dataset_arrays_for_mixer(test_src, test_size, edges, nmap, trainer_task, leaf_sampler=_sample_base_vector_dataset_arrays)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_paired_split(builder, ds_type, rng, dd_train, dd_test, train_size, test_size,
                              input_dim, output_dim, missing_msg) -> DatasetArrays:
    """uniform_linear_motion / kepler_2d / pde_field: one builder
    call draws both splits from the shared rng; the test_size>0 validation is
    the original inline check verbatim."""
    x_np, y_np, x_test_np, y_test_np = builder()
    if test_size > 0 and (x_test_np is None or y_test_np is None):
        raise HTTPException(status_code=500, detail=missing_msg)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_memorization_b_dense(
    rng, dd_train, dd_test, ds_test_raw, train_size, test_size,
    out_dist, mem_alpha, out_dist_test, mem_alpha_test, input_dim, output_dim,
) -> DatasetArrays:
    """memB dense train+test draws (atomic and vector, aligned in)."""
    vocab = _memorization_b_vocab_size(dd_train)
    x_np, y_np = _memorization_b_xy_numpy(rng, train_size, vocab, out_dist, mem_alpha)
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _memorization_b_vocab_size(dd_test if ds_test_raw is not None else dd_train)
        x_test_np, y_test_np = _memorization_b_xy_numpy(
            rng, test_size, vocab_te, out_dist_test, mem_alpha_test
        )
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_memorization_a_dense(
    rng, train_size, test_size, input_dim, output_dim,
    input_dist, out_dist, mem_alpha, input_dist_test, out_dist_test, mem_alpha_test,
) -> DatasetArrays:
    x_np = sample_inputs(rng, train_size, input_dim, input_dist)
    y_np = _sample_memorization_labels(rng, train_size, output_dim, out_dist, mem_alpha)
    x_test_np = y_test_np = None
    if test_size > 0:
        x_test_np = sample_inputs(rng, test_size, input_dim, input_dist_test)
        y_test_np = _sample_memorization_labels(
            rng, test_size, output_dim, out_dist_test, mem_alpha_test
        )
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_vision_flatten_dense(
    ds_train, dd_train, rng, train_size, test_size, input_dim, output_dim,
) -> DatasetArrays:
    x4, y_np, x4_te, y_te = build_vision_numpy_arrays(
        ds_train.type, dd_train, train_size, test_size, rng
    )
    x_np = _vision_flatten_feature_matrix(x4)
    if test_size > 0:
        if x4_te is None or y_te is None or int(x4_te.shape[0]) == 0:
            raise HTTPException(
                status_code=400,
                detail="vision dataset has no test rows; increase test size or set test size to 0.",
            )
        x_test_np = _vision_flatten_feature_matrix(x4_te)
        y_test_np = y_te
    else:
        x_test_np = None
        y_test_np = None
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_linear_like(
    rng, dd_train, dd_test, ds_test_raw, train_size, test_size, input_dim, output_dim,
    input_dist, sigma, additive, input_dist_test, sigma_test, additive_test,
) -> DatasetArrays:
    x_np = sample_inputs(rng, train_size, input_dim, input_dist)
    w = None if _scalar_str(dd_train.get("targetKind"), "linear") == "xor2d" else _random_linear_weights(
        _linear_regression_weight_rng(dd_train), input_dim, output_dim
    )
    y_np = _linear_like_targets_from_x(
        dd_train, x_np, input_dim, output_dim, sigma=sigma, additive=additive, rng=rng
    )
    x_test_np = y_test_np = None
    if test_size > 0:
        x_test_np = sample_inputs(rng, test_size, input_dim, input_dist_test)
        if _scalar_str(dd_train.get("targetKind"), "linear") == "xor2d":
            y_test_np = _linear_like_targets_from_x(
                dd_test if ds_test_raw is not None else dd_train,
                x_test_np,
                input_dim,
                output_dim,
                sigma=sigma_test,
                additive=additive_test,
                rng=rng,
            )
        else:
            if w is None:
                raise HTTPException(status_code=500, detail="Internal: linear dataset weights missing.")
            y_test_np = _apply_linear_map(x_test_np, w, sigma_test, additive_test, rng)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


def _materialize_symbolic_func(
    rng, dd_train, dd_test, ds_test_raw, train_size, test_size, input_dim, output_dim,
    input_dist, sigma, additive, input_dist_test, sigma_test, additive_test,
) -> DatasetArrays:
    x_np = sample_inputs(rng, train_size, input_dim, input_dist)
    y_fn = build_y_numpy_fn(dd_train)
    y_np = y_fn(x_np)
    if additive and sigma > 0:
        y_np = y_np + sigma * rng.standard_normal((y_np.shape[0], y_np.shape[1])).astype(np.float32)
    x_test_np = y_test_np = None
    if test_size > 0:
        x_test_np = sample_inputs(rng, test_size, input_dim, input_dist_test)
        y_test_fn = build_y_numpy_fn(dd_test if ds_test_raw is not None else dd_train)
        y_test_np = y_test_fn(x_test_np)
        if additive_test and sigma_test > 0:
            y_test_np = y_test_np + sigma_test * rng.standard_normal(
                (y_test_np.shape[0], y_test_np.shape[1])
            ).astype(np.float32)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=input_dim, output_dim=output_dim)


# ---------------------------------------------------------------------------
# Token dataset families.
# ---------------------------------------------------------------------------

def _materialize_memorization_b_token(
    rng, dd_train, dd_test, ds_test_raw, train_size, test_size,
) -> DatasetArrays:
    """memB as a real token family : [batch, 1] token->token
    memorization pairs drawn from the class-probability vocab."""
    vocab_ds = _memorization_b_vocab_size(dd_train)
    ctx_len_ds = 1
    alpha = _scalar_float(dd_train.get("alpha"), 1.0)
    out_dist_tr = _scalar_str(dd_train.get("outputDistribution"), "uniform_class_probs").strip()
    probs = _memorization_class_probs(vocab_ds, out_dist_tr, alpha)
    x_ids = rng.choice(vocab_ds, size=(train_size,), p=probs).astype(np.int64)
    y_np = rng.choice(vocab_ds, size=(train_size,), p=probs).astype(np.int64)
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _memorization_b_vocab_size(dd_test if ds_test_raw is not None else dd_train)
        if vocab_te != vocab_ds:
            raise HTTPException(
                status_code=400,
                detail="Memorization B test dataset vocabSize must match train.",
            )
        alpha_te = _scalar_float(dd_test.get("alpha"), alpha)
        out_dist_te = _scalar_str(dd_test.get("outputDistribution"), out_dist_tr).strip()
        probs_te = _memorization_class_probs(vocab_ds, out_dist_te, alpha_te)
        x_te_ids = rng.choice(vocab_ds, size=(test_size,), p=probs_te).astype(np.int64)
        y_te = rng.choice(vocab_ds, size=(test_size,), p=probs_te).astype(np.int64)
        x_test_np = x_te_ids[:, None]
        y_test_np = y_te
    return DatasetArrays(x_np=x_ids[:, None], y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_unigram(
    rng, dd_train, dd_test, train_size, test_size, vocab_ds,
) -> DatasetArrays:
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    alpha = _scalar_float(dd_train.get("alpha"), 1.0)
    out_dist_tr = _scalar_str(dd_train.get("outputDistribution"), "power_law_class_probs").strip()
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="Unigram dataset vocabSize must be >= 2.")
    if ctx_len_ds < 1:
        raise HTTPException(status_code=400, detail="Unigram dataset contextLength must be >= 1.")
    if alpha <= 0:
        raise HTTPException(status_code=400, detail="Unigram dataset alpha must be > 0.")
    probs = _unigram_token_probs(vocab_ds, out_dist_tr, alpha)
    tokens_np = rng.choice(vocab_ds, size=(train_size, ctx_len_ds), p=probs).astype(np.int64)
    y_np = rng.choice(vocab_ds, size=(train_size,), p=probs).astype(np.int64)
    x_test_np = y_test_np = None
    if test_size > 0:
        alpha_te = _scalar_float(dd_test.get("alpha"), alpha)
        ctx_te = _scalar_int(dd_test.get("contextLength"), ctx_len_ds)
        vocab_te = _scalar_int(dd_test.get("vocabSize"), vocab_ds)
        if vocab_te != vocab_ds or ctx_te != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail="Unigram test dataset vocabSize/contextLength must match train.",
            )
        out_dist_te = _scalar_str(dd_test.get("outputDistribution"), out_dist_tr).strip()
        if out_dist_te != out_dist_tr:
            raise HTTPException(
                status_code=400,
                detail="Unigram test dataset outputDistribution must match train.",
            )
        probs_te = _unigram_token_probs(vocab_ds, out_dist_te, alpha_te)
        x_test_np = rng.choice(vocab_ds, size=(test_size, ctx_len_ds), p=probs_te).astype(np.int64)
        y_test_np = rng.choice(vocab_ds, size=(test_size,), p=probs_te).astype(np.int64)
    return DatasetArrays(x_np=tokens_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_token_prediction(
    rng, dd_train, dd_test, train_size, test_size, vocab_ds,
) -> DatasetArrays:
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 4)
    retrieval_mode = _scalar_str(dd_train.get("retrievalMode"), "position").strip().lower()
    which_token = _scalar_int(dd_train.get("whichToken"), -1)
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="vocabSize must be >= 2")
    if ctx_len_ds < 1:
        raise HTTPException(status_code=400, detail="contextLength must be >= 1")
    if retrieval_mode not in {"position", "content"}:
        raise HTTPException(status_code=400, detail="retrievalMode must be 'position' or 'content'.")
    if retrieval_mode == "position" and not (-ctx_len_ds <= which_token <= ctx_len_ds - 1):
        raise HTTPException(
            status_code=400,
            detail=(
                f"whichToken must be in [0, {ctx_len_ds - 1}] or "
                f"[-1, -{ctx_len_ds}] for contextLength={ctx_len_ds}."
            ),
        )
    if retrieval_mode == "content" and ctx_len_ds < 2:
        raise HTTPException(status_code=400, detail="content retrieval mode requires contextLength >= 2.")
    tokens_np = rng.integers(0, vocab_ds, size=(train_size, ctx_len_ds), dtype=np.int64)
    if retrieval_mode == "content":
        current = tokens_np[:, -1][:, None]
        previous = tokens_np[:, :-1]
        distances = np.abs(previous - current)
        nearest_idx = np.argmin(distances, axis=1)
        y_np = previous[np.arange(previous.shape[0]), nearest_idx].copy()
    else:
        y_np = tokens_np[:, which_token].copy()
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _scalar_int(dd_test.get("vocabSize"), vocab_ds)
        ctx_te = _scalar_int(dd_test.get("contextLength"), ctx_len_ds)
        retrieval_mode_te = _scalar_str(dd_test.get("retrievalMode"), retrieval_mode).strip().lower()
        which_token_te = _scalar_int(dd_test.get("whichToken"), which_token)
        if vocab_te != vocab_ds or ctx_te != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail="Test token dataset vocabSize/contextLength must match train when both are set.",
            )
        if retrieval_mode_te != retrieval_mode:
            raise HTTPException(status_code=400, detail="Test retrievalMode must match train retrievalMode.")
        if retrieval_mode_te == "position" and not (-ctx_te <= which_token_te <= ctx_te - 1):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"test whichToken must be in [0, {ctx_te - 1}] or "
                    f"[-1, -{ctx_te}] for contextLength={ctx_te}."
                ),
            )
        if retrieval_mode_te == "content" and ctx_te < 2:
            raise HTTPException(status_code=400, detail="content retrieval mode requires contextLength >= 2.")
        tokens_te = rng.integers(0, vocab_ds, size=(test_size, ctx_len_ds), dtype=np.int64)
        if retrieval_mode_te == "content":
            current_te = tokens_te[:, -1][:, None]
            previous_te = tokens_te[:, :-1]
            distances_te = np.abs(previous_te - current_te)
            nearest_te = np.argmin(distances_te, axis=1)
            y_te = previous_te[np.arange(previous_te.shape[0]), nearest_te].copy()
        else:
            y_te = tokens_te[:, which_token_te].copy()
        x_test_np = tokens_te
        y_test_np = y_te
    return DatasetArrays(x_np=tokens_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_bigram_low_rank(
    dd_train, dd_test, ds_test_raw, train_size, test_size,
) -> DatasetArrays:
    """No shared-rng draws: the builder derives its own generator from dd."""
    x_np, y_np, x_test_np, y_test_np, vocab_ds = _build_bigram_low_rank_arrays(
        dd_train,
        dd_test if ds_test_raw is not None else dd_train,
        train_size,
        test_size,
    )
    if x_test_np is None or y_test_np is None:  # original converted only when BOTH present
        x_test_np = y_test_np = None
    ctx_len_ds = 1
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_circle_random_walk(
    rng, dd_train, dd_test, train_size, test_size,
) -> DatasetArrays:
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 10)
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    p_right = _scalar_float(dd_train.get("rightStepProb"), 0.5)
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="Circle random walk dataset vocabSize must be >= 2.")
    if ctx_len_ds < 1:
        raise HTTPException(status_code=400, detail="Circle random walk dataset contextLength must be >= 1.")
    if p_right < 0.0 or p_right > 1.0:
        raise HTTPException(status_code=400, detail="Circle random walk dataset rightStepProb must be in [0, 1].")

    def _sample_circle_walk(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, ctx_len_ds), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        seq = np.empty((n, ctx_len_ds + 1), dtype=np.int64)
        seq[:, 0] = rng.integers(0, vocab_ds, size=(n,), dtype=np.int64)
        steps = np.where(rng.random((n, ctx_len_ds)) < p_right, 1, -1).astype(np.int64)
        for t in range(ctx_len_ds):
            seq[:, t + 1] = (seq[:, t] + steps[:, t]) % vocab_ds
        return seq[:, :ctx_len_ds], seq[:, ctx_len_ds]

    x_np, y_np = _sample_circle_walk(train_size)
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _scalar_int(dd_test.get("vocabSize"), vocab_ds)
        ctx_te = _scalar_int(dd_test.get("contextLength"), ctx_len_ds)
        p_right_te = _scalar_float(dd_test.get("rightStepProb"), p_right)
        if vocab_te != vocab_ds or ctx_te != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail="Circle random walk test dataset vocabSize/contextLength must match train.",
            )
        if p_right_te != p_right:
            raise HTTPException(
                status_code=400,
                detail="Circle random walk test dataset rightStepProb must match train.",
            )
        x_test_np, y_test_np = _sample_circle_walk(test_size)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_circular_motion(
    dd_train, dd_test, ds_test_raw, train_size, test_size,
) -> DatasetArrays:
    """No shared-rng draws: the builder derives its own generator from dd."""
    x_np, y_np, x_test_np, y_test_np, vocab_ds, ctx_len_ds = _build_circular_motion_arrays(
        dd_train,
        dd_test if ds_test_raw is not None else dd_train,
        train_size,
        test_size,
    )
    if x_test_np is None or y_test_np is None:  # original converted only when BOTH present
        x_test_np = y_test_np = None
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_in_context_associative_recall(
    rng, dd_train, dd_test, train_size, test_size,
) -> DatasetArrays:
    vocab_ds = _scalar_int(dd_train.get("vocabSize"), 64)
    num_pairs = _scalar_int(dd_train.get("numPairs"), 32)
    in_context_repeat = _scalar_int(dd_train.get("inContextRepeat"), 1)
    cross_repeat_prob = _scalar_float(dd_train.get("crossSampleRepeatProb"), 0.0)
    repeated_token_count = _scalar_int(dd_train.get("repeatedTokenCount"), 2)
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="Associative recall dataset vocabSize must be >= 2.")
    if num_pairs < 1:
        raise HTTPException(status_code=400, detail="Associative recall dataset numPairs must be >= 1.")
    if in_context_repeat < 1:
        raise HTTPException(status_code=400, detail="Associative recall dataset inContextRepeat must be >= 1.")
    if cross_repeat_prob < 0.0 or cross_repeat_prob > 1.0:
        raise HTTPException(
            status_code=400,
            detail="Associative recall dataset crossSampleRepeatProb must be in [0, 1].",
        )
    if repeated_token_count < 1:
        raise HTTPException(status_code=400, detail="Associative recall dataset repeatedTokenCount must be >= 1.")
    if repeated_token_count > vocab_ds:
        raise HTTPException(
            status_code=400,
            detail="Associative recall dataset repeatedTokenCount must be <= vocabSize.",
        )

    ctx_len_ds = 2 * num_pairs + 1
    repeated_pool = np.arange(repeated_token_count, dtype=np.int64)

    def _sample_associative_batch(n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.zeros((0, ctx_len_ds), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        x = np.empty((n, ctx_len_ds), dtype=np.int64)
        y = np.empty((n,), dtype=np.int64)
        for i in range(n):
            keys = rng.permutation(vocab_ds)[:num_pairs].astype(np.int64)
            vals = rng.permutation(vocab_ds)[:num_pairs].astype(np.int64)
            query_slot = int(rng.integers(0, num_pairs))
            query = int(keys[query_slot])
            if cross_repeat_prob > 0 and repeated_pool.size > 0 and float(rng.random()) < cross_repeat_prob:
                query = int(repeated_pool[int(rng.integers(0, repeated_pool.size))])
                hit = np.where(keys == query)[0]
                if hit.size == 0:
                    query_slot = int(rng.integers(0, num_pairs))
                    keys[query_slot] = query
                else:
                    query_slot = int(hit[0])
            if in_context_repeat > 1:
                candidates = np.arange(num_pairs, dtype=np.int64)
                candidates = candidates[candidates != query_slot]
                if candidates.size > 0:
                    rng.shuffle(candidates)
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

    x_np, y_np = _sample_associative_batch(train_size)
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _scalar_int(dd_test.get("vocabSize"), vocab_ds)
        pairs_te = _scalar_int(dd_test.get("numPairs"), num_pairs)
        b_te = _scalar_int(dd_test.get("inContextRepeat"), in_context_repeat)
        p_te = _scalar_float(dd_test.get("crossSampleRepeatProb"), cross_repeat_prob)
        k_te = _scalar_int(dd_test.get("repeatedTokenCount"), repeated_token_count)
        if (
            vocab_te != vocab_ds
            or pairs_te != num_pairs
            or b_te != in_context_repeat
            or p_te != cross_repeat_prob
            or k_te != repeated_token_count
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Associative recall test dataset must match train for vocabSize, numPairs, "
                    "inContextRepeat, crossSampleRepeatProb, and repeatedTokenCount."
                ),
            )
        x_test_np, y_test_np = _sample_associative_batch(test_size)
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_modular_addition(
    rng, dd_train, dd_test, train_size, test_size,
) -> DatasetArrays:
    vocab_ds = _scalar_int(dd_train.get("modulus"), 59)
    ctx_len_ds = 2
    train_fraction = _scalar_float(dd_train.get("trainFraction"), 0.3)
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="Modular addition dataset modulus must be >= 2.")
    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise HTTPException(status_code=400, detail="Modular addition dataset trainFraction must be in (0, 1).")
    a, b = np.meshgrid(
        np.arange(vocab_ds, dtype=np.int64),
        np.arange(vocab_ds, dtype=np.int64),
        indexing="ij",
    )
    x_all = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
    y_all = ((x_all[:, 0] + x_all[:, 1]) % vocab_ds).astype(np.int64)
    perm = rng.permutation(x_all.shape[0])
    x_all = x_all[perm]
    y_all = y_all[perm]
    n_train = int(round(train_fraction * x_all.shape[0]))
    n_train = min(max(n_train, 1), x_all.shape[0])
    x_np = x_all[:n_train].copy()
    y_np = y_all[:n_train].copy()
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _scalar_int(dd_test.get("modulus"), vocab_ds)
        frac_te = _scalar_float(dd_test.get("trainFraction"), train_fraction)
        if vocab_te != vocab_ds:
            raise HTTPException(
                status_code=400,
                detail="Test modular addition dataset modulus must match train.",
            )
        if frac_te != train_fraction:
            raise HTTPException(
                status_code=400,
                detail="Test modular addition dataset trainFraction must match train.",
            )
        n_remain = int(x_all.shape[0]) - n_train
        n_take = min(int(test_size), n_remain)
        if n_take > 0:
            x_test_np = x_all[n_train : n_train + n_take].copy()
            y_test_np = y_all[n_train : n_train + n_take].copy()
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_toy_language(
    ds_train, dd_train, dd_test, train_size, test_size,
) -> DatasetArrays:
    """No shared-rng draws: the toy-LM builders seed their own generators."""
    vocab_ds = max(
        2,
        _scalar_int(
            dd_train.get("vocabSize"),
            _scalar_int(dd_train.get("vocabCap"), 256 if ds_train.type in _TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES else 32),
        ),
    )
    ctx_len_ds = max(1, _scalar_int(dd_train.get("contextLength"), 64 if ds_train.type in _TEXT_HEAVY_TOY_LANGUAGE_DATASET_TYPES else 16))
    x_np, y_np, x_te_np, y_te_np = _materialize_toy_language_lm_numpy(ds_train.type, dd_train, train_size, test_size)
    if int(x_np.shape[1]) != ctx_len_ds and x_np.shape[0] > 0:
        ctx_len_ds = int(x_np.shape[1])
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = max(
            2,
            _scalar_int(
                dd_test.get("vocabSize"),
                _scalar_int(dd_test.get("vocabCap"), vocab_ds),
            ),
        )
        ctx_te = max(1, _scalar_int(dd_test.get("contextLength"), ctx_len_ds))
        if vocab_te != vocab_ds or ctx_te != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail="Toy language dataset test node must match train vocabSize/vocabCap and contextLength.",
            )
        x_test_np = x_te_np
        y_test_np = y_te_np
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)


def _materialize_fallback_circle_walk(
    rng, dd_train, dd_test, train_size, test_size, vocab_ds,
) -> DatasetArrays:
    """Terminal else of the token family chain: circle-walk with the CALLER's
    vocab_ds (dataset vocabSize default 4). Near-copy of
    _materialize_circle_random_walk but with different vocab sourcing and
    test-mismatch messages
    (KNOWN_DIVERGENCES: circle_random_walk/token-family-vs-fallback)."""
    ctx_len_ds = _scalar_int(dd_train.get("contextLength"), 1)
    p_right = _scalar_float(dd_train.get("rightStepProb"), 0.5)
    if vocab_ds < 2:
        raise HTTPException(status_code=400, detail="Circle random walk dataset vocabSize must be >= 2.")
    if ctx_len_ds < 1:
        raise HTTPException(status_code=400, detail="Circle random walk dataset contextLength must be >= 1.")
    if p_right < 0.0 or p_right > 1.0:
        raise HTTPException(
            status_code=400,
            detail="Circle random walk dataset rightStepProb must be in [0, 1].",
        )
    seq = np.empty((train_size, ctx_len_ds + 1), dtype=np.int64)
    seq[:, 0] = rng.integers(0, vocab_ds, size=(train_size,), dtype=np.int64)
    steps = np.where(rng.random((train_size, ctx_len_ds)) < p_right, 1, -1).astype(np.int64)
    for t in range(ctx_len_ds):
        seq[:, t + 1] = (seq[:, t] + steps[:, t]) % vocab_ds
    x_np = seq[:, :ctx_len_ds].copy()
    y_np = seq[:, ctx_len_ds].copy()
    x_test_np = y_test_np = None
    if test_size > 0:
        vocab_te = _scalar_int(dd_test.get("vocabSize"), vocab_ds)
        ctx_te = _scalar_int(dd_test.get("contextLength"), ctx_len_ds)
        p_right_te = _scalar_float(dd_test.get("rightStepProb"), p_right)
        if vocab_te != vocab_ds or ctx_te != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail="Test circle random walk dataset vocabSize/contextLength must match train.",
            )
        if p_right_te != p_right:
            raise HTTPException(
                status_code=400,
                detail="Test circle random walk dataset rightStepProb must match train.",
            )
        seq_te = np.empty((test_size, ctx_len_ds + 1), dtype=np.int64)
        seq_te[:, 0] = rng.integers(0, vocab_ds, size=(test_size,), dtype=np.int64)
        steps_te = np.where(rng.random((test_size, ctx_len_ds)) < p_right, 1, -1).astype(np.int64)
        for t in range(ctx_len_ds):
            seq_te[:, t + 1] = (seq_te[:, t] + steps_te[:, t]) % vocab_ds
        x_test_np = seq_te[:, :ctx_len_ds].copy()
        y_test_np = seq_te[:, ctx_len_ds].copy()
    return DatasetArrays(x_np=x_np, y_np=y_np, x_test_np=x_test_np, y_test_np=y_test_np,
                         input_dim=ctx_len_ds, output_dim=vocab_ds)
