"""Trainer graph resolution, task determination, and compatibility checks.

Each function reads its inputs from PrepareState and writes validated values
back to the same state object.
"""
from typing import Any

from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import (
    SEQUENTIAL_MODEL_TYPES,
    collect_flat_atomic_chain_under_combined,
)
from comfy_research.engine.losses.loss_builders import TrainerTask
from comfy_research.engine.node_builder_registry import registered_trainer_model_node_types
from comfy_research.engine.trainer.dataset_helpers import (
    _DIFFUSION_NOISE_DATASET_TYPES,
    _TOKEN_CLASSIFICATION_DATASET_TYPES,
    _VECTOR_REGRESSION_DATASET_TYPES,
    _VISION_DATASET_TYPES,
    _vision_flatten_enabled,
)
from comfy_research.engine.trainer.device_runtime import (
    _resolve_trainer_compute_device,
    _trainer_force_single_thread_cpu,
)
from comfy_research.engine.trainer.graph import (
    _incoming,
    _incoming_all,
    _node_map,
    _require,
    _require_dataset,
    _require_optimizer,
)
from comfy_research.engine.trainer.loss_terms import _trainer_loss_wiring
from comfy_research.engine.trainer.prepare_state import PrepareState
from comfy_research.engine.trainer.scalar import _scalar_str
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.nodes.registry import observable_def_types
from comfy_research.schemas.graph import Node, NodeKind

_TRAINER_REGISTERED_MODEL_TYPES = frozenset(
    NodeKind(node_type) for node_type in registered_trainer_model_node_types()
)
_TRAINER_MODEL_TYPE_LABEL = ", ".join(
    node_type.value for node_type in sorted(_TRAINER_REGISTERED_MODEL_TYPES, key=lambda t: t.value)
)

_PAPER_CLASSIFICATION_MSE_MODES = frozenset(
    {"rank_figure5_mse", "rank_orthogonal_scaleup_mse", "rank_mse_control"}
)


def _paper_classification_uses_mse(node: Node) -> bool:
    """Return whether a paper-classification graph uses its MSE control protocol."""
    mode = _scalar_str((node.data or {}).get("experimentMode"), "rank_collapse_skewed")
    return mode in _PAPER_CLASSIFICATION_MSE_MODES

# Hoisted verbatim from validate_compatibility_stage : the trainer's
# Observable allowlist exposed to sync guards via ``allowed_observable_kinds()``.
_ALLOWED_OBSERVABLE_KINDS: frozenset[str] = frozenset(
    {
        "kan_reg",
    }
)


# NodeDef 通道并集
_ALLOWED_WITH_DEFS: frozenset[str] = _ALLOWED_OBSERVABLE_KINDS | observable_def_types()


def allowed_observable_kinds() -> frozenset[str]:
    """Runtime source of truth for sync guards, replacing AST literal scans."""
    return _ALLOWED_WITH_DEFS


def resolve_graph_stage(s: PrepareState) -> None:
    """Reads: nodes, edges, trainer_node_id
    Writes: nodes, nmap, trainer, model_in, combined_flat_chain, model_node, is_sequential_atomic, opt_node, ds_train, ds_test_raw, legacy_optional_test_wire, loss_incoming, weight_reg_loss_nodes, l2_projection_nodes
    """
    nodes = s.nodes
    edges = s.edges
    trainer_node_id = s.trainer_node_id
    # Canvas-only decoration (e.g. graph-assist failure strike); never part of the train graph.
    nodes = [n for n in nodes if n.type != NodeKind.graph_assist_failure_overlay]
    nmap = _node_map(nodes)
    trainer = nmap.get(trainer_node_id)
    if trainer is None:
        raise HTTPException(status_code=404, detail="Trainer node not found")
    if trainer.type != "trainer":
        raise HTTPException(status_code=400, detail="Target node is not a trainer")

    td_probe: dict[str, Any] = trainer.data or {}
    compute_spec_probe = str(td_probe.get("computeDevice") or "auto").strip() or "auto"
    if _resolve_trainer_compute_device(compute_spec_probe).type == "cpu":
        # Before NumPy dataset draw + PyTorch model init: threaded BLAS otherwise reorders float sums.
        _trainer_force_single_thread_cpu()

    model_in = _incoming(edges, nmap, trainer_node_id, "model")
    if model_in is None:
        raise HTTPException(status_code=400, detail="Trainer is missing connection: model")
    if (
        model_in.type not in _TRAINER_REGISTERED_MODEL_TYPES
        and model_in.type not in SEQUENTIAL_MODEL_TYPES
        and model_in.type != NodeKind.combined_model
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"model must be one of: {_TRAINER_MODEL_TYPE_LABEL}, combined_model, or a chain of "
                "linear_layer / activation_layer / layer_norm_layer / rms_norm_layer / embedding_layer / unembedding_layer / "
                "absolute_pos_embed_layer / rotary_embed_layer / local_mixing_layer "
                "(tip wired to trainer), "
                f"got {model_in.type}"
            ),
        )
    combined_flat_chain: list[Node] | None = None
    if model_in.type == NodeKind.combined_model:
        combined_flat_chain = collect_flat_atomic_chain_under_combined(model_in, nodes, edges)
        if not combined_flat_chain or combined_flat_chain[-1].type not in SEQUENTIAL_MODEL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Trainer connected to combined_model requires a non-empty atomic layer chain "
                    "inside that wrapper (including inside nested combined models)."
                ),
            )
        model_node = combined_flat_chain[-1]
        is_sequential_atomic = True
    else:
        model_node = model_in
        is_sequential_atomic = model_node.type in SEQUENTIAL_MODEL_TYPES

    opt_node = _require_optimizer(_incoming(edges, nmap, trainer_node_id, "optimizer"), "optimizer")

    ds_unified = _incoming(edges, nmap, trainer_node_id, "dataset")
    if ds_unified is not None:
        ds_train = _require_dataset(ds_unified, "dataset")
        ds_test_raw: Node | None = ds_train
        legacy_optional_test_wire = False
    else:
        ds_train = _require_dataset(_incoming(edges, nmap, trainer_node_id, "train_dataset"), "train dataset")
        ds_test_raw = _incoming(edges, nmap, trainer_node_id, "test_dataset")
        legacy_optional_test_wire = True
    loss_incoming, weight_reg_loss_nodes, l2_projection_nodes = _trainer_loss_wiring(
        edges, nmap, trainer_node_id
    )
    s.nodes = nodes
    s.nmap = nmap
    s.trainer = trainer
    s.model_in = model_in
    s.combined_flat_chain = combined_flat_chain
    s.model_node = model_node
    s.is_sequential_atomic = is_sequential_atomic
    s.opt_node = opt_node
    s.ds_train = ds_train
    s.ds_test_raw = ds_test_raw
    s.legacy_optional_test_wire = legacy_optional_test_wire
    s.loss_incoming = loss_incoming
    s.weight_reg_loss_nodes = weight_reg_loss_nodes
    s.l2_projection_nodes = l2_projection_nodes


def determine_task_stage(s: PrepareState) -> None:
    """Reads: ds_train, is_sequential_atomic, loss_incoming, model_node
    Writes: trainer_task, loss_node
    """
    ds_train = s.ds_train
    is_sequential_atomic = s.is_sequential_atomic
    loss_incoming = s.loss_incoming
    model_node = s.model_node
    if has_capability(model_node.type, "diffusion_loss_model"):
        if loss_incoming is None:
            raise HTTPException(status_code=400, detail="Trainer is missing connection: loss")
        if loss_incoming.type != NodeKind.diffusion_mse_loss:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Diffusion models require diffusion_mse_loss "
                    "connected to the trainer loss socket."
                ),
            )
        loss_node = _require(loss_incoming, "diffusion_mse_loss", "loss")
        trainer_task: TrainerTask = "diffusion_noise"
    elif has_capability(model_node.type, "vision_model"):
        if loss_incoming is None:
            raise HTTPException(status_code=400, detail="Trainer is missing connection: loss")
        if loss_incoming.type != NodeKind.cross_entropy_loss:
            raise HTTPException(
                status_code=400,
                detail="resnet_model and vit_model require cross_entropy_loss wired to the trainer loss socket.",
            )
        loss_node = _require(loss_incoming, "cross_entropy_loss", "loss")
        trainer_task = "vision_classification"
    elif has_capability(model_node.type, "vector_model") or is_sequential_atomic:
        if ds_train.type == NodeKind.paper_classification_dataset:
            if _paper_classification_uses_mse(ds_train):
                loss_node = _require(loss_incoming, "mse_loss", "loss")
                trainer_task = "mse_regression"
            else:
                loss_node = _require(loss_incoming, "cross_entropy_loss", "loss")
                trainer_task = "cross_entropy_dense"
        elif has_capability(ds_train.type, "memorization_dataset"):
            if loss_incoming is None:
                raise HTTPException(status_code=400, detail="Trainer is missing connection: loss")
            if loss_incoming.type in (
                NodeKind.cross_entropy_loss,
                NodeKind.binary_cross_entropy_with_logits_loss,
            ):
                loss_node = loss_incoming
                trainer_task = "cross_entropy_dense"
            elif loss_incoming.type == NodeKind.mse_loss:
                if ds_train.type == NodeKind.memorization_a_dataset:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "memorization_a_dataset is a random-label classification benchmark: class labels are "
                            "sampled independently of the input. Use cross_entropy_loss and set model outputDim to the "
                            "number of classes (dataset outputDim, at least 2). MSE loss is not supported for this "
                            "dataset type; use linear_dataset or random_noise_dataset for regression with MSE."
                        ),
                    )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "memorization_b_dataset only supports cross_entropy_loss "
                        "(inputs and targets are discrete class labels)."
                    ),
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "memorization_a_dataset / memorization_b_dataset require a classification loss "
                        f"(random-label classification), got {loss_incoming.type}"
                    ),
                )
        elif ds_train.type in _VISION_DATASET_TYPES and _vision_flatten_enabled(ds_train.data or {}):
            if loss_incoming is None:
                raise HTTPException(status_code=400, detail="Trainer is missing connection: loss")
            if loss_incoming.type != NodeKind.cross_entropy_loss:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Flattened vision datasets require cross_entropy_loss with mlp_model, gated_mlp_model, or "
                        "moe_mlp_model (set inputDim = C·H·W, outputDim = number of classes)."
                    ),
                )
            if not has_capability(model_node.type, "mlp_family"):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Flattened vision inputs are supported only for mlp_model, gated_mlp_model, or moe_mlp_model; "
                        f"got {model_node.type}."
                    ),
                )
            loss_node = _require(loss_incoming, "cross_entropy_loss", "loss")
            trainer_task = "cross_entropy_dense"
        else:
            loss_node = _require(loss_incoming, "mse_loss", "loss")
            trainer_task = "mse_regression"
    else:
        if loss_incoming is None:
            raise HTTPException(status_code=400, detail="Trainer is missing connection: loss")
        if loss_incoming.type != NodeKind.cross_entropy_loss:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Token-classification models require cross_entropy_loss. "
                    f"Model node is NodeKind.{model_node.type.value}, "
                    f"but loss is NodeKind.{loss_incoming.type.value}. "
                    "Use cross_entropy_loss for token training, or switch to a vector-output model "
                    "(mlp_model / gated_mlp_model / moe_mlp_model / kan_model / numeric_transformer_model / numeric_hyena_model / "
                    "mpp_spatiotemporal_model / afno_lite_spatiotemporal_model) for mse_loss."
                ),
            )
        loss_node = loss_incoming
        trainer_task = "token_classification"
    s.trainer_task = trainer_task
    s.loss_node = loss_node


def validate_compatibility_stage(s: PrepareState) -> None:
    """Reads: ds_test_raw, ds_train, edges, is_sequential_atomic, model_node, nmap, trainer_node_id, trainer_task
    Writes: observable_nodes, want_kan_reg
    """
    ds_test_raw = s.ds_test_raw
    ds_train = s.ds_train
    edges = s.edges
    is_sequential_atomic = s.is_sequential_atomic
    model_node = s.model_node
    nmap = s.nmap
    trainer_node_id = s.trainer_node_id
    trainer_task = s.trainer_task
    if trainer_task == "vision_classification" and ds_train.type in _VISION_DATASET_TYPES and _vision_flatten_enabled(
        ds_train.data or {}
    ):
        raise HTTPException(
            status_code=400,
            detail="Turn off 'flatten to vector' on the vision dataset when using resnet_model or vit_model.",
        )

    if ds_test_raw is not None and ds_test_raw.type != ds_train.type:
        raise HTTPException(
            status_code=400,
            detail="test dataset must be the same node type as the train dataset.",
        )
    if trainer_task == "diffusion_noise" and model_node.type == NodeKind.unet_ddpm_model:
        if ds_train.type != NodeKind.cifar10_dataset:
            raise HTTPException(
                status_code=400,
                detail="unet_ddpm_model currently supports cifar10_dataset only.",
            )
    elif trainer_task in ("mse_regression", "diffusion_noise"):
        if ds_train.type not in _VECTOR_REGRESSION_DATASET_TYPES and not (
            ds_train.type == NodeKind.paper_classification_dataset
            and _paper_classification_uses_mse(ds_train)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Vector-output model training requires linear_dataset, random_noise_dataset, memorization_b_dataset, "
                    "symbolic_func_dataset, teacher_dataset, uniform_linear_motion_dataset, kepler_2d_dataset, diffusion_pde_dataset, "
                    "reaction_diffusion_dataset, advection_dataset, dataset_mixer, or dataset_mixer_b. "
                    "memorization_a_dataset is only for cross_entropy_loss (random discrete labels); it cannot be used with MSE."
                ),
            )
        if trainer_task == "diffusion_noise" and ds_train.type not in _DIFFUSION_NOISE_DATASET_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "diffusion_score_model supports linear_dataset, random_noise_dataset, symbolic_func_dataset, "
                    "dataset_mixer, dataset_mixer_b, or memorization_b_dataset (not teacher_dataset "
                    "or uniform_linear_motion_dataset / kepler_2d_dataset)."
                ),
            )
    elif trainer_task == "cross_entropy_dense":
        if ds_train.type in _VISION_DATASET_TYPES:
            if not _vision_flatten_enabled(ds_train.data or {}):
                raise HTTPException(
                    status_code=400,
                    detail="Vision datasets use cross_entropy_dense only when 'flatten to vector' is enabled on the dataset node.",
                )
        elif (
            ds_train.type != NodeKind.paper_classification_dataset
            and not has_capability(ds_train.type, "memorization_dataset")
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dense classification losses are only supported with "
                    "memorization-style datasets in this trainer path."
                ),
            )
    elif trainer_task == "vision_classification":
        if ds_train.type not in _VISION_DATASET_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "resnet_model / vit_model require mnist_dataset, gaussian_blob_dataset, "
                    "shape_world_dataset, or hole_counting_dataset."
                ),
            )
    elif trainer_task == "token_classification" and ds_train.type not in _TOKEN_CLASSIFICATION_DATASET_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Token-classification models require token_prediction_dataset, "
                "circle_random_walk_dataset, circular_motion_dataset, unigram_dataset, bigram_low_rank_dataset, "
                "in_context_associative_recall_dataset, modular_addition_dataset, memorization_b_dataset, "
                "or a toy language dataset node (pcfg/dyck/ngram/formal/scan/cogs/listops/tinystories/phi1_style/"
                "biography_lm/relation_tuple/synthetic_playground/multi_hop_fact_chain)."
            ),
        )

    if trainer_task == "token_classification":
        if ds_train.type == NodeKind.memorization_b_dataset and not has_capability(
            model_node.type, "mlp_token_family"
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "memorization_b_dataset token mode is currently supported with mlp_token_model, "
                    "gated_mlp_token_model, or moe_mlp_token_model."
                ),
            )
        if ds_train.type == NodeKind.circular_motion_dataset and model_node.type != NodeKind.transformer_multi_token_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "circular_motion_dataset emits token ids with shape [batch, context_length, 2] "
                    "(x and y quantization per timestep). Use transformer_multi_token_model with "
                    "tokensPerPosition 2 and matching contextLength / vocabSize."
                ),
            )
        if model_node.type == NodeKind.transformer_multi_token_model and ds_train.type != NodeKind.circular_motion_dataset:
            raise HTTPException(
                status_code=400,
                detail="transformer_multi_token_model is only supported with circular_motion_dataset.",
            )

    observable_nodes = _incoming_all(edges, nmap, trainer_node_id, "observables")
    want_kan_reg = any(on.type == NodeKind.kan_reg for on in observable_nodes)
    _allowed_obs = _ALLOWED_WITH_DEFS
    for on in observable_nodes:
        if on.type not in _allowed_obs:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported observable type connected to trainer: {on.type}",
            )
        if on.type == "observable_attention_relation_score":
            from comfy_research.engine.trainer.attention_relation_metrics import attention_relation_pairs
            from comfy_research.engine.trainer.attention_relation_dsl import (
                AttentionRelationDslError,
                compile_attention_relation_predicate,
            )

            data = on.data or {}
            attention_relation_pairs(data)
            try:
                query_predicate = compile_attention_relation_predicate(
                    data.get("queryFilter"), field="Query filter", required=False
                )
                key_predicate = compile_attention_relation_predicate(
                    data.get("keyRelation"), field="Key relation", required=True
                )
            except AttentionRelationDslError as exc:
                raise HTTPException(400, f"Attention relation score: {exc}") from exc
            if (query_predicate.uses_tokens or key_predicate.uses_tokens) and model_node.type not in {
                NodeKind.attention_only_model,
                NodeKind.transformer_token_model,
            }:
                raise HTTPException(
                    400,
                    "Attention relation score: tok() predicates are supported only for rank-2 "
                    "single-token attention models (attention_only_model or transformer_token_model).",
                )
    if trainer_task == "token_classification":
        for on in observable_nodes:
            if on.type == "observable_relu_nonlinear_count":
                raise HTTPException(
                    status_code=400,
                    detail="ReLU nonlinear observable is only supported for MLP training.",
                )
    else:
        if is_sequential_atomic or model_node.type == NodeKind.kan_model:
            for on in observable_nodes:
                if on.type == "observable_relu_nonlinear_count":
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "ReLU nonlinear count observable is only supported for mlp_model, "
                            "not for kan_model or atomic layer chains."
                        ),
                    )
        for on in observable_nodes:
            if on.type == "observable_embedding_evolution":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{on.type} is only supported for token LM training "
                        "(embedding + LM-head bundles such as attention_only_model and related token models)."
                    ),
                )
    if want_kan_reg:
        if trainer_task not in ("mse_regression",):
            raise HTTPException(
                status_code=400,
                detail="KAN reg observable is only supported for MSE regression training.",
            )
        if is_sequential_atomic or model_node.type != NodeKind.kan_model:
            raise HTTPException(
                status_code=400,
                detail="KAN reg requires a kan_model on the trainer (not MLP or atomic layer chains).",
            )

    if trainer_task == "diffusion_noise":
        for on in observable_nodes:
            if on.type == NodeKind.observable_hessian_eigenvalues:
                raise HTTPException(
                    status_code=400,
                    detail="Hessian eigenvalue observable is not supported for diffusion_score_model training.",
                )

    if trainer_task == "vision_classification":
        for on in observable_nodes:
            if on.type == "observable_relu_nonlinear_count":
                raise HTTPException(
                    status_code=400,
                    detail="ReLU nonlinear count observable is not supported for resnet_model / vit_model.",
                )
    s.observable_nodes = observable_nodes
    s.want_kan_reg = want_kan_reg
