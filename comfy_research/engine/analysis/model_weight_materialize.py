"""Build trainable modules from the graph and export weight tensors (for weight list / selector nodes)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.models.atomic_layer_chain import (
    SEQUENTIAL_MODEL_TYPES,
    build_atomic_layer_module,
    build_sequential_from_atomic_tip,
    collect_atomic_layer_chain_front_to_back,
)
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node, build_model_from_type
from comfy_research.engine.models.model_loop_expand import expand_model_loop_stacking, resolve_loop_repeat_config
from comfy_research.engine.node_builder_registry import registered_weight_export_model_node_types
from comfy_research.engine.runs.trainer_run import (
    _ATOMIC_CHAIN_FIRST,
    _ATOMIC_CHAIN_LAST,
    _atomic_chain_dataset_dims,
    apply_parameter_tensor_payloads_from_atomic_chain,
    apply_parameter_tensor_payloads_from_node,
    _incoming,
    _node_map,
    _scalar_bool,
    _scalar_float,
    _scalar_int,
    load_model_weights_from_checkpoint_b64,
)
from comfy_research.engine.analysis.activation_collect import resolve_model_upstream_of_model_output
from comfy_research.schemas.graph import Edge, Node, NodeKind, Position

_WEIGHT_EXPORT_MODEL_TYPES = frozenset(NodeKind(node_type) for node_type in registered_weight_export_model_node_types())
_WEIGHT_EXPORT_MODEL_TYPE_LABEL = ", ".join(
    node_type.value for node_type in sorted(_WEIGHT_EXPORT_MODEL_TYPES, key=lambda t: t.value)
)


def _pick1(x: Any, default: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else default
    return x if x is not None else default


def _expand_if_looped(
    model: nn.Module,
    shell: Node,
    core: Node,
    nmap: dict[str, Node],
    in_d: int,
    out_d: int,
    *,
    rebuild_block: Callable[[], nn.Module] | None = None,
) -> nn.Module:
    ln, share = resolve_loop_repeat_config(shell, core, nmap)
    if ln < 2:
        return model
    if in_d != out_d:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model loop ×{ln} requires matching input and output dimensions "
                f"(got input_dim={in_d}, output_dim={out_d})."
            ),
        )
    rb = None if share else rebuild_block
    return expand_model_loop_stacking(model, ln, share, rebuild_block=rb)


def resolve_model_for_weight_tensors(
    nmap: dict[str, Node], edges: list[Edge], weight_node_id: str
) -> Node | None:
    """Model feeding the ``model_weight_tensors`` node."""
    for e in edges:
        if e.target != weight_node_id:
            continue
        # React Flow may omit handle id; this node only has a ``model`` target.
        th = e.targetHandle or ""
        if th not in ("model", ""):
            continue
        src = nmap.get(e.source)
        if src is None:
            continue
        if src.type in _WEIGHT_EXPORT_MODEL_TYPES:
            return src
        if src.type in SEQUENTIAL_MODEL_TYPES:
            return src
        if src.type == NodeKind.model_checkpoint:
            found = resolve_model_upstream_of_model_output(nmap, edges, src.id)
            if found is not None:
                return found
    return None


def _load_checkpoint_weights_if_any_for_node(
    model: nn.Module,
    nmap: dict[str, Node],
    edges: list[Edge],
    weight_node_id: str,
) -> bool:
    inc = _incoming(edges, nmap, weight_node_id, "model")
    if inc is None:
        return False
    if inc.type != NodeKind.model_checkpoint:
        return False
    dd: dict[str, Any] = inc.data or {}
    b64 = str(dd.get("checkpoint_b64") or "").strip()
    if not b64:
        b64 = str(dd.get("memoryCheckpoint_b64") or "").strip()
    if not b64:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model checkpoint node has no weights yet. Train or load checkpoint, then try again."
            ),
        )
    load_model_weights_from_checkpoint_b64(model, b64)
    return True


def build_model_for_weight_node(
    nodes: list[Node],
    edges: list[Edge],
    weight_node_id: str,
    *,
    include_upstream_chain: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Construct ``nn.Module`` from the connected model node only (with optional checkpoint)."""
    nmap = _node_map(nodes)
    wnode = nmap.get(weight_node_id)
    if wnode is None:
        raise HTTPException(status_code=404, detail="model_weight_tensors node not found.")
    if wnode.type != NodeKind.model_weight_tensors:
        raise HTTPException(status_code=400, detail="Target is not a model_weight_tensors node.")

    model_node = resolve_model_for_weight_tensors(nmap, edges, weight_node_id)
    if model_node is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Connect one of: {_WEIGHT_EXPORT_MODEL_TYPE_LABEL}, an atomic chain, or a checkpoint chain "
                "to the model socket."
            ),
        )

    md: dict[str, Any] = model_node.data or {}
    model_type = model_node.type

    if model_type == NodeKind.kan_model:
        input_dim = _scalar_int(md.get("inputDim"), 1)
        output_dim = _scalar_int(md.get("outputDim"), 1)
        depth = int(_pick1(md.get("depth"), 2))
        width = int(_pick1(md.get("width"), 5))
        grid = int(_pick1(md.get("grid"), 3))
        spline_k = int(_pick1(md.get("k"), 3))
        base_fun = str(_pick1(md.get("baseFun"), "silu"))
        model_seed = _scalar_int(md.get("seed"), 0)
        model = build_model_for_node(
            model_node,
            ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
        )
        model = _expand_if_looped(model, model_node, model_node, nmap, input_dim, output_dim)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {
            "model_type": model_type.value,
            "used_checkpoint": used_ckpt,
            "depth": depth,
        }
        return model, meta

    if has_capability(model_type, "mlp_family"):
        input_dim = _scalar_int(md.get("inputDim"), 1)
        output_dim = _scalar_int(md.get("outputDim"), 1)
        depth = int(_pick1(md.get("depth"), 2))
        width = int(_pick1(md.get("width"), 64))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(
            model_node,
            ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
        )
        model = _expand_if_looped(model, model_node, model_node, nmap, input_dim, output_dim)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {
            "model_type": model_type.value,
            "used_checkpoint": used_ckpt,
            "depth": depth,
        }
        return model, meta

    if has_capability(model_type, "mlp_token_family"):
        vocab_size = _scalar_int(md.get("vocabSize"), 59)
        embed_dim = _scalar_int(md.get("embedDim"), 32)
        tokens_per_input = _scalar_int(md.get("tokensPerInput"), 2)
        tie_weights = str(md.get("tieWeights") or "yes").strip().lower() not in {"no", "false", "0"}
        depth = int(_pick1(md.get("depth"), 2))
        width = int(_pick1(md.get("width"), 64))
        act_name = str(
            _pick1(
                md.get("activation"),
                "silu" if model_type == NodeKind.gated_mlp_token_model else "relu",
            )
        )
        num_experts = int(_pick1(md.get("numExperts"), 4))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_from_type(
            model_type,
            {
                **md,
                "vocabSize": vocab_size,
                "embedDim": embed_dim,
                "tokensPerInput": tokens_per_input,
                "tieWeights": "yes" if tie_weights else "no",
                "depth": depth,
                "width": width,
                "numExperts": num_experts,
                "activation": act_name,
            },
        )
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {
            "model_type": model_type.value,
            "used_checkpoint": used_ckpt,
            "depth": depth,
        }
        return model, meta

    if model_type == NodeKind.crl_residual_mlp:
        actor_depth = max(4, int(_pick1(md.get("actorDepth"), 4)))
        critic_depth = max(4, int(_pick1(md.get("criticDepth"), 4)))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {
            "model_type": model_type.value,
            "used_checkpoint": used_ckpt,
            "depth": max(actor_depth, critic_depth),
        }
        return model, meta

    if model_type in SEQUENTIAL_MODEL_TYPES:
        tip_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(tip_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(tip_seed)
        if include_upstream_chain:
            model = build_sequential_from_atomic_tip(model_node, edges, nmap)
            chain = collect_atomic_layer_chain_front_to_back(model_node, edges, nmap)
            if not chain or chain[0].type not in _ATOMIC_CHAIN_FIRST or chain[-1].type not in _ATOMIC_CHAIN_LAST:
                raise HTTPException(status_code=500, detail="Internal: atomic chain head/tail invalid for weight export.")
            md_ad = _atomic_chain_dataset_dims(chain)
            ad_in = _scalar_int(md_ad.get("inputDim"), 1)
            ad_out = _scalar_int(md_ad.get("outputDim"), 1)
            model = _expand_if_looped(
                model,
                model_node,
                model_node,
                nmap,
                ad_in,
                ad_out,
                rebuild_block=lambda: build_sequential_from_atomic_tip(model_node, edges, nmap),
            )
        else:
            model = build_atomic_layer_module(model_node)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        if include_upstream_chain:
            apply_parameter_tensor_payloads_from_atomic_chain(model, chain)
        else:
            apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {
            "model_type": f"{'atomic_chain' if include_upstream_chain else 'atomic_node'}:{model_type.value}",
            "used_checkpoint": used_ckpt,
            "depth": 1,
        }
        return model, meta

    if model_type == NodeKind.residual_ln_model:
        depth = int(_pick1(md.get("depth"), 100))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth}
        return model, meta

    if model_type == NodeKind.attention_only_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 2))
        ctx_len = int(_pick1(md.get("contextLength"), 4))
        num_heads = _scalar_int(md.get("numHeads"), 1)
        if num_heads < 1 or embed_dim % num_heads != 0:
            raise HTTPException(
                status_code=400,
                detail="attention_only_model numHeads must be >= 1 and divide embedDim.",
            )
        causal_s = str(md.get("causalAttention") or "yes").strip().lower()
        causal = causal_s not in {"no", "false", "0"}
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        qk_n = _scalar_bool(md.get("qkNorm"), False)
        at_temp = max(_scalar_float(md.get("attnTemperature"), 1.0), 1e-6)
        at_cap = max(_scalar_float(md.get("attnLogitCap"), 0.0), 0.0)
        at_do = max(0.0, min(_scalar_float(md.get("attnDropout"), 0.0), 1.0))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": 1}
        return model, meta

    if model_type == NodeKind.linear_attention_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 2))
        ctx_len = int(_pick1(md.get("contextLength"), 4))
        num_heads = _scalar_int(md.get("numHeads"), 1)
        if num_heads < 1 or embed_dim % num_heads != 0:
            raise HTTPException(
                status_code=400,
                detail="linear_attention_model numHeads must be >= 1 and divide embedDim.",
            )
        causal_s = str(md.get("causalAttention") or "yes").strip().lower()
        causal = causal_s not in {"no", "false", "0"}
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": 1}
        return model, meta

    if model_type == NodeKind.diagonal_ssm_token_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 32))
        ctx_len = int(_pick1(md.get("contextLength"), 8))
        nlayers = int(_pick1(md.get("numLayers"), 1))
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": nlayers}
        return model, meta

    if model_type == NodeKind.rwkv_time_mix_token_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 32))
        ctx_len = int(_pick1(md.get("contextLength"), 8))
        depth_rw = int(_pick1(md.get("depth"), 2))
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth_rw}
        return model, meta

    if model_type == NodeKind.hyena_like_conv_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 32))
        ctx_len = int(_pick1(md.get("contextLength"), 8))
        depth_h = int(_pick1(md.get("depth"), 2))
        kern = int(_pick1(md.get("convKernel"), 7))
        ffm = int(_pick1(md.get("ffMult"), 2))
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth_h}
        return model, meta

    if model_type == NodeKind.slot_attention_token_model:
        vocab_w = int(_pick1(md.get("vocabSize"), 100))
        embed_dim = int(_pick1(md.get("embedDim"), 32))
        ctx_len = int(_pick1(md.get("contextLength"), 8))
        nslot = int(_pick1(md.get("numSlots"), 4))
        siters = int(_pick1(md.get("slotIters"), 3))
        lm_k = _scalar_int(md.get("localMixingKernel"), 0)
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": siters}
        return model, meta

    if model_type == NodeKind.diffusion_score_model:
        data_dim = int(_pick1(md.get("inputDim"), 8))
        hid_d = int(_pick1(md.get("hiddenDim"), 128))
        dep_d = int(_pick1(md.get("depth"), 3))
        te_d = int(_pick1(md.get("timeEmbedDim"), 64))
        tmax_d = int(_pick1(md.get("diffusionTimesteps"), 100))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext(input_dim=data_dim))
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": dep_d}
        return model, meta

    if model_type == NodeKind.numeric_transformer_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        ctx, tin, tout = model.context_length, model.token_dim, model.output_token_dim
        flat_in = int(ctx * tin)
        flat_out = int(ctx * tout)
        model = _expand_if_looped(model, model_node, model_node, nmap, flat_in, flat_out)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        n_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": n_layers}
        return model, meta

    if model_type == NodeKind.numeric_hyena_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        ctx, tin, tout = model.context_length, model.token_dim, model.output_token_dim
        flat_in = int(ctx * tin)
        flat_out = int(ctx * tout)
        model = _expand_if_looped(model, model_node, model_node, nmap, flat_in, flat_out)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        depth = len(model.blocks) if hasattr(model, "blocks") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth}
        return model, meta

    if model_type == NodeKind.mpp_spatiotemporal_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        flat_io = int(model.flat_dim)
        model = _expand_if_looped(model, model_node, model_node, nmap, flat_io, flat_io)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        depth = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth}
        return model, meta

    if model_type == NodeKind.afno_lite_spatiotemporal_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        flat_io = int(model.cfg.flat_dim)
        model = _expand_if_looped(model, model_node, model_node, nmap, flat_io, flat_io)
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        depth = len(model.encoder.blocks) if hasattr(model.encoder, "blocks") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": depth}
        return model, meta

    if model_type == NodeKind.transformer_token_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        n_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": n_layers}
        return model, meta

    if model_type == NodeKind.transformer_multi_token_model:
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any_for_node(model, nmap, edges, weight_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        n_layers = len(model.encoder.layers) if hasattr(model.encoder, "layers") else 1
        meta = {"model_type": model_type.value, "used_checkpoint": used_ckpt, "depth": n_layers}
        return model, meta

    raise HTTPException(status_code=400, detail=f"Unsupported model type for weights: {model_type}")


def materialize_named_parameters_payload(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Flatten each parameter to a numeric list + shape for the canvas."""
    out: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            arr = p.detach().cpu().float().numpy()
            flat = arr.reshape(-1).astype(np.float64, copy=False)
            # Plain Python ints — JSON encoders reject numpy scalar types in ``shape``.
            shape_py = [int(x) for x in arr.shape]
            out[name] = {"shape": shape_py, "values": flat.tolist()}
    return out


def run_collect_model_weights(
    nodes: list[Node],
    edges: list[Edge],
    weight_node_id: str,
    *,
    include_upstream_chain: bool = True,
) -> dict[str, Any]:
    model, meta = build_model_for_weight_node(
        nodes,
        edges,
        weight_node_id,
        include_upstream_chain=include_upstream_chain,
    )
    model.eval()
    weights = materialize_named_parameters_payload(model)
    names = sorted(weights.keys())
    summary = (
        f"Materialized {len(names)} parameter tensor(s) from {meta.get('model_type', '?')}"
        + (" (checkpoint)" if meta.get("used_checkpoint") else "")
    )
    return {"weights": weights, "summary": summary, "meta": meta}


def _synthetic_weight_node_id(model_node_id: str) -> str:
    return f"__obs_specs__{model_node_id}"


def run_model_weight_specs(
    nodes: list[Node],
    edges: list[Edge],
    model_node_id: str,
    *,
    include_upstream_chain: bool = True,
) -> dict[str, Any]:
    """Parameter names + shapes for a canvas model node (no numeric payload)."""
    mid = model_node_id.strip()
    if not mid:
        raise HTTPException(status_code=400, detail="model_node_id is required.")
    nmap = _node_map(nodes)
    if mid not in nmap:
        raise HTTPException(status_code=404, detail="Model node not found.")

    synth_id = _synthetic_weight_node_id(mid)
    synth_nodes = list(nodes)
    synth_edges = list(edges)
    if synth_id not in nmap:
        src = nmap[mid]
        synth_nodes.append(
            Node(
                id=synth_id,
                type=NodeKind.model_weight_tensors,
                data={},
                position=Position(x=float(src.position.x), y=float(src.position.y)),
            )
        )
        synth_edges.append(
            Edge(
                id=f"e-{mid}-{synth_id}",
                source=mid,
                target=synth_id,
                sourceHandle="model",
                targetHandle="model",
            )
        )

    model, meta = build_model_for_weight_node(
        synth_nodes,
        synth_edges,
        synth_id,
        include_upstream_chain=include_upstream_chain,
    )
    model.eval()
    specs: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            specs[name] = {"shape": [int(x) for x in p.shape]}
    names = sorted(specs.keys())
    summary = (
        f"Listed {len(names)} parameter tensor(s) from {meta.get('model_type', '?')}"
        + (" (checkpoint)" if meta.get("used_checkpoint") else "")
    )
    return {"specs": specs, "summary": summary, "meta": meta}
