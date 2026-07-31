from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.generated.node_capabilities import has_capability
from comfy_research.engine.models.atomic_layer_chain import (
    SEQUENTIAL_MODEL_TYPES,
    build_sequential_from_atomic_tip,
    build_sequential_from_flat_atomic_chain,
    collect_atomic_layer_chain_front_to_back,
    collect_flat_atomic_chain_under_combined,
)
from comfy_research.engine.datasets.random_input_distribution_runtime import sample_inputs
from comfy_research.engine.runs.trainer_run import (
    _ATOMIC_CHAIN_FIRST,
    _ATOMIC_CHAIN_LAST,
    _apply_linear_map,
    _atomic_chain_dataset_dims,
    apply_parameter_tensor_payloads_from_atomic_chain,
    apply_parameter_tensor_payloads_from_node,
    _memorization_b_xy_numpy,
    _memorization_b_vocab_size,
    _node_map,
    _prepare_x_for_atomic_sequential,
    _random_linear_weights,
    _resolve_linear_dataset_mlp_dims,
    _sample_teacher_input_tensor,
    _scalar_float,
    _scalar_int,
    load_model_weights_from_checkpoint_b64,
)
from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_for_node
from comfy_research.engine.node_builder_registry import registered_activation_model_node_types
from comfy_research.engine.models.residual_ln_model import (
    compute_residual_stream_tensors,
    residual_representation_ids,
)
from comfy_research.engine.analysis.activation_run_store import manifest_for_run, store_activation_tensors
from comfy_research.engine.models.model_loop_expand import resolve_loop_repeat_config
from comfy_research.schemas.graph import Edge, Node, NodeKind

_ACTIVATION_MODEL_TYPES = frozenset(NodeKind(node_type) for node_type in registered_activation_model_node_types())


def _incoming(
    edges: list[Edge], nodes: dict[str, Node], target_id: str, handle: str
) -> Node | None:
    for e in edges:
        if e.target == target_id and e.targetHandle == handle:
            return nodes.get(e.source)
    return None


def _pick1(x: Any, default: Any) -> Any:
    if isinstance(x, list):
        return x[0] if x else default
    return x if x is not None else default


def resolve_model_from_trainer(nmap: dict[str, Node], edges: list[Edge], trainer_id: str) -> Node | None:
    for e2 in edges:
        if e2.target != trainer_id or e2.targetHandle != "model":
            continue
        mlp = nmap.get(e2.source)
        if mlp is None:
            continue
        if mlp.type in _ACTIVATION_MODEL_TYPES:
            return mlp
        if mlp.type == NodeKind.combined_model:
            return mlp
        if mlp.type in SEQUENTIAL_MODEL_TYPES:
            return mlp
        if mlp.type == NodeKind.model_checkpoint:
            return resolve_model_upstream_of_model_output(nmap, edges, mlp.id)
    return None


def resolve_model_upstream_of_model_output(
    nmap: dict[str, Node], edges: list[Edge], checkpoint_id: str
) -> Node | None:
    for e in edges:
        if e.target != checkpoint_id or e.targetHandle != "model_checkpoint":
            continue
        tr = nmap.get(e.source)
        if tr is None or not has_capability(tr.type, "trainer_runner"):
            continue
        if e.sourceHandle is not None and e.sourceHandle != "checkpoint":
            continue
        found = resolve_model_from_trainer(nmap, edges, tr.id)
        if found is not None:
            return found
    return None


def resolve_model_for_activation(
    nmap: dict[str, Node], edges: list[Edge], activation_id: str
) -> Node | None:
    for e in edges:
        if e.target != activation_id or e.targetHandle != "model":
            continue
        src = nmap.get(e.source)
        if src is None:
            continue
        if src.type in _ACTIVATION_MODEL_TYPES:
            return src
        if src.type == NodeKind.combined_model:
            return src
        if src.type in SEQUENTIAL_MODEL_TYPES:
            return src
        if src.type == NodeKind.trainer:
            if e.sourceHandle is not None and e.sourceHandle != "checkpoint":
                continue
            found = resolve_model_from_trainer(nmap, edges, src.id)
            if found is not None:
                return found
        if src.type == NodeKind.model_checkpoint:
            found = resolve_model_upstream_of_model_output(nmap, edges, src.id)
            if found is not None:
                return found
    return None


def _all_representation_ids(depth: int) -> set[str]:
    ids: set[str] = {"input", "output"}
    for i in range(1, depth + 1):
        ids.add(f"h{i}_preact")
        ids.add(f"h{i}_postact")
    return ids


def _all_representation_ids_for_model(model_type: NodeKind, depth: int) -> set[str]:
    if model_type == NodeKind.residual_ln_model:
        return residual_representation_ids(depth)
    return _all_representation_ids(depth)


def _compute_activation_tensors(
    model: nn.Sequential, x: torch.Tensor, depth: int
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {"input": x}
    h = x
    for i in range(depth):
        li = 2 * i
        h = model[li](h)
        out[f"h{i + 1}_preact"] = h
        h = model[li + 1](h)
        out[f"h{i + 1}_postact"] = h
    h = model[2 * depth](h)
    out["output"] = h
    return out


def _load_checkpoint_weights_if_any(
    model: nn.Module,
    nmap: dict[str, Node],
    edges: list[Edge],
    activation_node_id: str,
) -> bool:
    """Load weights when the activation ``model`` input is a checkpoint node or a trainer ``checkpoint`` handle.

    Returns True if weights were loaded from a checkpoint, False if the model input is a bare ``mlp_model``.
    """
    inc = _incoming(edges, nmap, activation_node_id, "model")
    if inc is None:
        return False
    if inc.type == NodeKind.model_checkpoint:
        dd: dict[str, Any] = inc.data or {}
        b64 = str(dd.get("checkpoint_b64") or "").strip()
        if not b64:
            b64 = str(dd.get("memoryCheckpoint_b64") or "").strip()
        if not b64:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation is wired through a Model checkpoint node that has no weights yet. "
                    "Train (or use Load from memory / file on the checkpoint node), then try again."
                ),
            )
        load_model_weights_from_checkpoint_b64(model, b64)
        return True
    if inc.type == NodeKind.trainer:
        td: dict[str, Any] = inc.data or {}
        b64 = str(td.get("memoryCheckpoint_b64") or td.get("checkpoint_b64") or "").strip()
        if not b64:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation is wired from the trainer's model-checkpoint output, but this trainer "
                    "has no in-memory checkpoint yet. Run training (or pause) once, or use a Model checkpoint node."
                ),
            )
        load_model_weights_from_checkpoint_b64(model, b64)
        return True
    return False


def _normalize_wire_picks(ad: dict[str, Any]) -> list[dict[str, Any]]:
    raw = ad.get("activationWirePicks") or ad.get("activation_wire_picks") or []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        tk = str(p.get("tensorKey") or p.get("tensor_key") or "").strip()
        idx_raw = p.get("afterModuleIndex", p.get("after_module_index"))
        try:
            idx = int(idx_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not tk:
            continue
        out.append({"tensorKey": tk, "afterModuleIndex": idx})
    return out


def _sanitize_wire_tensor_key(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:120] if s else "activation")


def _collect_sequential_wire_tensors(
    model: nn.Sequential,
    x: torch.Tensor,
    picks: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    modules = list(model.children())
    n_mod = len(modules)
    if n_mod == 0:
        raise HTTPException(status_code=400, detail="Model has no submodules to hook.")
    out: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(key: str):
        def _hook(_m: nn.Module, _inp: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                out[key] = output.detach().clone()
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                out[key] = output[0].detach().clone()

        return _hook

    try:
        for p in picks:
            key = _sanitize_wire_tensor_key(str(p.get("tensorKey") or ""))
            idx = int(p.get("afterModuleIndex", -1))
            if idx < 0 or idx >= n_mod:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Wire pick afterModuleIndex {idx} is out of range for this model (0..{n_mod - 1}). "
                        "Re-open the wire picker after changing depth or architecture."
                    ),
                )
            h = modules[idx].register_forward_hook(make_hook(key))
            handles.append(h)
        with torch.no_grad():
            x_in = _prepare_x_for_atomic_sequential(model, x)
            _ = model(x_in)
    finally:
        for h in handles:
            h.remove()
    missing: list[str] = []
    for p in picks:
        k = _sanitize_wire_tensor_key(str(p.get("tensorKey") or ""))
        if k not in out:
            missing.append(k)
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Wire hooks did not capture tensors for keys: {missing!r}.",
        )
    return out


def activation_forward_tensors(
    nodes: list[Node],
    edges: list[Edge],
    activation_node_id: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Forward-pass the MLP used by an activation node (with optional checkpoint weights) on its dataset.

    Returns ``(tensors_by_rep_id, meta)`` where ``meta`` includes ``used_checkpoint: bool`` for summaries.
    """
    nmap = _node_map(nodes)
    act_node = nmap.get(activation_node_id)
    if act_node is None:
        raise HTTPException(status_code=404, detail="Activation node not found")
    if act_node.type != NodeKind.activation:
        raise HTTPException(status_code=400, detail="Target node is not an activation node")

    model_node = resolve_model_for_activation(nmap, edges, activation_node_id)
    if model_node is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Connect a model (MLP, residual LN model, trainer checkpoint, or model-checkpoint chain) "
                "to the model socket."
            ),
        )

    ds_node = _incoming(edges, nmap, activation_node_id, "dataset")
    if ds_node is None:
        raise HTTPException(
            status_code=400,
            detail="Connect a dataset to the activation dataset socket.",
        )
    if not has_capability(ds_node.type, "activation_sample_dataset"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Activation dataset must be linear_dataset, memorization_a_dataset, memorization_b_dataset, "
                "symbolic_func_dataset, or teacher_dataset, "
                f"got {ds_node.type}"
            ),
        )

    dd_train: dict[str, Any] = ds_node.data or {}

    md: dict[str, Any] = model_node.data or {}
    model_type = model_node.type
    train_size = int(dd_train.get("trainSize", 800))
    if train_size < 1:
        raise HTTPException(status_code=400, detail="train size must be >= 1")
    seed = _scalar_int(dd_train.get("seed"), 0)
    ad = act_node.data or {}
    wire_picks = _normalize_wire_picks(ad)

    if model_type == NodeKind.crl_residual_mlp:
        if ds_node.type != NodeKind.linear_dataset:
            raise HTTPException(
                status_code=400,
                detail="crl_residual_mlp activation collection requires linear_dataset (for trainSize/seed only).",
            )
        if wire_picks:
            raise HTTPException(
                status_code=400,
                detail="Wire pick activation is not supported for crl_residual_mlp; use representation checkboxes.",
            )
        state_dim = _scalar_int(md.get("stateDim"), 4)
        action_dim = _scalar_int(md.get("actionDim"), 2)
        goal_dim = _scalar_int(md.get("goalDim"), 2)
        obs_dim_full = state_dim + goal_dim
        actor_depth = max(4, int(_pick1(md.get("actorDepth"), 4)))
        critic_depth = max(4, int(_pick1(md.get("criticDepth"), 4)))
        model_seed = _scalar_int(md.get("seed"), 0)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        model = build_model_for_node(model_node, ModelBuildContext())
        used_ckpt = _load_checkpoint_weights_if_any(model, nmap, edges, activation_node_id)
        apply_parameter_tensor_payloads_from_node(model, model_node)
        model.eval()
        rng = np.random.default_rng(seed)
        obs_full = torch.from_numpy(rng.standard_normal((train_size, obs_dim_full)).astype(np.float32))
        s = obs_full[:, :state_dim]
        goal = obs_full[:, state_dim : state_dim + goal_dim]
        a = torch.from_numpy(rng.standard_normal((train_size, action_dim)).astype(np.float32))
        with torch.no_grad():
            phi = model.sa_encoder(s, a)
            psi = model.g_encoder(goal)
            mean, log_std = model.actor(obs_full)
        tensors = {"phi": phi, "psi": psi, "actor_mean": mean, "actor_log_std": log_std}
        meta = {
            "used_checkpoint": used_ckpt,
            "train_size": train_size,
            "input_dim": obs_dim_full,
            "depth": max(actor_depth, critic_depth),
            "model_type": model_type.value,
        }
        return tensors, meta

    if wire_picks:
        if model_type == NodeKind.residual_ln_model:
            raise HTTPException(
                status_code=400,
                detail="Wire pick activation is not supported for residual_ln_model; use representation checkboxes instead.",
            )
        if model_type == NodeKind.kan_model:
            raise HTTPException(
                status_code=400,
                detail="Wire pick activation is not supported for kan_model.",
            )
        if model_type == NodeKind.gated_mlp_model:
            raise HTTPException(
                status_code=400,
                detail="Wire pick activation is not supported for gated_mlp_model.",
            )
        if model_type == NodeKind.moe_mlp_model:
            raise HTTPException(
                status_code=400,
                detail="Wire pick activation is not supported for moe_mlp_model.",
            )

    if wire_picks and model_type in (NodeKind.combined_model, *tuple(SEQUENTIAL_MODEL_TYPES)):
        if ds_node.type == NodeKind.teacher_dataset:
            raise HTTPException(
                status_code=400,
                detail="Wire picks for combined / atomic-chain models require linear_dataset or symbolic_func_dataset.",
            )
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tip: Node
        if model_type == NodeKind.combined_model:
            chain = collect_flat_atomic_chain_under_combined(model_node, nodes, edges)
            if not chain or chain[-1].type not in SEQUENTIAL_MODEL_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail="combined_model must contain a non-empty atomic tensor chain for wire activation collection.",
                )
            tip = chain[-1]
        else:
            tip = model_node
            chain = collect_atomic_layer_chain_front_to_back(tip, edges, nmap)
        if not chain:
            raise HTTPException(status_code=400, detail="Atomic model chain is empty.")
        if chain[0].type not in _ATOMIC_CHAIN_FIRST or chain[-1].type not in _ATOMIC_CHAIN_LAST:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Atomic model chain must start with linear_layer or embedding_layer and end with "
                    "linear_layer or unembedding_layer (absolute_pos_embed_layer, rotary_embed_layer, and "
                    "local_mixing_layer are allowed between)."
                ),
            )
        md_atomic = _atomic_chain_dataset_dims(chain)
        input_dim = int(md_atomic["inputDim"])
        output_dim = int(md_atomic["outputDim"])
        input_dist = str(dd_train.get("inputDistribution", "standard_normal"))
        if ds_node.type == NodeKind.memorization_b_dataset:
            out_dist = str(dd_train.get("outputDistribution", "uniform_class_probs"))
            mem_alpha = _scalar_float(dd_train.get("alpha"), 1.0)
            vocab = _memorization_b_vocab_size(dd_train)
            x_np, _ = _memorization_b_xy_numpy(rng, train_size, vocab, out_dist, mem_alpha)
        else:
            x_np = sample_inputs(rng, train_size, input_dim, input_dist)
            if ds_node.type in (NodeKind.linear_dataset, NodeKind.memorization_a_dataset):
                out_dist = str(dd_train.get("outputDistribution", "additive_gaussian"))
                noise_level = float(dd_train.get("noiseLevel", 0.25))
                additive = out_dist == "additive_gaussian"
                sigma = noise_level if additive else 0.0
                w = _random_linear_weights(rng, input_dim, output_dim)
                _ = _apply_linear_map(x_np, w, sigma, additive, rng)
        x_t = torch.from_numpy(x_np)
        tip_data = tip.data or {}
        model_seed = _scalar_int(tip_data.get("seed"), 0) if "seed" in tip_data else seed
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)
        if model_type == NodeKind.combined_model:
            model = build_sequential_from_flat_atomic_chain(chain)
        else:
            model = build_sequential_from_atomic_tip(tip, edges, nmap)
        loop_n, _ = resolve_loop_repeat_config(model_node, tip, nmap)
        if loop_n >= 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation wire picks are not supported when the model shell loop count is 2 or more. "
                    "Set loop to 1 on the Loop dialog, or remove wire picks."
                ),
            )
        used_ckpt = _load_checkpoint_weights_if_any(model, nmap, edges, activation_node_id)
        apply_parameter_tensor_payloads_from_atomic_chain(model, chain)
        model.eval()
        tensors = _collect_sequential_wire_tensors(model, x_t, wire_picks)
        depth = len(chain)
        meta = {
            "used_checkpoint": used_ckpt,
            "train_size": train_size,
            "input_dim": input_dim,
            "depth": depth,
            "model_type": model_type.value,
            "output_dim": output_dim,
        }
        return tensors, meta

    if ds_node.type == NodeKind.teacher_dataset:
        if not has_capability(model_type, "mlp_family"):
            raise HTTPException(
                status_code=400,
                detail="teacher_dataset requires an MLP, gated MLP, or MoE MLP on the activation model socket.",
            )
        input_dim = int(_pick1(md.get("inputDim"), 10))
        output_dim = int(_pick1(md.get("outputDim"), 1))
        x_np, _ = _sample_teacher_input_tensor(edges, nmap, ds_node, "train_input", None)
        if x_np.shape[1] != input_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Teacher train input dim ({x_np.shape[1]}) must match the MLP input dim ({input_dim})."
                ),
            )
        depth = int(_pick1(md.get("depth"), 2))
        width = int(_pick1(md.get("width"), 64))
        act_name = str(_pick1(md.get("activation"), "relu"))
    elif has_capability(model_type, "mlp_family"):
        input_dim, output_dim = _resolve_linear_dataset_mlp_dims(
            dd_train, md, ds_node.type, model_kind=model_type
        )
        depth = int(_pick1(md.get("depth"), 2))
        width = int(_pick1(md.get("width"), 64))
        act_name = str(
            _pick1(
                md.get("activation"),
                "silu" if has_capability(model_type, "silu_default_activation_model") else "relu",
            )
        )
        num_experts = int(_pick1(md.get("numExperts"), 4))
    elif model_type == NodeKind.kan_model:
        raise HTTPException(
            status_code=400,
            detail="Activation collection is not supported for kan_model; use mlp_model or residual_ln_model.",
        )
    elif model_type == NodeKind.residual_ln_model:
        depth = int(_pick1(md.get("depth"), 100))
        input_dim = int(_pick1(md.get("dim"), 256))
        output_dim = input_dim
        width = input_dim
        act_name = str(_pick1(md.get("activation"), "relu"))
    elif model_type in (NodeKind.combined_model, *tuple(SEQUENTIAL_MODEL_TYPES)):
        raise HTTPException(
            status_code=400,
            detail=(
                "This model type on the activation socket needs wire picks: open Pick wires (preview) "
                "on the Activation node, then Collect."
            ),
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type for activation collection: {model_type}",
        )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if ds_node.type != NodeKind.teacher_dataset:
        input_dist = str(dd_train.get("inputDistribution", "standard_normal"))
        if ds_node.type == NodeKind.memorization_b_dataset:
            out_dist = str(dd_train.get("outputDistribution", "uniform_class_probs"))
            mem_alpha = _scalar_float(dd_train.get("alpha"), 1.0)
            vocab = _memorization_b_vocab_size(dd_train)
            x_np, _ = _memorization_b_xy_numpy(rng, train_size, vocab, out_dist, mem_alpha)
        else:
            x_np = sample_inputs(rng, train_size, input_dim, input_dist)
            if ds_node.type in (NodeKind.linear_dataset, NodeKind.memorization_a_dataset):
                out_dist = str(dd_train.get("outputDistribution", "additive_gaussian"))
                noise_level = float(dd_train.get("noiseLevel", 0.25))
                additive = out_dist == "additive_gaussian"
                sigma = noise_level if additive else 0.0
                w = _random_linear_weights(rng, input_dim, output_dim)
                _ = _apply_linear_map(x_np, w, sigma, additive, rng)

    x_t = torch.from_numpy(x_np)
    model_seed = _scalar_int(md.get("seed"), 0) if "seed" in md else seed
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    if has_capability(model_type, "mlp_family"):
        model = build_model_for_node(
            model_node,
            ModelBuildContext(input_dim=input_dim, output_dim=output_dim),
        )
        ln_mlp, _ = resolve_loop_repeat_config(model_node, model_node, nmap)
        if ln_mlp >= 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation collection with checkbox representations does not support loop count > 1. "
                    "Set loop to 1, or adjust the graph."
                ),
            )
    else:
        model = build_model_for_node(model_node, ModelBuildContext())
    used_ckpt = _load_checkpoint_weights_if_any(model, nmap, edges, activation_node_id)
    apply_parameter_tensor_payloads_from_node(model, model_node)
    model.eval()

    with torch.no_grad():
        if wire_picks and model_type == NodeKind.mlp_model:
            if not isinstance(model, nn.Sequential):
                raise HTTPException(status_code=500, detail="Internal: expected nn.Sequential for MLP wire picks.")
            tensors = _collect_sequential_wire_tensors(model, x_t, wire_picks)
        elif model_type == NodeKind.mlp_model:
            tensors = _compute_activation_tensors(model, x_t, depth)
        elif model_type == NodeKind.gated_mlp_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation collection by predefined representations is not supported for gated_mlp_model yet. "
                    "Use mlp_model or residual_ln_model."
                ),
            )
        elif model_type == NodeKind.moe_mlp_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Activation collection by predefined representations is not supported for moe_mlp_model yet. "
                    "Use mlp_model or residual_ln_model."
                ),
            )
        else:
            tensors = compute_residual_stream_tensors(model, x_t)

    meta = {
        "used_checkpoint": used_ckpt,
        "train_size": train_size,
        "input_dim": input_dim,
        "depth": depth,
        "model_type": model_type.value,
    }
    return tensors, meta


def run_collect_activations(
    nodes: list[Node],
    edges: list[Edge],
    activation_node_id: str,
) -> dict[str, Any]:
    nmap = _node_map(nodes)
    act_node = nmap.get(activation_node_id)
    if act_node is None:
        raise HTTPException(status_code=404, detail="Activation node not found")
    if act_node.type != NodeKind.activation:
        raise HTTPException(status_code=400, detail="Target node is not an activation node")

    ad: dict[str, Any] = act_node.data or {}
    wire_picks = _normalize_wire_picks(ad)
    if wire_picks:
        tensors, meta = activation_forward_tensors(nodes, edges, activation_node_id)
        train_size = int(meta["train_size"])
        input_dim = int(meta["input_dim"])
        used_ckpt = bool(meta["used_checkpoint"])
        run_id = store_activation_tensors(tensors)
        manifest = manifest_for_run(run_id)
        if manifest is None:
            raise HTTPException(status_code=500, detail="Failed to store activation tensors.")
        ckpt_part = " using weights from a checkpoint (Model checkpoint node or trainer output)." if used_ckpt else ""
        summary = (
            f"Collected {len(wire_picks)} wire activation(s) on train set "
            f"({train_size} samples, input dim {input_dim}){ckpt_part}"
            " Tensors stay on the server; downstream nodes fetch on demand."
        )
        return {
            "activation_run_id": run_id,
            "manifest": manifest,
            "summary": summary,
        }

    selected: list[str] = list(ad.get("selectedRepresentationIds") or [])
    if not selected:
        raise HTTPException(
            status_code=400,
            detail="Add wire picks (Pick wires preview) or select at least one hidden representation (checkbox) to collect.",
        )

    tensors, meta = activation_forward_tensors(nodes, edges, activation_node_id)
    depth = int(meta["depth"])
    model_type_raw = str(meta.get("model_type", NodeKind.mlp_model))
    model_type = NodeKind(model_type_raw)
    allowed = _all_representation_ids_for_model(model_type, depth)
    unknown = [s for s in selected if s not in allowed]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                "Some selected representations do not match the current model depth; "
                f"reconnect the model or adjust its depth. Unknown: {unknown!r}"
            ),
        )
    train_size = int(meta["train_size"])
    input_dim = int(meta["input_dim"])
    used_ckpt = bool(meta["used_checkpoint"])

    selected_torch = {rep_id: tensors[rep_id] for rep_id in selected}
    run_id = store_activation_tensors(selected_torch)
    manifest = manifest_for_run(run_id)
    if manifest is None:
        raise HTTPException(status_code=500, detail="Failed to store activation tensors.")

    ckpt_part = " using weights from a checkpoint (Model checkpoint node or trainer output)." if used_ckpt else ""
    summary = (
        f"Collected {len(selected)} representation(s) on train set "
        f"({train_size} samples, input dim {input_dim}){ckpt_part}"
        + " Tensors stay on the server; downstream nodes fetch on demand."
    )
    return {
        "activation_run_id": run_id,
        "manifest": manifest,
        "summary": summary,
    }
