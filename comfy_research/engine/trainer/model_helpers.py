"""Model-core extraction, batch shaping, and forward helpers (extracted from trainer_run)."""

from typing import Any

import torch
import torch.nn as nn
from fastapi import HTTPException

from comfy_research.engine.models.atomic_layer_chain import SEQUENTIAL_MODEL_TYPES
from comfy_research.engine.models.hyena_like_numeric_model import NumericHyenaModel
from comfy_research.engine.models.model_loop_expand import ModelBlockLoop
from comfy_research.engine.models.mpp_spatiotemporal_model import MppSpatiotemporalModel
from comfy_research.engine.models.multi_token_transformer_model import MultiTokenTransformerModel
from comfy_research.engine.models.numeric_transformer_model import NumericTransformerModel
from comfy_research.engine.trainer.scalar import _scalar_float, _scalar_int
from comfy_research.schemas.graph import Node, NodeKind


def _numeric_transformer_core(model: nn.Module) -> NumericTransformerModel | None:
    if isinstance(model, NumericTransformerModel):
        return model
    if isinstance(model, ModelBlockLoop) and isinstance(model.inner, NumericTransformerModel):
        return model.inner
    return None


def _multi_token_transformer_core(model: nn.Module) -> MultiTokenTransformerModel | None:
    m = model
    while isinstance(m, ModelBlockLoop):
        m = m.inner
    return m if isinstance(m, MultiTokenTransformerModel) else None


def _numeric_hyena_core(model: nn.Module) -> NumericHyenaModel | None:
    if isinstance(model, NumericHyenaModel):
        return model
    if isinstance(model, ModelBlockLoop) and isinstance(model.inner, NumericHyenaModel):
        return model.inner
    return None


def _token_lm_vocab_seq_from_model(model: nn.Module) -> tuple[int, int]:
    m = model
    while isinstance(m, ModelBlockLoop):
        m = m.inner
    return int(getattr(m, "vocab_size", 0) or 0), int(getattr(m, "context_length", 0) or 0)


def _mpp_spatiotemporal_core(model: nn.Module) -> MppSpatiotemporalModel | None:
    if isinstance(model, MppSpatiotemporalModel):
        return model
    if isinstance(model, ModelBlockLoop) and isinstance(model.inner, MppSpatiotemporalModel):
        return model.inner
    return None


def _flatten_features_for_mse(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 2:
        return t
    if t.dim() == 3:
        return t.reshape(t.shape[0], -1)
    raise HTTPException(status_code=400, detail="MSE regression expects targets rank 2 or 3.")


def _regression_batch_for_model(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Flatten [B,T,D] to [B,T*D] for flat MLPs; pass through [B,T,D] for the numeric / multi-token transformer."""
    if x.dim() == 4:
        return x
    nt = _numeric_transformer_core(model)
    if nt is not None:
        if x.dim() == 3:
            return x
        if x.dim() == 2 and int(x.shape[1]) == nt.context_length * nt.token_dim:
            return x.reshape(x.shape[0], nt.context_length, nt.token_dim)
        return x
    nh = _numeric_hyena_core(model)
    if nh is not None:
        if x.dim() == 3:
            return x
        if x.dim() == 2 and int(x.shape[1]) == nh.context_length * nh.token_dim:
            return x.reshape(x.shape[0], nh.context_length, nh.token_dim)
        return x
    mt = _multi_token_transformer_core(model)
    if mt is not None:
        if x.dim() == 3:
            if int(x.shape[1]) == mt.context_length and int(x.shape[2]) == mt.tokens_per_position:
                return x
        if x.dim() == 2 and int(x.shape[1]) == mt.context_length * mt.tokens_per_position:
            return x.reshape(x.shape[0], mt.context_length, mt.tokens_per_position)
        return x
    if x.dim() == 3:
        return x.reshape(x.shape[0], -1)
    return x


def _forward_reg(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    bx = _regression_batch_for_model(model, x)
    return model(_prepare_x_for_atomic_sequential(model, bx))


# Back-compat exports: other engine modules import these from trainer_run.
# Keep permissive so atomic chains can start/end with any atomic layer type.
_ATOMIC_CHAIN_FIRST = frozenset(SEQUENTIAL_MODEL_TYPES)
_ATOMIC_CHAIN_LAST = frozenset(SEQUENTIAL_MODEL_TYPES)


def _atomic_chain_dataset_dims(chain: list[Node]) -> dict[str, Any]:
    """Infer ``inputDim`` / ``outputDim`` from an atomic chain, allowing activation/norm endpoints."""
    if not chain:
        raise HTTPException(status_code=400, detail="Atomic model layer chain is empty.")
    inferred_input: int | None = None
    current_dim: int | None = None
    for n in chain:
        nd = n.data or {}
        t = n.type
        if t in (NodeKind.linear_layer, NodeKind.unembedding_layer):
            in_f = _scalar_int(nd.get("inFeatures"), 1)
            out_f = _scalar_int(nd.get("outFeatures"), 1)
            if current_dim is not None and current_dim != in_f:
                raise HTTPException(
                    status_code=400,
                    detail=f"Atomic chain dimension mismatch before {t}: expected {current_dim}, got {in_f}.",
                )
            if inferred_input is None:
                inferred_input = in_f
            current_dim = out_f
            continue
        if t == NodeKind.embedding_layer:
            in_cols = _scalar_int(nd.get("numIndexColumns"), 1)
            emb_dim = _scalar_int(nd.get("embeddingDim"), 1)
            if current_dim is not None and current_dim != in_cols:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Atomic chain dimension mismatch before embedding_layer: "
                        f"expected {current_dim}, got {in_cols}."
                    ),
                )
            if inferred_input is None:
                inferred_input = in_cols
            current_dim = emb_dim
            continue
        if t in (NodeKind.layer_norm_layer, NodeKind.rms_norm_layer):
            shape = _scalar_int(nd.get("normalizedShape"), 1)
            tag = "layer_norm_layer" if t == NodeKind.layer_norm_layer else "rms_norm_layer"
            if current_dim is not None and current_dim != shape:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Atomic chain dimension mismatch at {tag}: "
                        f"expected {current_dim}, got {shape}."
                    ),
                )
            if inferred_input is None:
                inferred_input = shape
            current_dim = shape
            continue
        if t == NodeKind.absolute_pos_embed_layer:
            emb_dim = _scalar_int(nd.get("embeddingDim"), 64)
            if current_dim is not None and current_dim != emb_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Atomic chain dimension mismatch at absolute_pos_embed_layer: "
                        f"expected {current_dim}, got {emb_dim}."
                    ),
                )
            if inferred_input is None:
                inferred_input = emb_dim
            current_dim = emb_dim
            continue
        if t == NodeKind.rotary_embed_layer:
            rotary_dim = _scalar_int(nd.get("rotaryDim"), 64)
            if current_dim is not None and rotary_dim > current_dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Atomic chain dimension mismatch at rotary_embed_layer: "
                        f"rotaryDim {rotary_dim} exceeds channel dim {current_dim}."
                    ),
                )
            continue
        if t == NodeKind.local_mixing_layer:
            mdim = _scalar_int(nd.get("modelDim"), 64)
            if current_dim is not None and current_dim != mdim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Atomic chain dimension mismatch at local_mixing_layer: "
                        f"upstream channel dim {current_dim} != modelDim {mdim}."
                    ),
                )
            if inferred_input is None:
                inferred_input = mdim
            current_dim = mdim
            continue
        if t == NodeKind.activation_layer:
            continue
    if inferred_input is None or current_dim is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not infer atomic chain input/output dims. Include at least one dimension-defining layer "
                "(linear_layer, unembedding_layer, embedding_layer, layer_norm_layer, rms_norm_layer, absolute_pos_embed_layer, "
                "or local_mixing_layer)."
            ),
        )
    return {"inputDim": inferred_input, "outputDim": current_dim}


def _find_first_embedding(module: nn.Module) -> nn.Embedding | None:
    if isinstance(module, nn.Embedding):
        return module
    if isinstance(module, ModelBlockLoop):
        return _find_first_embedding(module.inner)
    if isinstance(module, nn.Sequential):
        for child in module:
            found = _find_first_embedding(child)
            if found is not None:
                return found
    return None


def _prepare_x_for_atomic_sequential(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Cast / clamp index batches when an ``nn.Embedding`` appears first in the forward path."""
    emb = _find_first_embedding(model)
    if emb is None:
        return x
    ub = int(emb.num_embeddings) - 1
    return x.long().clamp(min=0, max=max(ub, 0))


def _count_nonlinear_relu_neurons(
    model: nn.Sequential,
    x: torch.Tensor,
    hidden_layer_index: int,
    depth: int,
) -> int:
    """Count post-ReLU units that are on for some samples and off for some (batch dim 0)."""
    if hidden_layer_index < 0 or hidden_layer_index >= depth:
        raise HTTPException(
            status_code=400,
            detail=f"hiddenLayerIndex must be in [0, {depth - 1}] for depth={depth}.",
        )
    relu_idx = 2 * hidden_layer_index + 1
    if relu_idx >= len(model):
        raise HTTPException(status_code=400, detail="Model has fewer layers than expected for this depth.")
    captured: list[torch.Tensor] = []

    def hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        captured.clear()
        captured.append(out)

    h = model[relu_idx].register_forward_hook(hook)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(_prepare_x_for_atomic_sequential(model, _regression_batch_for_model(model, x)))
        if not captured:
            return 0
        act = captured[0]
        on_some = (act > 0).any(dim=0)
        off_some = (act == 0).any(dim=0)
        nonlinear = on_some & off_some
        return int(nonlinear.sum().item())
    finally:
        h.remove()
        if was_training:
            model.train()


def _apply_symmetrized_mlp_init(model: nn.Module, md: dict[str, Any]) -> None:
    """Symmetrized initialization (Chizat et al. 2019, Section 3.1).

    Mirrors the second half of neurons so the network output is exactly zero at
    initialization, which is required for the rich/feature-learning regime study.
    """
    tau = max(1e-8, _scalar_float(md.get("tau"), 1.0))
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return
    W1, Wout = linears[0], linears[-1]
    m = W1.weight.shape[0]
    half = m // 2
    if half == 0:
        return
    with torch.no_grad():
        nn.init.normal_(W1.weight[:half], mean=0.0, std=tau)
        if W1.bias is not None:
            W1.bias[:half].zero_()
        W1.weight[half : half + half].copy_(W1.weight[:half])
        if W1.bias is not None:
            W1.bias[half : half + half].zero_()
        nn.init.normal_(Wout.weight[:, :half], mean=0.0, std=tau)
        Wout.weight[:, half : half + half].copy_(-Wout.weight[:, :half])
        if Wout.bias is not None:
            Wout.bias.zero_()


def _apply_mlp_output_scale(model: nn.Module, md: dict[str, Any]) -> None:
    """Scale output layer weights/bias by ``outputScale`` α (rich↔lazy regime control, Chizat 2019).

    outputScale=1 is a no-op (default). SCOPE (PR-C): only the plain
    ``nn.Sequential`` mlp_model path is supported — loop-stacked or otherwise
    wrapped models silently no-op here by design; extend deliberately if a
    wrapped-model use case appears (do NOT assume all MLP wrappers work).
    """
    alpha = _scalar_float(md.get("outputScale"), 1.0)
    if alpha == 1.0:
        return
    last: nn.Module | None = None
    if isinstance(model, nn.Sequential) and len(model) > 0:
        last = model[-1]
    if not isinstance(last, nn.Linear):
        return
    with torch.no_grad():
        last.weight.mul_(alpha)
        if last.bias is not None:
            last.bias.mul_(alpha)
