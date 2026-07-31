"""Token model provider.

Receives a ModelBuildRequest (never PrepareState) and returns a
ModelBuildResult; the token model dispatch is verbatim (one unit with its
local torch.manual_seed, freeze rule 3). Criterion is
build_criterion_stage's; the per-position-target 400 check stays.
Token materialize convention: DatasetArrays carries the token width as
input_dim (ctx_len_ds) and the vocab/class count as output_dim (vocab_ds);
the shape guards read arrays.x_np/arrays.y_np.
"""
from typing import Any

import torch
from fastapi import HTTPException

from comfy_research.engine.models.model_builders import build_model_from_type
from comfy_research.engine.trainer.provider_types import ModelBuildRequest, ModelBuildResult
from comfy_research.engine.trainer.scalar import _scalar_int, _scalar_str
from comfy_research.generated.node_capabilities import has_capability
from comfy_research.schemas.graph import NodeKind


def build_token_model(req: ModelBuildRequest) -> ModelBuildResult:
    arrays = req.arrays
    ds_train = req.ds_train
    model_node = req.model_node
    x_np = req.arrays.x_np
    y_np = req.arrays.y_np
    md_a: dict[str, Any] = model_node.data or {}
    vocab_ds = arrays.output_dim
    ctx_len_ds = arrays.input_dim

    # --- model section (local re-seed moves with the dispatch; freeze rule 3) ---
    model_seed = _scalar_int(md_a.get("seed"), 0)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    embed_dim = _scalar_int(md_a.get("embedDim"), 2)
    if has_capability(model_node.type, "mlp_token_family"):
        tie_weights = str(md_a.get("tieWeights") or "yes").strip().lower() not in {"no", "false", "0"}
        model_vocab = _scalar_int(md_a.get("vocabSize"), vocab_ds)
        tokens_per_input = _scalar_int(md_a.get("tokensPerInput"), int(x_np.shape[1]) if x_np.ndim == 2 else 1)
        if model_vocab != vocab_ds:
            raise HTTPException(
                status_code=400, detail=f"Model vocabSize ({model_vocab}) must match dataset vocabSize ({vocab_ds})."
            )
        if x_np.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"MLP_token model expects dataset tokens shaped [batch, tokens], got "
                    f"{tuple(int(x) for x in x_np.shape)}."
                ),
            )
        if tokens_per_input != int(x_np.shape[1]):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model tokensPerInput ({tokens_per_input}) must match dataset token width "
                    f"({int(x_np.shape[1])})."
                ),
            )
        depth_mt = _scalar_int(md_a.get("depth"), 2)
        width_mt = _scalar_int(md_a.get("width"), 64)
        act_mt = _scalar_str(md_a.get("activation"), "relu")
        if model_node.type == NodeKind.gated_mlp_token_model:
            model = build_model_from_type(
                model_node.type,
                {
                    **md_a,
                    "vocabSize": vocab_ds,
                    "embedDim": embed_dim,
                    "tokensPerInput": tokens_per_input,
                    "tieWeights": "yes" if tie_weights else "no",
                    "depth": depth_mt,
                    "width": width_mt,
                    "activation": act_mt,
                },
            )
        elif model_node.type == NodeKind.moe_mlp_token_model:
            num_experts_mt = _scalar_int(md_a.get("numExperts"), 4)
            model = build_model_from_type(
                model_node.type,
                {
                    **md_a,
                    "vocabSize": vocab_ds,
                    "embedDim": embed_dim,
                    "tokensPerInput": tokens_per_input,
                    "tieWeights": "yes" if tie_weights else "no",
                    "depth": depth_mt,
                    "width": width_mt,
                    "numExperts": num_experts_mt,
                    "activation": act_mt,
                },
            )
        else:
            model = build_model_from_type(
                model_node.type,
                {
                    **md_a,
                    "vocabSize": vocab_ds,
                    "embedDim": embed_dim,
                    "tokensPerInput": tokens_per_input,
                    "tieWeights": "yes" if tie_weights else "no",
                    "depth": depth_mt,
                    "width": width_mt,
                    "activation": act_mt,
                },
            )
    elif model_node.type == NodeKind.transformer_token_model:
        model_vocab = _scalar_int(md_a.get("vocabSize"), vocab_ds)
        if model_vocab != vocab_ds:
            raise HTTPException(
                status_code=400, detail=f"Model vocabSize ({model_vocab}) must match dataset vocabSize ({vocab_ds})."
            )
        ctx_len_m = _scalar_int(md_a.get("contextLength"), ctx_len_ds)
        if ctx_len_m != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail=f"Model contextLength ({ctx_len_m}) must match dataset contextLength ({ctx_len_ds}).",
            )
        model = build_model_from_type(
            model_node.type,
            {
                **md_a,
                "vocabSize": vocab_ds,
                "contextLength": ctx_len_ds,
            },
        )
    elif model_node.type == NodeKind.transformer_multi_token_model:
        model_vocab = _scalar_int(md_a.get("vocabSize"), vocab_ds)
        if model_vocab != vocab_ds:
            raise HTTPException(
                status_code=400, detail=f"Model vocabSize ({model_vocab}) must match dataset vocabSize ({vocab_ds})."
            )
        ctx_len_m = _scalar_int(md_a.get("contextLength"), ctx_len_ds)
        if ctx_len_m != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail=f"Model contextLength ({ctx_len_m}) must match dataset contextLength ({ctx_len_ds}).",
            )
        k_m = _scalar_int(md_a.get("tokensPerPosition"), 2)
        if ds_train.type == NodeKind.circular_motion_dataset and k_m != 2:
            raise HTTPException(
                status_code=400,
                detail="circular_motion_dataset requires transformer_multi_token_model tokensPerPosition 2.",
            )
        if x_np.ndim != 3 or int(x_np.shape[2]) != k_m:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Dataset train input shape {tuple(int(x) for x in x_np.shape)} does not match "
                    f"model tokensPerPosition ({k_m})."
                ),
            )
        model = build_model_from_type(
            model_node.type,
            {
                **md_a,
                "vocabSize": vocab_ds,
                "contextLength": ctx_len_ds,
            },
        )
    else:
        ctx_len_m = _scalar_int(md_a.get("contextLength"), ctx_len_ds)
        if ctx_len_m != ctx_len_ds:
            raise HTTPException(
                status_code=400,
                detail=f"Model contextLength ({ctx_len_m}) must match dataset contextLength ({ctx_len_ds}).",
            )
        if model_node.type == NodeKind.linear_attention_model:
            num_heads_m = _scalar_int(md_a.get("numHeads"), 1)
            if num_heads_m < 1 or embed_dim % num_heads_m != 0:
                raise HTTPException(
                    status_code=400,
                    detail="linear_attention_model numHeads must be >= 1 and divide embedDim.",
                )
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )
        elif model_node.type == NodeKind.diagonal_ssm_token_model:
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )
        elif model_node.type == NodeKind.rwkv_time_mix_token_model:
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )
        elif model_node.type == NodeKind.hyena_like_conv_model:
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )
        elif model_node.type == NodeKind.slot_attention_token_model:
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )
        else:
            num_heads_m = _scalar_int(md_a.get("numHeads"), 1)
            if num_heads_m < 1 or embed_dim % num_heads_m != 0:
                raise HTTPException(
                    status_code=400,
                    detail="attention_only_model numHeads must be >= 1 and divide embedDim.",
                )
            model = build_model_from_type(
                model_node.type,
                {**md_a, "vocabSize": vocab_ds, "embedDim": embed_dim, "contextLength": ctx_len_ds},
            )

    # The per-position-target guard runs here; criterion construction happens later.
    depth = 1
    if int(y_np.ndim) == 2 and (
        has_capability(model_node.type, "mlp_token_family")
        or model_node.type == NodeKind.slot_attention_token_model
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "This dataset returns per-position targets [batch, context_length]; use a sequence token model "
                "(transformer_token_model, attention_only_model, linear_attention_model, diagonal_ssm_token_model, "
                "rwkv_time_mix_token_model, hyena_like_conv_model), not mlp_token_model, gated_mlp_token_model, "
                "moe_mlp_token_model, or slot_attention_token_model."
            ),
        )
    return ModelBuildResult(model=model, depth=depth, stack_io=None)
