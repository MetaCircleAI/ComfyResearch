from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException

from comfy_research.nodes.registry import model_defs_builders
from comfy_research.engine.models.afno_lite_spatiotemporal_model import afno_lite_spatiotemporal_from_canvas_md
from comfy_research.engine.models.attention_only_model import AttentionTokenPredictBundle
from comfy_research.engine.models.cifar_models import build_small_inception_cifar, build_vgg11_cifar
from comfy_research.engine.crl.crl_networks import CrlResidualAgent, parse_crl_residual_activation
from comfy_research.engine.models.diffusion_score_model import DiffusionScoreMLP
from comfy_research.engine.models.unet_ddpm_model import build_unet_ddpm_from_md
from comfy_research.engine.models.diagonal_ssm_token_model import DiagonalSsmTokenPredictBundle
from comfy_research.engine.models.hyena_like_numeric_model import numeric_hyena_from_canvas_md
from comfy_research.engine.models.hyena_like_conv_model import HyenaLikeConvTokenPredictBundle
from comfy_research.engine.models.kan_model_build import build_kan_regression
from comfy_research.engine.models.keskar_cnn import build_keskar_from_md
from comfy_research.engine.models.linear_attention_model import LinearAttentionTokenPredictBundle
from comfy_research.engine.models.multi_token_transformer_model import multi_token_transformer_from_canvas_md
from comfy_research.engine.models.mpp_spatiotemporal_model import mpp_spatiotemporal_from_canvas_md
from comfy_research.engine.models.numeric_transformer_model import numeric_transformer_from_canvas_md
from comfy_research.engine.models.rwkv_time_mix_token_model import RwkvTimeMixTokenPredictBundle
from comfy_research.engine.models.residual_ln_model import ResidualLNModel
from comfy_research.engine.models.slot_attention_token_model import SlotAttentionTokenPredictBundle
from comfy_research.engine.models.token_transformer_model import token_transformer_from_canvas_md
from comfy_research.engine.models.vision_models import build_resnet_from_md, build_vit_from_md
from comfy_research.schemas.graph import Node, NodeKind


@dataclass(frozen=True)
class ModelBuildContext:
    input_channels: int | None = None
    image_size: int | None = None
    num_classes: int | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    kan_fast_training: bool = True


ModelBuilder = Callable[[dict[str, Any], ModelBuildContext], nn.Module]


def _require_int(value: int | None, name: str, node_type: str) -> int:
    if value is None:
        raise HTTPException(status_code=500, detail=f"Internal: {node_type} builder requires {name}.")
    return int(value)


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


def _scalar_str(x: Any, default: str = "") -> str:
    if isinstance(x, list):
        if not x:
            return default
        return str(x[0])
    return str(x) if x is not None else default


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


def _scalar_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and x in (0, 1):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "1", "on"}:
            return True
        if s in {"false", "no", "0", "off"}:
            return False
    if isinstance(x, list) and x:
        return _scalar_bool(x[0], default)
    return default


def _activation(name: str) -> nn.Module:
    m: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "identity": nn.Identity,
    }
    cls = m.get(name)
    if cls is None:
        raise HTTPException(status_code=400, detail=f"Unknown activation: {name}")
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    return cls()


def _build_mlp(
    input_dim: int,
    output_dim: int,
    depth: int,
    width: int,
    activation: str,
) -> nn.Module:
    if depth < 1 or width < 1:
        raise HTTPException(status_code=400, detail="MLP depth and width must be >= 1")
    layers: list[nn.Module] = []
    in_f = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(in_f, width))
        layers.append(_activation(activation))
        in_f = width
    layers.append(nn.Linear(in_f, output_dim))
    return nn.Sequential(*layers)


class GatedMlp(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, depth: int, width: int, activation: str) -> None:
        super().__init__()
        if depth < 1 or width < 1:
            raise HTTPException(status_code=400, detail="Gated MLP depth and width must be >= 1")
        self.gates = nn.ModuleList()
        self.values = nn.ModuleList()
        in_f = input_dim
        for _ in range(depth):
            self.gates.append(nn.Linear(in_f, width))
            self.values.append(nn.Linear(in_f, width))
            in_f = width
        self.out = nn.Linear(in_f, output_dim)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for g, v in zip(self.gates, self.values):
            h = self.act(g(h)) * v(h)
        return self.out(h)


def _build_gated_mlp(
    input_dim: int,
    output_dim: int,
    depth: int,
    width: int,
    activation: str,
) -> nn.Module:
    return GatedMlp(input_dim, output_dim, depth, width, activation)


class MoeMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int,
        num_experts: int,
        activation: str,
    ) -> None:
        super().__init__()
        if depth < 1 or width < 1 or num_experts < 1:
            raise HTTPException(status_code=400, detail="MoE MLP depth, width, and numExperts must be >= 1")
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [_build_mlp(input_dim, output_dim, depth, width, activation) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_probs = torch.softmax(self.gate(x), dim=-1)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (expert_outs * gate_probs.unsqueeze(-1)).sum(dim=1)


def _build_moe_mlp(
    input_dim: int,
    output_dim: int,
    depth: int,
    width: int,
    num_experts: int,
    activation: str,
) -> nn.Module:
    return MoeMlp(input_dim, output_dim, depth, width, num_experts, activation)


class MlpTokenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        tokens_per_input: int = 1,
        tie_weights: bool = True,
        *,
        depth: int = 2,
        width: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise HTTPException(status_code=400, detail="MLP_token model vocabSize must be >= 2.")
        if embed_dim < 1:
            raise HTTPException(status_code=400, detail="MLP_token model embedDim must be >= 1.")
        if tokens_per_input < 1:
            raise HTTPException(status_code=400, detail="MLP_token model tokensPerInput must be >= 1.")
        d_flat = int(embed_dim) * int(tokens_per_input)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.body = _build_mlp(d_flat, d_flat, depth, width, activation)
        self.unembed = nn.Linear(d_flat, vocab_size, bias=True)
        self.tokens_per_input = int(tokens_per_input)
        self.tie_weights = tie_weights
        if tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            self.embedding.weight = self.unembed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise HTTPException(status_code=400, detail="MLP_token model expects x with shape [batch, context_length].")
        if int(token_ids.shape[1]) != self.tokens_per_input:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"MLP_token model expects input width {self.tokens_per_input}, "
                    f"got {int(token_ids.shape[1])}."
                ),
            )
        h = self.embedding(token_ids.long()).reshape(token_ids.shape[0], -1)
        h = self.body(h)
        if self.tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            return F.linear(h, self.unembed.weight, self.unembed.bias)
        return self.unembed(h)


class GatedMlpTokenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        tokens_per_input: int = 1,
        tie_weights: bool = True,
        *,
        depth: int = 2,
        width: int = 64,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise HTTPException(status_code=400, detail="Gated MLP_token model vocabSize must be >= 2.")
        if embed_dim < 1:
            raise HTTPException(status_code=400, detail="Gated MLP_token model embedDim must be >= 1.")
        if tokens_per_input < 1:
            raise HTTPException(status_code=400, detail="Gated MLP_token model tokensPerInput must be >= 1.")
        d_flat = int(embed_dim) * int(tokens_per_input)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.body = _build_gated_mlp(d_flat, d_flat, depth, width, activation)
        self.unembed = nn.Linear(d_flat, vocab_size, bias=True)
        self.tokens_per_input = int(tokens_per_input)
        self.tie_weights = tie_weights
        if tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            self.embedding.weight = self.unembed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise HTTPException(
                status_code=400, detail="Gated MLP_token model expects x with shape [batch, context_length]."
            )
        if int(token_ids.shape[1]) != self.tokens_per_input:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Gated MLP_token model expects input width {self.tokens_per_input}, "
                    f"got {int(token_ids.shape[1])}."
                ),
            )
        h = self.embedding(token_ids.long()).reshape(token_ids.shape[0], -1)
        h = self.body(h)
        if self.tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            return F.linear(h, self.unembed.weight, self.unembed.bias)
        return self.unembed(h)


class MoeMlpTokenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        tokens_per_input: int = 1,
        tie_weights: bool = True,
        *,
        depth: int = 2,
        width: int = 64,
        num_experts: int = 4,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise HTTPException(status_code=400, detail="MoE MLP_token model vocabSize must be >= 2.")
        if embed_dim < 1:
            raise HTTPException(status_code=400, detail="MoE MLP_token model embedDim must be >= 1.")
        if tokens_per_input < 1:
            raise HTTPException(status_code=400, detail="MoE MLP_token model tokensPerInput must be >= 1.")
        d_flat = int(embed_dim) * int(tokens_per_input)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.body = _build_moe_mlp(d_flat, d_flat, depth, width, num_experts, activation)
        self.unembed = nn.Linear(d_flat, vocab_size, bias=True)
        self.tokens_per_input = int(tokens_per_input)
        self.tie_weights = tie_weights
        if tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            self.embedding.weight = self.unembed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise HTTPException(
                status_code=400, detail="MoE MLP_token model expects x with shape [batch, context_length]."
            )
        if int(token_ids.shape[1]) != self.tokens_per_input:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"MoE MLP_token model expects input width {self.tokens_per_input}, "
                    f"got {int(token_ids.shape[1])}."
                ),
            )
        h = self.embedding(token_ids.long()).reshape(token_ids.shape[0], -1)
        h = self.body(h)
        if self.tie_weights and self.unembed.weight.shape == self.embedding.weight.shape:
            return F.linear(h, self.unembed.weight, self.unembed.bias)
        return self.unembed(h)


def _build_resnet_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_resnet_from_md(
        md,
        in_channels=_require_int(context.input_channels, "input_channels", NodeKind.resnet_model.value),
        num_classes=_require_int(context.num_classes, "num_classes", NodeKind.resnet_model.value),
    )


def _build_keskar_c1_c2_cnn_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_keskar_from_md(
        md,
        in_channels=_require_int(context.input_channels, "input_channels", NodeKind.keskar_c1_c2_cnn_model.value),
        num_classes=_require_int(context.num_classes, "num_classes", NodeKind.keskar_c1_c2_cnn_model.value),
    )


def _build_vgg11_cifar_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_vgg11_cifar(
        in_channels=_require_int(context.input_channels, "input_channels", NodeKind.vgg11_cifar_model.value),
        num_classes=_require_int(context.num_classes, "num_classes", NodeKind.vgg11_cifar_model.value),
    )


def _build_small_inception_cifar_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_small_inception_cifar(
        in_channels=_require_int(context.input_channels, "input_channels", NodeKind.small_inception_cifar_model.value),
        num_classes=_require_int(context.num_classes, "num_classes", NodeKind.small_inception_cifar_model.value),
    )


def _build_vit_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_vit_from_md(
        md,
        in_channels=_require_int(context.input_channels, "input_channels", NodeKind.vit_model.value),
        num_classes=_require_int(context.num_classes, "num_classes", NodeKind.vit_model.value),
        image_size=_require_int(context.image_size, "image_size", NodeKind.vit_model.value),
    )


def _build_numeric_transformer_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return numeric_transformer_from_canvas_md(md)


def _build_numeric_hyena_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return numeric_hyena_from_canvas_md(md)


def _build_mpp_spatiotemporal_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return mpp_spatiotemporal_from_canvas_md(md)


def _build_afno_lite_spatiotemporal_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return afno_lite_spatiotemporal_from_canvas_md(md)


def _build_mlp_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return _build_mlp(
        _require_int(context.input_dim, "input_dim", NodeKind.mlp_model.value),
        _require_int(context.output_dim, "output_dim", NodeKind.mlp_model.value),
        _scalar_int(md.get("depth"), 2),
        _scalar_int(md.get("width"), 64),
        _scalar_str(md.get("activation"), "relu"),
    )


def _build_gated_mlp_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return _build_gated_mlp(
        _require_int(context.input_dim, "input_dim", NodeKind.gated_mlp_model.value),
        _require_int(context.output_dim, "output_dim", NodeKind.gated_mlp_model.value),
        _scalar_int(md.get("depth"), 2),
        _scalar_int(md.get("width"), 64),
        _scalar_str(md.get("activation"), "silu"),
    )


def _build_moe_mlp_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return _build_moe_mlp(
        _require_int(context.input_dim, "input_dim", NodeKind.moe_mlp_model.value),
        _require_int(context.output_dim, "output_dim", NodeKind.moe_mlp_model.value),
        _scalar_int(md.get("depth"), 2),
        _scalar_int(md.get("width"), 64),
        _scalar_int(md.get("numExperts"), 4),
        _scalar_str(md.get("activation"), "silu"),
    )


def _build_kan_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    return build_kan_regression(
        _require_int(context.input_dim, "input_dim", NodeKind.kan_model.value),
        _require_int(context.output_dim, "output_dim", NodeKind.kan_model.value),
        _scalar_int(md.get("depth"), 2),
        _scalar_int(md.get("width"), 5),
        _scalar_int(md.get("grid"), 3),
        _scalar_int(md.get("k"), 3),
        _scalar_int(md.get("seed"), 0),
        _scalar_str(md.get("baseFun"), "silu"),
        fast_training=bool(context.kan_fast_training),
    )


def _tie_weights_from_md(md: dict[str, Any]) -> bool:
    return str(md.get("tieWeights") or "yes").strip().lower() not in {"no", "false", "0"}


def _build_mlp_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return MlpTokenModel(
        _scalar_int(md.get("vocabSize"), 59),
        _scalar_int(md.get("embedDim"), 2),
        tokens_per_input=_scalar_int(md.get("tokensPerInput"), 1),
        tie_weights=_tie_weights_from_md(md),
        depth=_scalar_int(md.get("depth"), 2),
        width=_scalar_int(md.get("width"), 64),
        activation=_scalar_str(md.get("activation"), "relu"),
    )


def _build_gated_mlp_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return GatedMlpTokenModel(
        _scalar_int(md.get("vocabSize"), 59),
        _scalar_int(md.get("embedDim"), 2),
        tokens_per_input=_scalar_int(md.get("tokensPerInput"), 1),
        tie_weights=_tie_weights_from_md(md),
        depth=_scalar_int(md.get("depth"), 2),
        width=_scalar_int(md.get("width"), 64),
        activation=_scalar_str(md.get("activation"), "silu"),
    )


def _build_moe_mlp_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return MoeMlpTokenModel(
        _scalar_int(md.get("vocabSize"), 59),
        _scalar_int(md.get("embedDim"), 2),
        tokens_per_input=_scalar_int(md.get("tokensPerInput"), 1),
        tie_weights=_tie_weights_from_md(md),
        depth=_scalar_int(md.get("depth"), 2),
        width=_scalar_int(md.get("width"), 64),
        num_experts=_scalar_int(md.get("numExperts"), 4),
        activation=_scalar_str(md.get("activation"), "silu"),
    )


def _build_transformer_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return token_transformer_from_canvas_md(md)


def _build_transformer_multi_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return multi_token_transformer_from_canvas_md(md)


def _token_bundle_dims(md: dict[str, Any], *, default_embed_dim: int = 2) -> tuple[int, int, int]:
    return (
        _scalar_int(md.get("vocabSize"), 59),
        _scalar_int(md.get("embedDim"), default_embed_dim),
        _scalar_int(md.get("contextLength"), 1),
    )


def _token_bundle_common(md: dict[str, Any]) -> tuple[bool, int]:
    causal_s = str(md.get("causalAttention") or "yes").strip().lower()
    causal = causal_s not in {"no", "false", "0"}
    local_mixing_kernel = _scalar_int(md.get("localMixingKernel"), 0)
    return causal, local_mixing_kernel


def _build_attention_only_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md)
    num_heads = _scalar_int(md.get("numHeads"), 1)
    if num_heads < 1 or embed_dim % num_heads != 0:
        raise HTTPException(
            status_code=400,
            detail="attention_only_model numHeads must be >= 1 and divide embedDim.",
        )
    causal, local_mixing_kernel = _token_bundle_common(md)
    return AttentionTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        num_heads,
        causal=causal,
        local_mixing_kernel=local_mixing_kernel,
        qk_norm=_scalar_bool(md.get("qkNorm"), False),
        attn_temperature=max(_scalar_float(md.get("attnTemperature"), 1.0), 1e-6),
        attn_logit_cap=max(_scalar_float(md.get("attnLogitCap"), 0.0), 0.0),
        attn_dropout_p=max(0.0, min(_scalar_float(md.get("attnDropout"), 0.0), 1.0)),
    )


def _build_linear_attention_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md)
    num_heads = _scalar_int(md.get("numHeads"), 1)
    if num_heads < 1 or embed_dim % num_heads != 0:
        raise HTTPException(
            status_code=400,
            detail="linear_attention_model numHeads must be >= 1 and divide embedDim.",
        )
    causal, local_mixing_kernel = _token_bundle_common(md)
    return LinearAttentionTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        num_heads,
        causal=causal,
        local_mixing_kernel=local_mixing_kernel,
    )


def _build_diagonal_ssm_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md, default_embed_dim=32)
    return DiagonalSsmTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        num_layers=_scalar_int(md.get("numLayers"), 1),
        local_mixing_kernel=_scalar_int(md.get("localMixingKernel"), 0),
    )


def _build_rwkv_time_mix_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md, default_embed_dim=32)
    return RwkvTimeMixTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        depth=_scalar_int(md.get("depth"), 2),
        local_mixing_kernel=_scalar_int(md.get("localMixingKernel"), 0),
    )


def _build_hyena_like_conv_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md, default_embed_dim=32)
    return HyenaLikeConvTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        depth=_scalar_int(md.get("depth"), 2),
        kernel_size=_scalar_int(md.get("convKernel"), 7),
        ff_mult=_scalar_int(md.get("ffMult"), 2),
        local_mixing_kernel=_scalar_int(md.get("localMixingKernel"), 0),
    )


def _build_slot_attention_token_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    vocab_size, embed_dim, context_length = _token_bundle_dims(md, default_embed_dim=32)
    return SlotAttentionTokenPredictBundle(
        vocab_size,
        embed_dim,
        context_length,
        num_slots=_scalar_int(md.get("numSlots"), 4),
        slot_iters=_scalar_int(md.get("slotIters"), 3),
        local_mixing_kernel=_scalar_int(md.get("localMixingKernel"), 0),
    )


def _build_diffusion_score_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    data_dim = context.input_dim if context.input_dim is not None else _scalar_int(md.get("inputDim"), 8)
    return DiffusionScoreMLP(
        int(data_dim),
        hidden_dim=_scalar_int(md.get("hiddenDim"), 128),
        depth=_scalar_int(md.get("depth"), 3),
        time_embed_dim=_scalar_int(md.get("timeEmbedDim"), 64),
        max_timesteps=_scalar_int(md.get("diffusionTimesteps"), 100),
    )


def _build_unet_ddpm_model(md: dict[str, Any], _context: ModelBuildContext) -> nn.Module:
    return build_unet_ddpm_from_md(md)


def _build_crl_residual_mlp(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    state_dim = _scalar_int(md.get("stateDim"), 4)
    action_dim = _scalar_int(md.get("actionDim"), 2)
    goal_dim = _scalar_int(md.get("goalDim"), 2)
    actor_depth = max(4, _scalar_int(md.get("actorDepth"), 4))
    critic_depth = max(4, _scalar_int(md.get("criticDepth"), 4))
    if actor_depth % 4 or critic_depth % 4:
        raise HTTPException(status_code=400, detail="crl_residual_mlp actorDepth/criticDepth must be multiples of 4.")
    return CrlResidualAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        obs_dim_full=state_dim + goal_dim,
        actor_width=max(8, _scalar_int(md.get("actorWidth"), 128)),
        critic_width=max(8, _scalar_int(md.get("criticWidth"), 128)),
        actor_depth=actor_depth,
        critic_depth=critic_depth,
        embed_dim=max(8, _scalar_int(md.get("embedDim"), 64)),
        activation=parse_crl_residual_activation(md),
    )


def _build_residual_ln_model(md: dict[str, Any], context: ModelBuildContext) -> nn.Module:
    ln_mode = _scalar_str(md.get("lnMode"), "pre_ln").strip().lower()
    return ResidualLNModel(
        _scalar_int(md.get("dim"), 256),
        _scalar_int(md.get("depth"), 100),
        _scalar_float(md.get("alpha"), 1.0),
        _scalar_str(md.get("activation"), "relu"),
        pre_ln=ln_mode != "post_ln",
    )


MODEL_BUILDERS: dict[str, ModelBuilder] = {
}


def model_builder_node_types() -> frozenset[str]:
    # All specialized and registered providers are supported model builders.
    return frozenset(MODEL_BUILDERS) | frozenset(model_defs_builders())


def build_model_for_node(model_node: Node, context: ModelBuildContext) -> nn.Module:
    return build_model_from_type(model_node.type, dict(model_node.data or {}), context)


def build_model_from_type(
    node_type: object,
    data: dict[str, Any],
    context: ModelBuildContext | None = None,
) -> nn.Module:
    key = str(getattr(node_type, "value", node_type))
    # Prefer registered NodeDef builders, then fall back to specialized builders.
    builder = model_defs_builders().get(key) or MODEL_BUILDERS.get(key)
    if builder is None:
        raise HTTPException(status_code=400, detail=f"Unsupported registered model builder type: {node_type}")
    return builder(dict(data), context or ModelBuildContext())
