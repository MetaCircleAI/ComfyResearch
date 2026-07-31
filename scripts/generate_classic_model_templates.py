#!/usr/bin/env python3
"""Generate deterministic classic model templates for the graph library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "graph_library" / "templates"
BASE_SAVED_AT_MS = 1_781_366_400_000.0
X_SPACING = 320


AtomicSpec = tuple[str, str, dict[str, Any]]


def _atomic_data(title: str, data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out["instanceTitle"] = title
    out["ioMode"] = "input-output"
    return out


def linear(in_features: int, out_features: int, title: str, *, bias: int = 1, seed: int = 0) -> AtomicSpec:
    return (
        "linear_layer",
        "linear",
        _atomic_data(
            title,
            {
                "inFeatures": in_features,
                "outFeatures": out_features,
                "bias": bias,
                "seed": seed,
            },
        ),
    )


def activation(kind: str, title: str, *, leaky_p: float = 0.0) -> AtomicSpec:
    return (
        "activation_layer",
        "activation",
        _atomic_data(title, {"activation": kind, "leakyP": leaky_p}),
    )


def layer_norm(width: int, title: str, *, eps: float = 1e-5) -> AtomicSpec:
    return (
        "layer_norm_layer",
        "layer-norm",
        _atomic_data(
            title,
            {"normalizedShape": width, "eps": eps, "elementwiseAffine": 1},
        ),
    )


def rms_norm(width: int, title: str, *, eps: float = 1e-6) -> AtomicSpec:
    return (
        "rms_norm_layer",
        "rms-norm",
        _atomic_data(
            title,
            {"normalizedShape": width, "eps": eps, "elementwiseAffine": 1},
        ),
    )


def embedding(vocab_size: int, embedding_dim: int, title: str, *, columns: int = 1, seed: int = 0) -> AtomicSpec:
    return (
        "embedding_layer",
        "embedding",
        _atomic_data(
            title,
            {
                "numEmbeddings": vocab_size,
                "embeddingDim": embedding_dim,
                "numIndexColumns": columns,
                "paddingIdx": -1,
                "scaleGradByFreq": 0,
                "seed": seed,
            },
        ),
    )


def unembedding(in_features: int, out_features: int, title: str, *, bias: int = 1, seed: int = 0) -> AtomicSpec:
    return (
        "unembedding_layer",
        "unembedding",
        _atomic_data(
            title,
            {
                "inFeatures": in_features,
                "outFeatures": out_features,
                "bias": bias,
                "seed": seed,
            },
        ),
    )


def absolute_pos_embed(max_seq_len: int, embedding_dim: int, title: str, *, seed: int = 0) -> AtomicSpec:
    return (
        "absolute_pos_embed_layer",
        "absolute-pos",
        _atomic_data(
            title,
            {"maxSeqLen": max_seq_len, "embeddingDim": embedding_dim, "seed": seed},
        ),
    )


def rotary_embed(rotary_dim: int, title: str, *, theta_base: float = 10_000.0, seed: int = 0) -> AtomicSpec:
    return (
        "rotary_embed_layer",
        "rotary",
        _atomic_data(
            title,
            {"rotaryDim": rotary_dim, "thetaBase": theta_base, "seed": seed},
        ),
    )


def local_mixing(model_dim: int, kernel_size: int, title: str, *, seed: int = 0) -> AtomicSpec:
    return (
        "local_mixing_layer",
        "local-mixing",
        _atomic_data(
            title,
            {"modelDim": model_dim, "kernelSize": kernel_size, "seed": seed},
        ),
    )


def _edge(source: str, target: str) -> dict[str, Any]:
    return {
        "id": f"edge__{source}__tensor_out__{target}__tensor_in",
        "source": source,
        "target": target,
        "sourceHandle": "tensor_out",
        "targetHandle": "tensor_in",
    }


def _chain_entry(
    template_id: str,
    name: str,
    tier: str,
    specs: list[AtomicSpec],
    order: int,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for idx, (node_type, slug, data) in enumerate(specs):
        node_id = f"{template_id}__{slug}-{idx}"
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "position": {"x": idx * X_SPACING, "y": 0},
                "data": data,
                "parentId": None,
                "extent": None,
                "hidden": None,
                "style": None,
            }
        )
        if idx > 0:
            edges.append(_edge(nodes[idx - 1]["id"], node_id))
    return {
        "id": template_id,
        "name": name,
        "tier": tier,
        "document": {"version": 1, "nodes": nodes, "edges": edges, "viewport": None},
        "savedAt": BASE_SAVED_AT_MS + order * 1000.0,
        "libraryOrigin": "combined_model",
    }


def _model_entry(
    template_id: str,
    name: str,
    tier: str,
    node_type: str,
    data: dict[str, Any],
    order: int,
) -> dict[str, Any]:
    model_data = dict(data)
    model_data["instanceTitle"] = name.removeprefix("Classic Model / ")
    return {
        "id": template_id,
        "name": name,
        "tier": tier,
        "document": {
            "version": 1,
            "nodes": [
                {
                    "id": f"{template_id}__model",
                    "type": node_type,
                    "position": {"x": 0, "y": 0},
                    "data": model_data,
                    "parentId": None,
                    "extent": None,
                    "hidden": None,
                    "style": None,
                }
            ],
            "edges": [],
            "viewport": None,
        },
        "savedAt": BASE_SAVED_AT_MS + order * 1000.0,
    }


ATOMIC_TEMPLATES: list[tuple[str, str, str, list[AtomicSpec]]] = [
    (
        "classic-atomic-linear-regression",
        "Classic / Linear Regression",
        "small",
        [linear(10, 1, "Linear 10 -> 1")],
    ),
    (
        "classic-atomic-logistic-regression",
        "Classic / Logistic Regression",
        "small",
        [linear(10, 1, "Logit 10 -> 1"), activation("sigmoid", "Sigmoid")],
    ),
    (
        "classic-atomic-softmax-classifier",
        "Classic / Softmax Classifier",
        "small",
        [linear(784, 10, "Class logits 784 -> 10")],
    ),
    (
        "classic-atomic-mlp-small",
        "Classic / MLP Small",
        "small",
        [
            linear(10, 64, "Input projection"),
            activation("relu", "ReLU"),
            linear(64, 1, "Output projection"),
        ],
    ),
    (
        "classic-atomic-mlp-deep",
        "Classic / MLP Deep",
        "medium",
        [
            linear(10, 128, "Input projection"),
            activation("relu", "ReLU 0"),
            linear(128, 128, "Hidden 0"),
            activation("relu", "ReLU 1"),
            linear(128, 128, "Hidden 1"),
            activation("relu", "ReLU 2"),
            linear(128, 1, "Output projection"),
        ],
    ),
    (
        "classic-atomic-mlp-gelu",
        "Classic / MLP GELU",
        "medium",
        [
            linear(10, 128, "Input projection"),
            activation("gelu", "GELU 0"),
            linear(128, 64, "Bottleneck"),
            activation("gelu", "GELU 1"),
            linear(64, 1, "Output projection"),
        ],
    ),
    (
        "classic-atomic-leaky-mlp",
        "Classic / Leaky MLP",
        "medium",
        [
            linear(16, 128, "Input projection"),
            activation("leaky_relu", "Leaky ReLU 0", leaky_p=0.1),
            linear(128, 64, "Hidden projection"),
            activation("leaky_relu", "Leaky ReLU 1", leaky_p=0.1),
            linear(64, 4, "Output projection"),
        ],
    ),
    (
        "classic-atomic-autoencoder-tiny",
        "Classic / Autoencoder Tiny",
        "medium",
        [
            linear(64, 16, "Encoder wide -> mid"),
            activation("tanh", "Tanh 0"),
            linear(16, 4, "Latent bottleneck"),
            activation("tanh", "Tanh 1"),
            linear(4, 16, "Decoder mid"),
            activation("tanh", "Tanh 2"),
            linear(16, 64, "Reconstruction"),
        ],
    ),
    (
        "classic-atomic-autoencoder-mnist",
        "Classic / Autoencoder MNIST",
        "large",
        [
            linear(784, 256, "Encoder 784 -> 256"),
            activation("relu", "ReLU 0"),
            linear(256, 64, "Latent 64"),
            activation("relu", "ReLU 1"),
            linear(64, 256, "Decoder 64 -> 256"),
            activation("relu", "ReLU 2"),
            linear(256, 784, "Reconstruction"),
            activation("sigmoid", "Pixel sigmoid"),
        ],
    ),
    (
        "classic-atomic-bottleneck-classifier",
        "Classic / Bottleneck Classifier",
        "large",
        [
            linear(784, 128, "Input projection"),
            activation("relu", "ReLU 0"),
            linear(128, 32, "Bottleneck"),
            activation("relu", "ReLU 1"),
            linear(32, 10, "Class logits"),
        ],
    ),
    (
        "classic-atomic-layernorm-mlp",
        "Classic / LayerNorm MLP",
        "medium",
        [
            linear(32, 128, "Input projection"),
            layer_norm(128, "LayerNorm 0"),
            activation("gelu", "GELU 0"),
            linear(128, 128, "Hidden projection"),
            layer_norm(128, "LayerNorm 1"),
            activation("gelu", "GELU 1"),
            linear(128, 8, "Output projection"),
        ],
    ),
    (
        "classic-atomic-bigram-lm",
        "Classic / Bigram Language Model",
        "small",
        [
            embedding(100, 64, "Token embedding", columns=1),
            unembedding(64, 100, "Token logits"),
        ],
    ),
    (
        "classic-atomic-token-mlp-lm",
        "Classic / Token MLP Language Model",
        "large",
        [
            embedding(4096, 64, "Token embedding", columns=1),
            layer_norm(64, "Pre-norm"),
            linear(64, 256, "Feed-forward up"),
            activation("gelu", "GELU"),
            linear(256, 64, "Feed-forward down"),
            layer_norm(64, "Post-norm"),
            unembedding(64, 4096, "LM head"),
        ],
    ),
    (
        "classic-atomic-gpt-mlp-mixer",
        "Classic / GPT MLP Mixer Approx",
        "large",
        [
            embedding(8192, 128, "Token embedding", columns=1),
            absolute_pos_embed(256, 128, "Absolute position"),
            rms_norm(128, "RMSNorm 0"),
            local_mixing(128, 5, "Causal local mixing"),
            linear(128, 512, "MLP up"),
            activation("gelu", "GELU"),
            linear(512, 128, "MLP down"),
            rms_norm(128, "RMSNorm 1"),
            unembedding(128, 8192, "LM head"),
        ],
    ),
    (
        "classic-atomic-rope-token-mlp",
        "Classic / RoPE Token MLP",
        "large",
        [
            embedding(4096, 96, "Token embedding", columns=1),
            rotary_embed(96, "Rotary embedding"),
            rms_norm(96, "RMSNorm"),
            linear(96, 384, "MLP up"),
            activation("silu", "SiLU"),
            linear(384, 96, "MLP down"),
            unembedding(96, 4096, "LM head"),
        ],
    ),
]


MODEL_TEMPLATES: list[tuple[str, str, str, str, dict[str, Any]]] = [
    (
        "classic-model-mlp-regressor",
        "Classic Model / MLP Regressor",
        "small",
        "mlp_model",
        {"levelMode": "high", "inputDim": 10, "outputDim": 1, "depth": 2, "width": 64, "activation": "relu", "seed": 0},
    ),
    (
        "classic-model-mlp-classifier",
        "Classic Model / MLP Classifier",
        "small",
        "mlp_model",
        {"levelMode": "high", "inputDim": 784, "outputDim": 10, "depth": 2, "width": 128, "activation": "relu", "seed": 1},
    ),
    (
        "classic-model-token-mlp",
        "Classic Model / Token MLP",
        "small",
        "mlp_token_model",
        {"vocabSize": 1000, "embedDim": 64, "tokensPerInput": 1, "depth": 2, "width": 128, "numExperts": 4, "activation": "gelu", "tieWeights": "yes", "seed": 2},
    ),
    (
        "classic-model-resnet18",
        "Classic Model / ResNet-18",
        "medium",
        "resnet_model",
        {"variant": "resnet18", "baseChannels": 32, "blocksStage1": 2, "blocksStage2": 2, "blocksStage3": 2, "blocksStage4": 2, "kernelSize": 3, "seed": 3, "specCodeName": "resnetModelSpec"},
    ),
    (
        "classic-model-resnet34",
        "Classic Model / ResNet-34",
        "large",
        "resnet_model",
        {"variant": "resnet34", "baseChannels": 32, "blocksStage1": 3, "blocksStage2": 4, "blocksStage3": 6, "blocksStage4": 3, "kernelSize": 3, "seed": 4, "specCodeName": "resnetModelSpec"},
    ),
    (
        "classic-model-vit-tiny",
        "Classic Model / ViT Tiny",
        "medium",
        "vit_model",
        {"variant": "tiny", "patchSize": 4, "hiddenDim": 128, "depth": 3, "numHeads": 4, "seed": 5, "specCodeName": "vitModelSpec"},
    ),
    (
        "classic-model-vit-small",
        "Classic Model / ViT Small",
        "large",
        "vit_model",
        {"variant": "small", "patchSize": 4, "hiddenDim": 256, "depth": 6, "numHeads": 8, "seed": 6, "specCodeName": "vitModelSpec"},
    ),
    (
        "classic-model-transformer-token-small",
        "Classic Model / Transformer Token Small",
        "medium",
        "transformer_token_model",
        {"vocabSize": 1024, "contextLength": 16, "modelDim": 64, "numHeads": 4, "numLayers": 2, "ffDim": 256, "activation": "gelu", "encoderBackend": "pytorch", "encoderDropout": 0, "spectralNormLinears": "no", "lmLogitScale": 1, "stableQkNorm": "no", "stableAttnTemperature": 1, "stableAttnLogitCap": 0, "stableAttnDropout": 0, "tieEmbeddingLmHead": "yes", "causalAttention": "yes", "localMixingKernel": 0, "seed": 7},
    ),
    (
        "classic-model-gpt2-tiny",
        "Classic Model / GPT-2 Tiny",
        "large",
        "transformer_token_model",
        {"vocabSize": 8192, "contextLength": 128, "modelDim": 192, "numHeads": 6, "numLayers": 4, "ffDim": 768, "activation": "gelu", "encoderBackend": "stable", "encoderDropout": 0.1, "spectralNormLinears": "no", "lmLogitScale": 1, "stableQkNorm": "yes", "stableAttnTemperature": 1, "stableAttnLogitCap": 30, "stableAttnDropout": 0.1, "tieEmbeddingLmHead": "yes", "causalAttention": "yes", "localMixingKernel": 0, "seed": 8},
    ),
    (
        "classic-model-transformer-multitoken",
        "Classic Model / Multi-Token Transformer",
        "large",
        "transformer_multi_token_model",
        {"vocabSize": 1024, "contextLength": 16, "tokensPerPosition": 2, "modelDim": 96, "numHeads": 4, "numLayers": 3, "ffDim": 384, "encoderBackend": "pytorch", "encoderDropout": 0, "spectralNormLinears": "no", "lmLogitScale": 1, "stableQkNorm": "no", "stableAttnTemperature": 1, "stableAttnLogitCap": 0, "stableAttnDropout": 0, "tieEmbeddingLmHead": "no", "causalAttention": "yes", "seed": 9},
    ),
    (
        "classic-model-numeric-transformer",
        "Classic Model / Numeric Transformer",
        "medium",
        "numeric_transformer_model",
        {"contextLength": 8, "inputDim": 2, "outputDim": 2, "modelDim": 64, "numHeads": 4, "numLayers": 2, "ffDim": 256, "activation": "gelu", "encoderBackend": "pytorch", "encoderDropout": 0, "spectralNormLinears": "no", "stableQkNorm": "no", "stableAttnTemperature": 1, "stableAttnLogitCap": 0, "stableAttnDropout": 0, "causalAttention": "yes", "seed": 10},
    ),
    (
        "classic-model-attention-only",
        "Classic Model / Attention Only",
        "medium",
        "attention_only_model",
        {"vocabSize": 1024, "embedDim": 64, "numHeads": 4, "contextLength": 32, "causalAttention": "yes", "localMixingKernel": 0, "qkNorm": "no", "attnTemperature": 1, "attnLogitCap": 0, "attnDropout": 0, "seed": 11},
    ),
    (
        "classic-model-residual-ln-mlp",
        "Classic Model / Residual-LN MLP",
        "medium",
        "residual_ln_model",
        {"dim": 256, "depth": 24, "alpha": 1, "lnMode": "pre_ln", "activation": "gelu", "seed": 12},
    ),
    (
        "classic-model-gated-mlp",
        "Classic Model / Gated MLP",
        "small",
        "gated_mlp_model",
        {"inputDim": 10, "outputDim": 1, "depth": 2, "width": 64, "activation": "silu", "seed": 13},
    ),
    (
        "classic-model-moe-mlp",
        "Classic Model / MoE MLP",
        "medium",
        "moe_mlp_model",
        {"inputDim": 10, "outputDim": 1, "depth": 2, "width": 64, "numExperts": 4, "activation": "silu", "seed": 14},
    ),
    (
        "classic-model-kan-regressor",
        "Classic Model / KAN Regressor",
        "small",
        "kan_model",
        {"inputDim": 2, "outputDim": 1, "depth": 2, "width": 5, "grid": 3, "k": 3, "baseFun": "silu", "seed": 15},
    ),
    (
        "classic-model-numeric-hyena",
        "Classic Model / Numeric Hyena",
        "medium",
        "numeric_hyena_model",
        {"contextLength": 16, "inputDim": 2, "outputDim": 2, "modelDim": 96, "depth": 3, "convKernel": 9, "ffMult": 2, "localMixingKernel": 0, "seed": 16},
    ),
    (
        "classic-model-diffusion-score-mlp",
        "Classic Model / Diffusion Score MLP",
        "medium",
        "diffusion_score_model",
        {"inputDim": 8, "hiddenDim": 128, "depth": 3, "timeEmbedDim": 64, "diffusionTimesteps": 100, "seed": 17},
    ),
    (
        "classic-model-mpp-spatiotemporal",
        "Classic Model / MPP Spatiotemporal",
        "large",
        "mpp_spatiotemporal_model",
        {"contextFrames": 4, "channels": 1, "gridSize": 16, "inputDim": 1024, "outputDim": 1024, "patchSize": 4, "embedDim": 128, "depth": 4, "numHeads": 4, "ffRatio": 4, "dropout": 0, "seed": 18},
    ),
    (
        "classic-model-afno-lite",
        "Classic Model / AFNO Lite",
        "large",
        "afno_lite_spatiotemporal_model",
        {"contextFrames": 4, "channels": 1, "gridSize": 16, "inputDim": 1024, "outputDim": 1024, "patchSize": 4, "embedDim": 64, "depth": 2, "numHeads": 4, "ffRatio": 2, "dropout": 0, "numSpectralBlocks": 1, "maxFrequencyModes": 4, "spectralShrinkFactor": 1, "seed": 19},
    ),
]


def build_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    order = 0
    for template_id, name, tier, specs in ATOMIC_TEMPLATES:
        entries.append(_chain_entry(template_id, name, tier, specs, order))
        order += 1
    for template_id, name, tier, node_type, data in MODEL_TEMPLATES:
        entries.append(_model_entry(template_id, name, tier, node_type, data, order))
        order += 1
    return entries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for entry in build_entries():
        path = OUT_DIR / f"{entry['id']}.json"
        path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(build_entries())} classic model templates to {OUT_DIR}")


if __name__ == "__main__":
    main()
