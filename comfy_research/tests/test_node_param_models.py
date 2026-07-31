import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from comfy_research.generated.node_manifest import load_node_manifest
from comfy_research.generated.node_params import NODE_PARAM_MODELS, validate_node_params
from comfy_research.schemas.graph import GraphDocument


def test_generated_param_models_match_manifest_types() -> None:
    manifest_types = {str(entry["type"]) for entry in load_node_manifest()}
    assert set(NODE_PARAM_MODELS) == manifest_types


def test_generated_param_models_include_registry_fields() -> None:
    adam_fields = set(NODE_PARAM_MODELS["adam_optimizer"].model_fields)
    assert {"learningRate", "beta1", "beta2", "epsilon", "weightDecay"} <= adam_fields

    linear_fields = set(NODE_PARAM_MODELS["linear_dataset"].model_fields)
    assert {"inputDim", "outputDim", "noiseLevel", "alpha", "samplingMode"} <= linear_fields

    mlp_fields = set(NODE_PARAM_MODELS["mlp_model"].model_fields)
    assert {"inputDim", "outputDim", "depth", "width", "activation", "seed"} <= mlp_fields

    ce_fields = set(NODE_PARAM_MODELS["cross_entropy_loss"].model_fields)
    assert {"lossScale", "labelSmoothing", "lossMaskContextLength", "lossMaskMode"} <= ce_fields

    mup_lr_fields = set(NODE_PARAM_MODELS["mup_lr_schedule"].model_fields)
    assert {"mupEmbedLrMult", "mupHiddenLrMult", "mupOutputLrMult"} <= mup_lr_fields

    trainer_fields = set(NODE_PARAM_MODELS["trainer"].model_fields)
    assert {"trainingSteps", "logFrequency", "computeDevice", "batchSize", "gradClipMaxNorm"} <= trainer_fields

    gradient_norm_fields = set(NODE_PARAM_MODELS["observable_gradient_norm"].model_fields)
    assert {"normAggregation", "gradientNormNormalized"} <= gradient_norm_fields

    hessian_fields = set(NODE_PARAM_MODELS["observable_hessian_eigenvalues"].model_fields)
    assert {"topK", "order"} <= hessian_fields


def test_generated_param_models_reject_invalid_registry_field_types() -> None:
    with pytest.raises(ValidationError):
        validate_node_params("adam_optimizer", {"learningRate": {"not": "a float"}})

    with pytest.raises(ValidationError):
        validate_node_params("trainer", {"trainingSteps": {"not": "an int"}})

    with pytest.raises(ValidationError):
        validate_node_params("observable_gradient_norm", {"gradientNormNormalized": {"not": "a bool"}})

    validate_node_params("linear_dataset", {"noiseLevel": 0.5, "alpha": 0.75})
    validate_node_params("activation_layer", {"leakyP": 0.1})
    validate_node_params("bigram_low_rank_dataset", {"corruptRatio": 0.01})
    validate_node_params("cross_entropy_loss", {"lossScale": 0.5, "labelSmoothing": 0.1})
    validate_node_params("mup_lr_schedule", {"mupEmbedLrMult": 0.5, "mupHiddenLrMult": 0.75})


def test_generated_param_models_validate_committed_templates() -> None:
    template_paths = sorted(Path("data/graph_library/templates").glob("*.json"))
    assert template_paths
    validated = 0
    for path in template_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        document = raw.get("document", raw)
        for node in document.get("nodes", []):
            node_type = str(node.get("type", ""))
            if node_type not in NODE_PARAM_MODELS:
                continue
            validate_node_params(node_type, node.get("data") if isinstance(node.get("data"), dict) else {})
            validated += 1
    assert validated > 0


def test_graph_document_validation_uses_generated_param_models_for_templates() -> None:
    template_paths = sorted(Path("data/graph_library/templates").glob("*.json"))
    assert template_paths
    parsed = 0
    for path in template_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        document = raw.get("document", raw)
        if isinstance(document, dict) and "nodes" in document:
            GraphDocument.model_validate(document)
            parsed += 1
    assert parsed > 0


def test_mlp_model_param_model_includes_output_scale() -> None:
    """outputScale 必须进入推断字段链（manifest → param model → sweep 识别）。"""
    mlp_fields = set(NODE_PARAM_MODELS["mlp_model"].model_fields)
    assert "outputScale" in mlp_fields
    scalar = validate_node_params("mlp_model", {"outputScale": 0.01})
    assert scalar is not None
    swept = validate_node_params("mlp_model", {"outputScale": [0.01, 1.0]})
    assert swept is not None
