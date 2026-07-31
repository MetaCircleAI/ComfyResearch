from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from comfy_research.generated.node_capabilities import node_types_with_capability
from comfy_research.generated.node_manifest import load_node_manifest

ConstructionCategory = Literal["dataset", "model", "optimizer", "loss"]
BuilderStatus = Literal["legacy_trainer_run", "codegen_owned", "registered", "codegen_only"]

CONSTRUCTION_CATEGORIES: frozenset[str] = frozenset({"dataset", "model", "optimizer", "loss"})
REGISTERED_BUILDER_IMPLEMENTATIONS: dict[str, str] = {
    "adam_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "adamw_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "signsgd_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "shampoo_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "soap_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "binary_cross_entropy_with_logits_loss": "comfy_research.engine.losses.loss_builders",
    "cross_entropy_loss": "comfy_research.engine.losses.loss_builders",
    "diffusion_mse_loss": "comfy_research.engine.losses.loss_builders",
    "mse_loss": "comfy_research.engine.losses.loss_builders",
    "afno_lite_spatiotemporal_model": "comfy_research.engine.models.model_builders",
    "attention_only_model": "comfy_research.engine.models.model_builders",
    "crl_residual_mlp": "comfy_research.engine.models.model_builders",
    "diagonal_ssm_token_model": "comfy_research.engine.models.model_builders",
    "diffusion_score_model": "comfy_research.engine.models.model_builders",
    "unet_ddpm_model": "comfy_research.engine.models.model_builders",
    "gated_mlp_model": "comfy_research.engine.models.model_builders",
    "gated_mlp_token_model": "comfy_research.engine.models.model_builders",
    "hyena_like_conv_model": "comfy_research.engine.models.model_builders",
    "kan_model": "comfy_research.engine.models.model_builders",
    "keskar_c1_c2_cnn_model": "comfy_research.engine.models.model_builders",
    "linear_attention_model": "comfy_research.engine.models.model_builders",
    "mlp_model": "comfy_research.engine.models.model_builders",
    "mlp_token_model": "comfy_research.engine.models.model_builders",
    "moe_mlp_model": "comfy_research.engine.models.model_builders",
    "moe_mlp_token_model": "comfy_research.engine.models.model_builders",
    "mpp_spatiotemporal_model": "comfy_research.engine.models.model_builders",
    "numeric_hyena_model": "comfy_research.engine.models.model_builders",
    "numeric_transformer_model": "comfy_research.engine.models.model_builders",
    "resnet_model": "comfy_research.engine.models.model_builders",
    "residual_ln_model": "comfy_research.engine.models.model_builders",
    "rwkv_time_mix_token_model": "comfy_research.engine.models.model_builders",
    "small_inception_cifar_model": "comfy_research.engine.models.model_builders",
    "sgd_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "muon_optimizer": "comfy_research.engine.optimizers.optimizer_builders",
    "slot_attention_token_model": "comfy_research.engine.models.model_builders",
    "transformer_multi_token_model": "comfy_research.engine.models.model_builders",
    "transformer_token_model": "comfy_research.engine.models.model_builders",
    "vgg11_cifar_model": "comfy_research.engine.models.model_builders",
    "vit_model": "comfy_research.engine.models.model_builders",
}


@dataclass(frozen=True)
class NodeBuilderRegistration:
    node_type: str
    category: ConstructionCategory
    status: BuilderStatus
    implementation: str


def construction_manifest_entries() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for entry in load_node_manifest():
        category = str(entry.get("category", ""))
        node_type = str(entry.get("type", ""))
        if category in CONSTRUCTION_CATEGORIES and node_type:
            out[node_type] = entry
    return out


def construction_manifest_node_types() -> dict[str, ConstructionCategory]:
    return {
        node_type: str(entry["category"])  # type: ignore[return-value]
        for node_type, entry in construction_manifest_entries().items()
    }


def construction_node_types_for(category: ConstructionCategory) -> tuple[str, ...]:
    return tuple(
        sorted(
            node_type
            for node_type, entry_category in construction_manifest_node_types().items()
            if entry_category == category
        )
    )


def _registration(node_type: str, entry: dict[str, object]) -> NodeBuilderRegistration:
    category = str(entry["category"])
    implementation = REGISTERED_BUILDER_IMPLEMENTATIONS.get(node_type, "comfy_research.engine.runs.trainer_run")
    if node_type in REGISTERED_BUILDER_IMPLEMENTATIONS:
        status: BuilderStatus = "registered"
    else:
        status = "codegen_owned" if entry.get("hasCodegen") is True else "legacy_trainer_run"
    return NodeBuilderRegistration(
        node_type=node_type,
        category=category,  # type: ignore[arg-type]
        status=status,
        implementation=implementation,
    )


BUILDERS: dict[str, NodeBuilderRegistration] = {
    node_type: _registration(node_type, entry)
    for node_type, entry in construction_manifest_entries().items()
}


def builder_for(node_type: str) -> NodeBuilderRegistration | None:
    return BUILDERS.get(str(node_type))


def builder_status_for(node_type: str) -> BuilderStatus | None:
    registration = builder_for(str(node_type))
    return registration.status if registration else None


def registered_builder_node_types_for(category: ConstructionCategory) -> tuple[str, ...]:
    return tuple(
        sorted(
            node_type
            for node_type, registration in BUILDERS.items()
            if registration.category == category and registration.status == "registered"
        )
    )


def registered_trainer_model_node_types() -> tuple[str, ...]:
    """Model node types accepted directly by the trainer model socket via registered builders."""
    return registered_builder_node_types_for("model")


def registered_weight_export_model_node_types() -> tuple[str, ...]:
    """Model node types accepted directly by model_weight_tensors."""
    return registered_builder_node_types_for("model")


def registered_activation_model_node_types() -> tuple[str, ...]:
    """Model node types accepted directly by activation collection without wire picks."""
    return tuple(sorted(node_types_with_capability("activation_model")))


def registered_trainer_dataset_node_types() -> tuple[str, ...]:
    """Dataset node types accepted by the trainer dataset sockets."""
    non_trainer_dataset_nodes = frozenset(
        {
            "crl_env_config",
            "input_sampler",
            "random_input_distribution",
        }
    )
    return tuple(
        node_type
        for node_type in construction_node_types_for("dataset")
        if node_type not in non_trainer_dataset_nodes
    )
