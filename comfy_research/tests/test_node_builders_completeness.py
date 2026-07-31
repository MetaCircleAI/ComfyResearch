from comfy_research.engine.node_builder_registry import (
    BUILDERS,
    CONSTRUCTION_CATEGORIES,
    builder_for,
    construction_manifest_node_types,
    construction_node_types_for,
    registered_activation_model_node_types,
    registered_builder_node_types_for,
    registered_trainer_dataset_node_types,
    registered_trainer_model_node_types,
    registered_weight_export_model_node_types,
)
from comfy_research.generated.node_manifest import load_node_manifest


def test_builder_registry_covers_construction_manifest_nodes() -> None:
    expected = construction_manifest_node_types()
    assert expected
    assert set(BUILDERS) == set(expected)
    for node_type, category in expected.items():
        registration = builder_for(node_type)
        assert registration is not None
        assert registration.node_type == node_type
        assert registration.category == category
        assert registration.status in {"legacy_trainer_run", "codegen_owned", "registered", "codegen_only"}


def test_construction_categories_are_known_manifest_categories() -> None:
    manifest_categories = {str(entry.get("category", "")) for entry in load_node_manifest()}
    assert CONSTRUCTION_CATEGORIES <= manifest_categories


def test_builder_registry_marks_codegen_owned_construction_nodes() -> None:
    construction_entries = {
        str(entry["type"]): entry
        for entry in load_node_manifest()
        if str(entry.get("category", "")) in CONSTRUCTION_CATEGORIES
    }
    codegen_owned = {node_type for node_type, entry in construction_entries.items() if entry.get("hasCodegen") is True}
    assert codegen_owned
    for node_type in codegen_owned:
        registration = builder_for(node_type)
        assert registration is not None
        if registration.status == "registered":
            continue
        assert registration.status == "codegen_owned"


def test_optimizer_construction_nodes_use_registered_builder_module() -> None:
    assert set(construction_node_types_for("optimizer")) >= {"adam_optimizer", "muon_optimizer", "sgd_optimizer"}
    assert registered_builder_node_types_for("optimizer") == (
        "adam_optimizer",
        "adamw_optimizer",
        "muon_optimizer",
        "sgd_optimizer",
        "shampoo_optimizer",
        "signsgd_optimizer",
        "soap_optimizer",
    )
    for node_type in (
        "adam_optimizer",
        "adamw_optimizer",
        "sgd_optimizer",
        "signsgd_optimizer",
        "muon_optimizer",
        "shampoo_optimizer",
        "soap_optimizer",
    ):
        registration = builder_for(node_type)
        assert registration is not None
        assert registration.category == "optimizer"
        assert registration.status == "registered"
        assert registration.implementation == "comfy_research.engine.optimizers.optimizer_builders"


def test_primary_loss_construction_nodes_use_registered_builder_module() -> None:
    assert set(construction_node_types_for("loss")) >= {
        "binary_cross_entropy_with_logits_loss",
        "cross_entropy_loss",
        "diffusion_mse_loss",
        "mse_loss",
    }
    assert registered_builder_node_types_for("loss") == (
        "binary_cross_entropy_with_logits_loss",
        "cross_entropy_loss",
        "diffusion_mse_loss",
        "mse_loss",
    )
    for node_type in (
        "binary_cross_entropy_with_logits_loss",
        "cross_entropy_loss",
        "diffusion_mse_loss",
        "mse_loss",
    ):
        registration = builder_for(node_type)
        assert registration is not None
        assert registration.category == "loss"
        assert registration.status == "registered"
        assert registration.implementation == "comfy_research.engine.losses.loss_builders"


def test_vision_model_construction_nodes_use_registered_builder_module() -> None:
    registered_models = {
        "afno_lite_spatiotemporal_model",
        "attention_only_model",
        "crl_residual_mlp",
        "diagonal_ssm_token_model",
        "diffusion_score_model",
        "gated_mlp_model",
        "gated_mlp_token_model",
        "hyena_like_conv_model",
        "kan_model",
        "keskar_c1_c2_cnn_model",
        "linear_attention_model",
        "mlp_model",
        "mlp_token_model",
        "moe_mlp_model",
        "moe_mlp_token_model",
        "mpp_spatiotemporal_model",
        "numeric_hyena_model",
        "numeric_transformer_model",
        "resnet_model",
        "residual_ln_model",
        "rwkv_time_mix_token_model",
        "slot_attention_token_model",
        "small_inception_cifar_model",
        "transformer_multi_token_model",
        "transformer_token_model",
        "vgg11_cifar_model",
        "vit_model",
    }
    assert set(construction_node_types_for("model")) >= registered_models
    assert set(registered_builder_node_types_for("model")) >= registered_models
    for node_type in registered_models:
        registration = builder_for(node_type)
        assert registration is not None
        assert registration.category == "model"
        assert registration.status == "registered"
        assert registration.implementation == "comfy_research.engine.models.model_builders"


def test_trainer_registered_model_whitelist_derives_from_builder_registry() -> None:
    assert set(registered_trainer_model_node_types()) == set(registered_builder_node_types_for("model"))


def test_trainer_dataset_whitelist_derives_from_construction_manifest_minus_auxiliary_nodes() -> None:
    trainer_datasets = set(registered_trainer_dataset_node_types())
    assert trainer_datasets == set(construction_node_types_for("dataset")) - {
        "crl_env_config",
        "input_sampler",
        "random_input_distribution",
    }
    assert {"linear_dataset", "memorization_a_dataset", "mnist_dataset"} <= trainer_datasets


def test_weight_export_model_whitelist_derives_from_builder_registry() -> None:
    assert set(registered_weight_export_model_node_types()) == set(registered_builder_node_types_for("model"))


def test_activation_model_whitelist_derives_from_capability() -> None:
    assert set(registered_activation_model_node_types()) == {
        "crl_residual_mlp",
        "gated_mlp_model",
        "kan_model",
        "mlp_model",
        "moe_mlp_model",
        "residual_ln_model",
    }
