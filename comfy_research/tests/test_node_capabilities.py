from pathlib import Path

from comfy_research.generated.node_capabilities import (
    NODE_CAPABILITIES,
    capabilities_for,
    has_capability,
    node_types_with_capability,
)
from comfy_research.generated.node_manifest import load_node_manifest


def test_generated_capabilities_match_manifest_family_tags() -> None:
    expected = {
        str(entry["type"]): frozenset(str(capability) for capability in entry.get("family", []))
        for entry in load_node_manifest()
        if entry.get("family")
    }
    assert NODE_CAPABILITIES == expected


def test_observable_capability_helpers() -> None:
    observable_types = node_types_with_capability("observable")
    assert "observable_weight_l2" in observable_types
    assert "observable_activation_stats" in observable_types
    assert has_capability("observable_weight_l2", "observable")
    assert not has_capability("mlp_model", "observable")
    assert capabilities_for("missing_node_type") == frozenset()


def test_dataset_tensor_family_routes_use_capabilities() -> None:
    """路由必须走 capability 或 NodeDef 通道,不得回退 literal type 元组。
    PDE 两个 capability 分支已迁 NodeDef preview providers
    (dataset_defs_previews 的 generated-first seam),故只断言 seam 存在。"""
    source = Path("comfy_research/api/dataset_tensor.py").read_text(encoding="utf-8")
    # direct-arrays capability elif 的最后用户(symbolic_func)
    # 迁走后该 elif 已删——capability 本身保留为 schema/能力元数据(manifest 字节),
    # /api/dataset_tensor 路由不再消费它。新不变式:generated-first provider 覆盖
    # 全部已迁 direct-array node。
    assert "dataset_defs_previews().get(ds_type)" in source
    assert 'has_capability(ds_type, "dataset_tensor_direct_arrays")' not in source
    assert '("linear_dataset", "random_noise_dataset", "memorization_a_dataset", "memorization_b_dataset", "symbolic_func_dataset")' not in source
    assert '("diffusion_pde_dataset", "reaction_diffusion_dataset", "advection_dataset")' not in source


def test_trainer_runner_routes_use_capability() -> None:
    for path in (
        Path("comfy_research/api/predict.py"),
        Path("comfy_research/engine/analysis/activation_collect.py"),
        Path("comfy_research/engine/runs/train_sweep.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert '"trainer_runner"' in source
        assert "(NodeKind.trainer, NodeKind.crl_trainer)" not in source
        assert "(str(NodeKind.trainer), str(NodeKind.crl_trainer))" not in source


def test_predict_loss_selection_capability_sets_match_legacy_lists() -> None:
    assert set(node_types_with_capability("diffusion_loss_model")) == {
        "diffusion_score_model",
        "unet_ddpm_model",
    }
    assert set(node_types_with_capability("token_model")) == {
        "attention_only_model",
        "diagonal_ssm_token_model",
        "gated_mlp_token_model",
        "hyena_like_conv_model",
        "linear_attention_model",
        "mlp_token_model",
        "moe_mlp_token_model",
        "rwkv_time_mix_token_model",
        "slot_attention_token_model",
        "transformer_multi_token_model",
        "transformer_token_model",
    }
    assert set(node_types_with_capability("vision_model")) == {
        "keskar_c1_c2_cnn_model",
        "resnet_model",
        "small_inception_cifar_model",
        "vgg11_cifar_model",
        "vit_model",
    }
    assert set(node_types_with_capability("vision_dataset")) == {
        "cifar10_dataset",
        "gaussian_blob_dataset",
        "hole_counting_dataset",
        "mnist_dataset",
        "shape_world_dataset",
    }
    assert set(node_types_with_capability("pde_field_dataset")) == {
        "advection_dataset",
        "diffusion_pde_dataset",
        "reaction_diffusion_dataset",
    }
    assert set(node_types_with_capability("toy_language_token_dataset")) == {
        "biography_lm_dataset",
        "cogs_dataset",
        "dyck_dataset",
        "formal_language_suite_dataset",
        "listops_dataset",
        "multi_hop_fact_chain_dataset",
        "ngram_language_dataset",
        "pcfg_dataset",
        "phi1_style_dataset",
        "relation_tuple_dataset",
        "scan_dataset",
        "synthetic_playground_dataset",
        "tinyshakespeare_lm_dataset",
        "tinystories_dataset",
    }
    assert set(node_types_with_capability("token_classification_dataset")) == {
        "bigram_low_rank_dataset",
        "biography_lm_dataset",
        "circle_random_walk_dataset",
        "circular_motion_dataset",
        "cogs_dataset",
        "dyck_dataset",
        "formal_language_suite_dataset",
        "in_context_associative_recall_dataset",
        "listops_dataset",
        "memorization_b_dataset",
        "modular_addition_dataset",
        "multi_hop_fact_chain_dataset",
        "ngram_language_dataset",
        "pcfg_dataset",
        "phi1_style_dataset",
        "relation_tuple_dataset",
        "scan_dataset",
        "synthetic_playground_dataset",
        "tinyshakespeare_lm_dataset",
        "tinystories_dataset",
        "token_prediction_dataset",
        "unigram_dataset",
    }
    assert set(node_types_with_capability("linear_like_dataset")) == {
        "linear_dataset",
        "random_noise_dataset",
    }
    assert set(node_types_with_capability("dataset_tensor_direct_arrays")) == {
        "linear_dataset",
        "information_bottleneck_dataset",
        "memorization_a_dataset",
        "memorization_b_dataset",
        "random_noise_dataset",
        "symbolic_func_dataset",
    }
    assert set(node_types_with_capability("vector_regression_dataset")) == {
        "advection_dataset",
        "dataset_mixer",
        "dataset_mixer_b",
        "diffusion_pde_dataset",
        "kepler_2d_dataset",
        "linear_dataset",
        "memorization_b_dataset",
        "random_noise_dataset",
        "reaction_diffusion_dataset",
        "symbolic_func_dataset",
        "teacher_dataset",
        "uniform_linear_motion_dataset",
    }
    assert set(node_types_with_capability("diffusion_noise_dataset")) == {
        "advection_dataset",
        "cifar10_dataset",
        "dataset_mixer",
        "dataset_mixer_b",
        "diffusion_pde_dataset",
        "linear_dataset",
        "memorization_b_dataset",
        "random_noise_dataset",
        "reaction_diffusion_dataset",
        "symbolic_func_dataset",
    }
    assert set(node_types_with_capability("dataset_mixer")) == {
        "dataset_mixer",
        "dataset_mixer_b",
    }
    assert set(node_types_with_capability("text_heavy_toy_language_dataset")) == {
        "phi1_style_dataset",
        "tinyshakespeare_lm_dataset",
        "tinystories_dataset",
    }
    assert set(node_types_with_capability("memorization_dataset")) == {
        "information_bottleneck_dataset",
        "memorization_a_dataset",
        "memorization_b_dataset",
    }
    assert set(node_types_with_capability("mlp_family")) == {
        "gated_mlp_model",
        "mlp_model",
        "moe_mlp_model",
    }
    assert set(node_types_with_capability("moe_model")) == {
        "moe_mlp_model",
        "moe_mlp_token_model",
    }
    assert set(node_types_with_capability("activation_model")) == {
        "crl_residual_mlp",
        "gated_mlp_model",
        "kan_model",
        "mlp_model",
        "moe_mlp_model",
        "residual_ln_model",
    }
    assert set(node_types_with_capability("silu_default_activation_model")) == {
        "gated_mlp_model",
        "moe_mlp_model",
    }
    assert set(node_types_with_capability("mlp_token_family")) == {
        "gated_mlp_token_model",
        "mlp_token_model",
        "moe_mlp_token_model",
    }
    assert set(node_types_with_capability("vector_model")) == {
        "afno_lite_spatiotemporal_model",
        "gated_mlp_model",
        "kan_model",
        "mlp_model",
        "moe_mlp_model",
        "mpp_spatiotemporal_model",
        "numeric_hyena_model",
        "numeric_transformer_model",
    }
    assert set(node_types_with_capability("trainer_runner")) == {
        "crl_trainer",
        "trainer",
    }
    assert set(node_types_with_capability("observable_user_tensor_transform")) == {
        "effective_rank",
        "pca",
        "series_endpoint_gap",
        "statistics",
    }
    assert set(node_types_with_capability("observable_user_tensor_viz_display")) == {
        "tensor_viz_0d",
        "tensor_viz_1d",
        "tensor_viz_2d",
        "tensor_viz_general",
    }
    assert set(node_types_with_capability("observable_user_tensor_viz_anchor")) == {
        "tensor_viz_0d",
        "tensor_viz_general",
    }
    assert set(node_types_with_capability("activation_sample_dataset")) == {
        "information_bottleneck_dataset",
        "linear_dataset",
        "memorization_a_dataset",
        "memorization_b_dataset",
        "symbolic_func_dataset",
        "teacher_dataset",
    }
    assert set(node_types_with_capability("trainer_weight_regularizer_loss")) == {
        "l1_reg",
        "l2_reg",
    }
    assert set(node_types_with_capability("trainer_loss_socket_aux")) == {
        "l1_reg",
        "l2_projection",
        "l2_reg",
    }
    assert set(node_types_with_capability("trainer_primary_loss")) == {
        "binary_cross_entropy_with_logits_loss",
        "cross_entropy_loss",
        "diffusion_mse_loss",
        "mse_loss",
    }
    assert set(node_types_with_capability("trainer_loss_viz_spawn")) == {
        "binary_cross_entropy_with_logits_loss",
        "cross_entropy_loss",
        "mse_loss",
    }
    assert set(node_types_with_capability("canvas_tensor_source")) == {
        "tensor_constant",
        "tensor_linspace",
    }
    assert set(node_types_with_capability("canvas_trainer_model_source")) == {
        "absolute_pos_embed_layer",
        "activation_layer",
        "afno_encoder_block_layer",
        "afno_lite_spatiotemporal_model",
        "afno_patch_decode_layer",
        "afno_patch_embed_layer",
        "afno_spectral_mixer_layer",
        "attention_only_model",
        "combined_model",
        "diagonal_ssm_token_model",
        "diffusion_score_model",
        "embedding_layer",
        "gated_mlp_model",
        "gated_mlp_token_model",
        "hyena_like_conv_model",
        "kan_model",
        "keskar_c1_c2_cnn_model",
        "layer_norm_layer",
        "linear_attention_model",
        "linear_layer",
        "local_mixing_layer",
        "mlp_model",
        "mlp_token_model",
        "moe_mlp_model",
        "moe_mlp_token_model",
        "mpp_spatiotemporal_model",
        "numeric_hyena_model",
        "numeric_transformer_model",
        "residual_ln_model",
        "resnet_model",
        "rms_norm_layer",
        "rotary_embed_layer",
        "rwkv_time_mix_token_model",
        "small_inception_cifar_model",
        "slot_attention_token_model",
        "transformer_multi_token_model",
        "transformer_token_model",
        "unembedding_layer",
        "unet_ddpm_model",
        "vgg11_cifar_model",
        "vit_model",
    }
    assert set(node_types_with_capability("agent_text_context")) == {
        "comment",
        "hypothesis",
    }
    assert set(node_types_with_capability("optimizer_node")) == {
        "adam_optimizer",
        "adamw_optimizer",
        "muon_optimizer",
        "sgd_optimizer",
        "signsgd_optimizer",
        "shampoo_optimizer",
        "soap_optimizer",
    }
    assert set(node_types_with_capability("atomic_layer_model")) == {
        "absolute_pos_embed_layer",
        "activation_layer",
        "afno_encoder_block_layer",
        "afno_patch_decode_layer",
        "afno_patch_embed_layer",
        "afno_spectral_mixer_layer",
        "embedding_layer",
        "layer_norm_layer",
        "linear_layer",
        "local_mixing_layer",
        "rms_norm_layer",
        "rotary_embed_layer",
        "unembedding_layer",
    }


def test_atomic_layer_chain_types_derive_from_capability() -> None:
    source = Path("comfy_research/engine/models/atomic_layer_chain.py").read_text(encoding="utf-8")
    assert 'node_types_with_capability("atomic_layer_model")' in source
    assert "NodeKind.linear_layer,\n" not in source
    assert "NodeKind.afno_patch_embed_layer,\n" not in source


def test_canvas_tensor_chain_sources_derive_from_capability() -> None:
    source = Path("comfy_research/engine/analysis/canvas_tensor_chain.py").read_text(encoding="utf-8")
    assert 'node_types_with_capability("canvas_tensor_source")' in source


def test_backend_mlp_family_routes_use_capabilities() -> None:
    loop_source = Path("comfy_research/engine/models/model_loop_expand.py").read_text(encoding="utf-8")
    assert 'has_capability(parent.type, "mlp_family")' in loop_source
    assert "NodeKind.mlp_model,\n" not in loop_source
    assert "NodeKind.gated_mlp_model,\n" not in loop_source
    assert "NodeKind.moe_mlp_model,\n" not in loop_source

    activation_source = Path("comfy_research/engine/analysis/activation_collect.py").read_text(encoding="utf-8")
    assert 'has_capability(model_type, "silu_default_activation_model")' in activation_source
    assert "model_type in (NodeKind.gated_mlp_model, NodeKind.moe_mlp_model)" not in activation_source


def test_observable_user_path_routes_use_capabilities() -> None:
    source = Path("comfy_research/engine/analysis/observable_user_eval.py").read_text(encoding="utf-8")
    assert "node_types_with_capability(capability)" in source
    assert '_node_kinds_with_capability("observable_user_tensor_transform")' in source
    assert '_node_kinds_with_capability("observable_user_tensor_viz_display")' in source
    assert '_node_kinds_with_capability("observable_user_tensor_viz_anchor")' in source
    assert "NodeKind.pca,\n            NodeKind.statistics,\n            NodeKind.effective_rank" not in source
    assert "NodeKind.tensor_viz_0d,\n            NodeKind.tensor_viz_general" not in source


def test_activation_sample_dataset_routes_use_capability() -> None:
    for path in (
        Path("comfy_research/engine/analysis/activation_collect.py"),
        Path("comfy_research/engine/analysis/kan_visualize.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert 'has_capability(ds_node.type, "activation_sample_dataset")' in source
        assert "NodeKind.linear_dataset,\n        NodeKind.memorization_a_dataset" not in source
        assert "NodeKind.symbolic_func_dataset,\n        NodeKind.teacher_dataset" not in source


def test_trainer_loss_socket_aux_routes_use_capabilities() -> None:
    # Kind-sets moved to trainer/loss_terms.py; they must stay
    # capability-derived at their single source of definition.
    source = Path("comfy_research/engine/trainer/loss_terms.py").read_text(encoding="utf-8")
    assert 'node_types_with_capability("trainer_weight_regularizer_loss")' in source
    assert 'node_types_with_capability("trainer_loss_socket_aux")' in source
    assert "_WEIGHT_REG_TRAINER_LOSS_KINDS = frozenset({NodeKind.l1_reg, NodeKind.l2_reg})" not in source
    assert "_LOSS_SOCKET_AUX_KINDS = frozenset({NodeKind.l1_reg, NodeKind.l2_reg, NodeKind.l2_projection})" not in source
