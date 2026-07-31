"""Dataset 通道台账 guard。

总账语义(镜像 observable 台账,test_trainer_pkg_boundaries):
- EXPECTED_DATASET_NODE_KINDS = 47 型全集(迁移期恒定;defs 集单调增、legacy 集单调减,
  并集经生成器不相交 guard 恒等于总账)。
- EXPECTED_DATASET_PREVIEW_KINDS = 迁移前 dataset_tensor.py 有 preview 覆盖的 type 集
  (capability 臂展开 + literal elif 实抄)。preview provider 集 ⊆ 此集，缺口不虚增。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_DATASET_NODE_KINDS: frozenset[str] = frozenset({
    "biography_lm_dataset", "cifar10_dataset",  # paper-reproduction datasets
    "advection_dataset", "bigram_low_rank_dataset",
    "circle_random_walk_dataset", "circular_motion_dataset",
    "cogs_dataset", "crl_env_config", "dataset_mixer", "dataset_mixer_b",
    "diffusion_pde_dataset", "dyck_dataset", "formal_language_suite_dataset",
    "gaussian_blob_dataset", "hole_counting_dataset", "in_context_associative_recall_dataset",
    "input_sampler", "information_bottleneck_dataset", "kepler_2d_dataset", "linear_dataset",
    "listops_dataset", "memorization_a_dataset", "memorization_b_dataset", "mnist_dataset",
    "modular_addition_dataset", "multi_hop_fact_chain_dataset", "ngram_language_dataset",
    "paper_classification_dataset",
    "pcfg_dataset", "phi1_style_dataset", "random_input_distribution", "random_noise_dataset",
    "reaction_diffusion_dataset", "relation_tuple_dataset", "scan_dataset",
    "shape_world_dataset", "symbolic_func_dataset", "synthetic_playground_dataset",
    "teacher_dataset", "tinyshakespeare_lm_dataset", "tinystories_dataset", "token_prediction_dataset",
    "uniform_linear_motion_dataset", "unigram_dataset",
})

# 迁移前 preview 覆盖面(dataset_tensor.py 实抄):capability 臂(direct_arrays 5 +
# PDE 3)+ literal elif 21;ai4science 三型经 alias 走 linear_dataset
# 分支,算 linear 的覆盖,不单列。
EXPECTED_DATASET_PREVIEW_KINDS: frozenset[str] = frozenset({
    # dataset_tensor_direct_arrays capability
    "linear_dataset", "memorization_a_dataset", "memorization_b_dataset",
    "random_noise_dataset", "symbolic_func_dataset",
    # pde_field_dataset capability
    "advection_dataset", "diffusion_pde_dataset", "reaction_diffusion_dataset",
    # literal elif branches
    "token_prediction_dataset", "circle_random_walk_dataset", "circular_motion_dataset",
    "bigram_low_rank_dataset", "in_context_associative_recall_dataset",
    "uniform_linear_motion_dataset", "modular_addition_dataset", "teacher_dataset",
    "pcfg_dataset", "dyck_dataset", "ngram_language_dataset",
    "formal_language_suite_dataset", "scan_dataset", "cogs_dataset", "listops_dataset",
    "tinystories_dataset", "phi1_style_dataset", "biography_lm_dataset",
    "relation_tuple_dataset", "synthetic_playground_dataset", "multi_hop_fact_chain_dataset",
    "tinyshakespeare_lm_dataset",
})


def test_dataset_ledger_matches_manifest() -> None:
    """总账 == manifest 中 category=dataset 的全集(总账漂移 = 有人加/删了 dataset node,
    必须显式改本文件)。"""
    manifest = json.loads((ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text())
    actual = {e["type"] for e in manifest if e.get("category") == "dataset"}
    assert actual == EXPECTED_DATASET_NODE_KINDS, (
        f"missing={sorted(EXPECTED_DATASET_NODE_KINDS - actual)}, extra={sorted(actual - EXPECTED_DATASET_NODE_KINDS)}"
    )


def test_dataset_defs_subset_of_ledger_and_disjoint_from_legacy() -> None:
    from comfy_research.nodes.registry import dataset_def_types

    defs = dataset_def_types()
    assert defs <= EXPECTED_DATASET_NODE_KINDS, sorted(defs - EXPECTED_DATASET_NODE_KINDS)


def test_dataset_preview_providers_subset_of_preview_ledger() -> None:
    from comfy_research.nodes.registry import dataset_defs_previews

    providers = set(dataset_defs_previews())
    assert providers <= EXPECTED_DATASET_PREVIEW_KINDS, (
        f"preview provider(s) outside the pre-migration coverage set (缺口虚增): "
        f"{sorted(providers - EXPECTED_DATASET_PREVIEW_KINDS)}"
    )


def test_migrated_preview_kinds_have_providers() -> None:
    """反向 guard:迁移了原本有 preview 覆盖的 dataset 却忘了注册
    preview provider → FATAL(否则该 node 的 preview 静默落回已删除的 literal 分支)。"""
    from comfy_research.nodes.registry import dataset_def_types, dataset_defs_previews

    migrated_preview = dataset_def_types() & EXPECTED_DATASET_PREVIEW_KINDS
    providers = set(dataset_defs_previews())
    assert migrated_preview <= providers, (
        f"migrated dataset(s) with pre-existing preview coverage but no preview provider: "
        f"{sorted(migrated_preview - providers)}"
    )


def test_preview_kinds_subset_of_node_kinds() -> None:
    assert EXPECTED_DATASET_PREVIEW_KINDS <= EXPECTED_DATASET_NODE_KINDS


# 迁移后必须有 materializer 的 type 集;随批显式扩——
# "只迁 schema 落回已删 arm" 在此 FATAL。auxiliary node(input_sampler 等)
# 不在此集,不被强制。
EXPECTED_DATASET_MATERIALIZE_KINDS: frozenset[str] = frozenset({
    "advection_dataset", "diffusion_pde_dataset", "reaction_diffusion_dataset",
    # linear-like 家族(memorization_a/b 刻意不在——其 materialize 是
    # trainer_task 条件分支,留 hand documented exception;ai4science 三型
    # alias-only,不注册 provider)。
    "linear_dataset", "random_noise_dataset",
    # regression 三型(dense-hook 之后的无任务条件 arm 原体)。token 系与
    # vision 刻意不在(stay-hand guard 另钉);crl_env_config 无 materialize 路径。
    "kepler_2d_dataset", "uniform_linear_motion_dataset", "symbolic_func_dataset",
    "information_bottleneck_dataset",
    "paper_classification_dataset",
})


def test_migrated_materializable_kinds_have_providers() -> None:
    from comfy_research.nodes.registry import dataset_def_types, dataset_defs_materializers

    migrated = dataset_def_types() & EXPECTED_DATASET_MATERIALIZE_KINDS
    providers = set(dataset_defs_materializers())
    assert migrated <= providers, (
        f"migrated materializable dataset(s) without materializer provider: {sorted(migrated - providers)}"
    )
    assert providers <= EXPECTED_DATASET_MATERIALIZE_KINDS, (
        f"materializer provider(s) outside the declared materializable set: "
        f"{sorted(providers - EXPECTED_DATASET_MATERIALIZE_KINDS)}"
    )
