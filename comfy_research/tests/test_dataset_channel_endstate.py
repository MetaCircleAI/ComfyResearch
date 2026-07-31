"""Dataset definition and dispatch contracts.

The tests pin the complete dataset registry and the intentionally specialized
dispatch paths that remain outside generic providers.
"""
from __future__ import annotations

from pathlib import Path

from comfy_research.tests.test_dataset_channel_ledger import (
    EXPECTED_DATASET_NODE_KINDS,
    EXPECTED_DATASET_PREVIEW_KINDS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_47_datasets_in_defs() -> None:
    from comfy_research.nodes.registry import dataset_def_types

    assert dataset_def_types() == EXPECTED_DATASET_NODE_KINDS, (
        f"missing={sorted(EXPECTED_DATASET_NODE_KINDS - dataset_def_types())}"
    )


def test_preview_providers_exactly_cover_supported_set() -> None:
    from comfy_research.nodes.registry import dataset_defs_previews

    assert set(dataset_defs_previews()) == EXPECTED_DATASET_PREVIEW_KINDS


def test_stay_hand_dispatches_stay_above_the_provider_hook() -> None:
    """源序契约——teacher 两处分派在 _materialize_dense_families 之前、
    mixer early-return 在 provider hook 之前。防"helpful provider"与 hook 位移回归。"""
    src = (ROOT / "comfy_research" / "engine" / "trainer" / "dataset_materialize.py").read_text()
    hook = src.index("dataset_defs_materializers().get(ds_train.type)")
    mixer = src.index("_DATASET_MIXER_TYPES:")
    assert mixer < hook, "mixer early-return must stay above the provider hook"
    dense = src.index("def _materialize_dense_families")
    # atomic 与 vector 两处 teacher 分派都必须在 dense-families 之前
    teacher_dispatches = [i for i in range(len(src)) if src.startswith("NodeKind.teacher_dataset", i)]
    pre_dense = [i for i in teacher_dispatches if i < dense]
    assert len(pre_dense) >= 2, f"expected both branch dispatches above dense-families, got {len(pre_dense)}"
    assert src.count("_materialize_teacher(") >= 3  # def + atomic + vector 两分派


def test_intentionally_hand_surfaces_enumerated() -> None:
    """Specialized handwritten paths remain explicitly enumerated."""
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    # (a) teacher sweep 排除 + 卫星轴 fns
    assert 'n.type !== "teacher_dataset"' in sweep
    assert "axesForInputSampler" in sweep and "axesForRandomInputDistribution" in sweep
    assert "axesForCrlEnvConfig" in sweep
    # (b) mixer streaming 表
    runtime = (ROOT / "comfy_research" / "engine" / "datasets" / "dataset_runtime.py").read_text()
    for name in ("_declared_dataset_mixer_split_sizes", "_MIXER_B_INPUT_DRAWERS", "_MIXER_B_OUTPUT_SAMPLERS"):
        assert name in runtime, name
    # (c) auxiliary 排除集
    nbr = (ROOT / "comfy_research" / "engine" / "node_builder_registry.py").read_text()
    for t in ("input_sampler", "random_input_distribution", "crl_env_config"):
        assert t in nbr, t
    # (d) canvas 连接/来源列表(canvas 波收编前不动)
    canvas = (ROOT / "frontend" / "src" / "components" / "ResearchCanvas.tsx").read_text()
    assert "UNIFIED_DATASET_SOURCE_NODE_TYPES" in canvas
    assert 'nodeType !== "linear_dataset"' in canvas and 'nodeType !== "symbolic_func_dataset"' in canvas


def test_no_provider_families_pinned() -> None:
    from comfy_research.nodes.registry import dataset_defs_materializers, dataset_defs_previews

    mats = set(dataset_defs_materializers())
    prevs = set(dataset_defs_previews())
    for t in ("teacher_dataset", "dataset_mixer", "dataset_mixer_b", "input_sampler",
              "random_input_distribution", "crl_env_config"):
        assert t not in mats, t
    for t in ("dataset_mixer", "dataset_mixer_b", "input_sampler", "random_input_distribution",
              "crl_env_config", "unigram_dataset", "kepler_2d_dataset"):
        assert t not in prevs, t
