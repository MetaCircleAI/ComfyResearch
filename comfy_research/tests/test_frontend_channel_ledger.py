"""Frontend node definitions and byte-level manifest contracts.

def_to_entry_frontend 必须与 committed manifest 逐键恒等（含键序）；
同时覆盖 defaults 的三态（None、空 tuple、非空 tuple）与 dict 值渲染。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "comfy_research" / "generated"

CORE_FRONTEND_NODES = ("comment", "graph_assist_failure_overlay", "hypothesis", "url_node")
ANALYSIS_FRONTEND_NODES = (
    "basic_calculator", "derivative_curve", "effective_rank",
    "model_weight_tensors", "pca", "prediction", "series_endpoint_gap", "shape_checker",
    "smoothing_curve", "statistics", "statistics2", "svd", "tensor_reader",
)
TENSOR_TABLE_FRONTEND_NODES = (
    "agent_trace_viz", "curve_annotator", "dimension_permutator", "regressor",
    "sweep_data_table", "table_viz", "tensor_selector", "tensor_slicing",
)
VISUALIZATION_FRONTEND_NODES = (
    "docking_pose_viz", "image_dataset_displayer", "interatomic_eval_viz",
    "protein_structure_comparison_viz", "protein_structure_displayer",
    "tensor_viz_0d", "tensor_viz_1d", "tensor_viz_2d", "tensor_viz_general",
    "tensor_viz_scatter", "training_visualization", "visualize_kan",
)
SPECIAL_FRONTEND_NODES = (
    "activation", "model_checkpoint", "observable_viz",
    "observable_viz_neuron_trajectory_2d",
)
PAPER_REPRO_FRONTEND_NODES = (
    "curve_series_table", "curve_series_viz", "metric_compare", "parametric_path_sampler",
)
DIFFUSION_REPRO = (
    "deterministic_diffusion_sampler", "observable_nearest_train_gl",
    "observable_paired_generation_similarity", "observable_rp_score_sscd",
)
LINEAR_MODE_CONNECTIVITY = ("observable_linear_interpolation_barrier", "observable_bezier_mode_connectivity")
REGISTERED_FRONTEND_TYPES = (
    CORE_FRONTEND_NODES
    + ANALYSIS_FRONTEND_NODES
    + TENSOR_TABLE_FRONTEND_NODES
    + VISUALIZATION_FRONTEND_NODES
    + SPECIAL_FRONTEND_NODES
    + PAPER_REPRO_FRONTEND_NODES
    + DIFFUSION_REPRO
    + LINEAR_MODE_CONNECTIVITY
)
# Nodes whose manifest entries intentionally include a hint.
HINTED = frozenset(REGISTERED_FRONTEND_TYPES) - frozenset(
    {"graph_assist_failure_overlay", "hypothesis",
     "prediction", "agent_trace_viz",
     "interatomic_eval_viz", "docking_pose_viz",
     "protein_structure_displayer", "protein_structure_comparison_viz",
     "observable_viz_neuron_trajectory_2d",
      # curve series 双件和 metric compare 无 hint(repro NODE_HINTS 无条目);sampler 有。
      "curve_series_table", "curve_series_viz", "metric_compare",
      "observable_nearest_train_gl", "observable_paired_generation_similarity", "observable_rp_score_sscd"}
)


def test_frontend_registry_count() -> None:
    """The registry exposes the complete supported frontend node set."""
    from comfy_research.nodes import registry

    assert len(registry.frontend_def_types()) == 51
    assert len(REGISTERED_FRONTEND_TYPES) == 51


def test_frontend_registry_types() -> None:
    from comfy_research.nodes import registry

    assert registry.frontend_def_types() == frozenset(REGISTERED_FRONTEND_TYPES)


def test_hinted_set_pinned() -> None:
    """每型 hint 有无与申报清单逐一对应(hint golden 恒等的 python 侧前提)。"""
    from comfy_research.nodes import registry

    registry.load_definitions()
    got = frozenset(t for t, d in registry.FRONTEND_DEFS.items() if d.hint is not None)
    assert got == HINTED, got.symmetric_difference(HINTED)


def test_pilot_entries_match_committed_manifest() -> None:
    """字节锚:def_to_entry_frontend 逐键(含键序)复刻 committed 条目。
    hint 为申报变化（comment/url_node 文案从 NODE_HINTS 迁入 def）。"""
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_frontend

    registry.load_definitions()
    committed = json.loads((GEN / "node_manifest.json").read_text())
    for t in REGISTERED_FRONTEND_TYPES:
        want = next(e for e in committed if e["type"] == t)
        got = def_to_entry_frontend(registry.FRONTEND_DEFS[t])
        assert got == want, t
        assert list(got) == list(want), t  # 键序一致(json 字节等价的前提)


def test_defaults_three_state_rendering() -> None:
    """defaults 三态：None → 无 defaults 键且 hasDefaults:false；() → defaults:{}；
    非空含 dict/None 值 → 有序全量渲染。"""
    from comfy_research.nodes.generate import _render_frontend_def_ts, def_to_entry_frontend
    from comfy_research.nodes.schema import FrontendNodeDef, FrontendSpec

    none_def = FrontendNodeDef(
        type="frontend__none_probe", label="P", category="internal",
        defaults=None, resizable=False, frontend=FrontendSpec(component_key="X"),
    )
    ts = _render_frontend_def_ts(none_def)
    assert "defaults" not in ts and "resizable: false," in ts
    assert def_to_entry_frontend(none_def)["hasDefaults"] is False
    assert def_to_entry_frontend(none_def)["resizable"] is False

    empty_def = FrontendNodeDef(
        type="frontend__empty_probe", label="P", category="visualization",
        defaults=(), frontend=FrontendSpec(component_key="X"),
    )
    ts = _render_frontend_def_ts(empty_def)
    assert "    defaults: {}," in ts and "resizable" not in ts
    assert def_to_entry_frontend(empty_def)["hasDefaults"] is True

    rich_def = FrontendNodeDef(
        type="frontend__rich_probe", label="P", category="analysis",
        defaults=(
            ("selectedTensorKey", ""),
            ("activationTensorCaches", {}),
            ("plotXParamKey", None),
            ("rows", []),
            ("nested", {"a": 1, "b": [2, 3]}),
            # list 套 dict(tensor_slicing.slices 实况)显式锁形。
            ("slices", [{"dimension": 0, "indices": "0"}]),
        ),
        frontend=FrontendSpec(component_key="X"),
    )
    ts = _render_frontend_def_ts(rich_def)
    assert '"activationTensorCaches": {}' in ts
    assert '"plotXParamKey": null' in ts
    assert '"rows": []' in ts
    assert '"nested": { "a": 1, "b": [2, 3] }' in ts
    assert '"slices": [{ "dimension": 0, "indices": "0" }]' in ts


def test_schema_guards() -> None:
    from comfy_research.nodes.schema import EnumField, FrontendNodeDef

    with pytest.raises(ValueError, match="category"):
        FrontendNodeDef(type="x", label="X", category="dataset")
    with pytest.raises(ValueError, match="fields require defaults"):
        FrontendNodeDef(
            type="x", label="X", category="analysis", defaults=None,
            fields=(EnumField(key="k", label="K", default=""),),
        )
    with pytest.raises(ValueError, match="mirror defaults"):
        FrontendNodeDef(
            type="x", label="X", category="analysis", defaults=(("k", "other"),),
            fields=(EnumField(key="k", label="K", default=""),),
        )
    with pytest.raises(ValueError, match="duplicate field keys"):
        FrontendNodeDef(
            type="x", label="X", category="analysis", defaults=(("k", ""),),
            fields=(EnumField(key="k", label="K", default=""), EnumField(key="k", label="K2", default="")),
        )


def test_registry_symmetry_frontend() -> None:
    """八表对称:frontend 型在其他任一注册函数处撞名必须 FATAL,反向同理。"""
    from comfy_research.nodes import registry
    from comfy_research.nodes.schema import DatasetDef, FrontendNodeDef, FrontendSpec

    registry.load_definitions()
    with pytest.raises(RuntimeError, match="duplicate"):
        registry.dataset_def(DatasetDef(type="comment", label="X", family=()))
    with pytest.raises(RuntimeError, match="duplicate"):
        registry.frontend_node_def(
            FrontendNodeDef(type="linear_dataset", label="X", category="analysis",
                            frontend=FrontendSpec(component_key="X"))
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        registry.frontend_node_def(
            FrontendNodeDef(type="comment", label="X", category="language",
                            frontend=FrontendSpec(component_key="X"))
        )
