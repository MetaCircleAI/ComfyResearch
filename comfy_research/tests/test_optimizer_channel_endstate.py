"""Optimizer definition, provider, and handwritten-exception contracts."""
from __future__ import annotations

from pathlib import Path

from comfy_research.tests.test_optimizer_channel_ledger import (
    EXPECTED_OPTIMIZER_BUILDER_KINDS,
    EXPECTED_OPTIMIZER_NODE_KINDS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_supported_optimizers_have_definitions() -> None:
    from comfy_research.nodes.registry import load_definitions, optimizer_def_types

    load_definitions()
    assert optimizer_def_types() == EXPECTED_OPTIMIZER_NODE_KINDS


def test_hand_builder_table_empty_providers_exactly_seven() -> None:
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
    from comfy_research.engine.optimizers.optimizer_builders import OPTIMIZER_BUILDERS
    from comfy_research.nodes.registry import optimizer_defs_builders

    assert OPTIMIZER_BUILDERS == {}
    provs = frozenset(optimizer_defs_builders())
    assert provs == EXPECTED_OPTIMIZER_BUILDER_KINDS
    assert provs == frozenset(registered_builder_node_types_for("optimizer"))


def test_intentionally_hand_surfaces_enumerated() -> None:
    """收官后的手写残留 = 枚举的例外。"""
    # (a) md→config 塑形阶梯(builder 表已 provider 化,阶梯不塌)
    stage = (ROOT / "comfy_research" / "engine" / "trainer" / "prepare_finalize.py").read_text()
    assert "def build_optimizer_stage" in stage
    assert "OptimizerBuildConfig(" in stage
    # (b) mup 双路径去重结构(同一节点两 handle 轴只收一次)
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    assert "mupLrAxisSources" in sweep
    assert '!mupLrAxisSources.has(mupLrSchedId)' in sweep
    # (c) lr/mup handle 归一化 + canvas/auto 行为表(canvas/agent 波收编前不动;
    # 钉实际类型覆盖,不是文件存在性)
    norm = (ROOT / "frontend" / "src" / "graph" / "normalizeOptimizerLrEdges.ts").read_text()
    assert '"lr_schedule"' in norm and '"mup_lr_schedule"' in norm
    auto_tune = (ROOT / "frontend" / "src" / "graph" / "autoTuneAxisSuggestions.ts").read_text()
    auto_layout = (ROOT / "frontend" / "src" / "graph" / "graphAutoLayout.ts").read_text()
    for t in ("sgd_optimizer", "signsgd_optimizer", "lr_schedule", "mup_lr_schedule"):
        assert f'"{t}"' in auto_tune, t
        assert f'"{t}"' in auto_layout, t
    # 搜索描述文案面(原钉 AddNodeSearchModal 的 NODE_HINTS): 起 NODE_HINTS
    # 清空,文案单源在 def hint——改钉 registry resolved 面存续。
    from comfy_research.nodes.registry import OPTIMIZER_DEFS, load_definitions

    load_definitions()
    assert any((d.hint or "") for d in OPTIMIZER_DEFS.values()) or any(
        "optimizer" in (d.label or "").lower() for d in OPTIMIZER_DEFS.values()
    )
    self_driving = (ROOT / "frontend" / "src" / "graph" / "selfDrivingGraph.ts").read_text()
    assert "optimizer" in self_driving
    # 起连接规则住 graph/connectionRules.ts
    rules = (ROOT / "frontend" / "src" / "graph" / "connectionRules.ts").read_text()
    assert '"optimizer"' in rules  # 连接规则/handle 面存续(细则 canvas 波接管)
    # (d) SchemaNode 消费面:三面从 NODE_SPEC_REGISTRY 读(泛型组件本体 stay)
    schema_node = (ROOT / "frontend" / "src" / "components" / "nodes" / "SchemaNode.tsx").read_text()
    assert "NODE_SPEC_REGISTRY" in schema_node and "spec.specCode" in schema_node


def test_info_content_lives_in_defs_now() -> None:
    """optimizerNodeInfoContent.ts 已删;info 文案在 def(python 持真相)。"""
    assert not (ROOT / "frontend" / "src" / "components" / "nodes" / "optimizerNodeInfoContent.ts").exists()
    from comfy_research.nodes.registry import OPTIMIZER_DEFS, load_definitions

    load_definitions()
    for t in EXPECTED_OPTIMIZER_BUILDER_KINDS:
        ui = OPTIMIZER_DEFS[t].ui
        assert ui is not None and ui.info_title and ui.info_text, t
    # 卫星二型无 ui 块(custom 组件自带 chrome)
    for t in ("lr_schedule", "mup_lr_schedule"):
        assert OPTIMIZER_DEFS[t].ui is None, t
