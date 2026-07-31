"""Model definition, provider, and handwritten-exception contracts.

The complete registry and intentionally handwritten paths are enumerated so
future cleanup cannot remove supported behavior accidentally.
"""
from __future__ import annotations

import json
from pathlib import Path

from comfy_research.tests.test_model_channel_ledger import (
    EXPECTED_MODEL_AXES_KINDS,
)

ROOT = Path(__file__).resolve().parents[2]


def _manifest_model_types() -> frozenset[str]:
    m = json.loads((ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text())
    return frozenset(e["type"] for e in m if e.get("category") == "model")


def test_all_manifest_models_have_definitions() -> None:
    from comfy_research.nodes.registry import load_definitions, model_def_types

    load_definitions()
    assert model_def_types() == _manifest_model_types(), (
        f"missing={sorted(_manifest_model_types() - model_def_types())}, "
        f"extra={sorted(model_def_types() - _manifest_model_types())}"
    )


def test_model_builders_are_registered_as_providers() -> None:
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.models.model_builders import MODEL_BUILDERS, model_builder_node_types
    from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
    from comfy_research.nodes.registry import model_defs_builders

    assert MODEL_BUILDERS == {}
    provs = frozenset(model_defs_builders())
    assert provs == frozenset(registered_builder_node_types_for("model"))


def test_intentionally_hand_surfaces_enumerated() -> None:
    """收官后的手写残留 = 枚举的例外(建档 §2/§8 documented exceptions)。"""
    # (a) atomic 链 literal switch(13 成员组装;provider 化留 documented exception)
    chain = (ROOT / "comfy_research" / "engine" / "models" / "atomic_layer_chain.py").read_text()
    assert "def build_atomic_layer_module" in chain
    # (b) ModelBuildContext 塑形阶梯(builder 表已 provider 化,阶梯不塌)
    assert (ROOT / "comfy_research" / "engine" / "trainer" / "prepare_build_vector.py").exists()
    assert (ROOT / "comfy_research" / "engine" / "trainer" / "prepare_build_token.py").exists()
    # (c) combined_model 的 canvas add 分支(template/options 线程)在 generic 排除表,
    # 且排除先于特殊分支、分支体确为 template/options 逻辑；仅检查裸
    # 'nodeType === "combined_model"' 会被无关路径满足。
    canvas = (ROOT / "frontend" / "src" / "components" / "ResearchCanvas.tsx").read_text()
    excl = canvas.index('nodeType !== "combined_model"')
    branch = canvas.index('if (nodeType === "combined_model")', excl)
    body = canvas[branch:branch + 2000]
    assert "combinedModelTemplateId" in body and "defaultCombinedModelData" in body
    # (d) canvas 连接/strip/init 三面 hand 列表(起连接规则住
    # graph/connectionRules.ts——hand 面存续断言跨两文件联合文本)
    rules = (ROOT / "frontend" / "src" / "graph" / "connectionRules.ts").read_text()
    surface = canvas + rules
    for name in ("ATOMIC_LAYER_NODE_TYPES", "FULL_MODEL_STRIP_NODE_TYPES", "modelInitializationTargetTypes"):
        assert name in surface, name
    # (e) sweep 卫星:crl_residual_mlp 的 activation 不 sweep 是 def 侧 sweepable=False,
    # 不是 allowlist 缺席——allowlist 恒等台账在 ledger，此处钉 def 旋钮
    from comfy_research.nodes.registry import MODEL_DEFS, load_definitions

    load_definitions()
    crl = MODEL_DEFS["crl_residual_mlp"]
    act = next(f for f in crl.fields if f.key == "activation")
    assert act.sweepable is False
    # (f) trainSeriesPlan 前缀表(assignments 命名)保持 hand,含已迁 model 型
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    assert 'crl_residual_mlp: "crl_model"' in sweep
    assert 'diffusion_score_model: "model"' in sweep


def test_spawn_defaults_only_where_hand_had_state_keys() -> None:
    """spawn_defaults 白名单恰等(防扩散成第二事实源)。"""
    from comfy_research.nodes.registry import MODEL_DEFS, load_definitions

    load_definitions()
    with_spawn = {t for t, d in MODEL_DEFS.items() if d.spawn_defaults is not None}
    assert with_spawn == {
        "tensor_add", "tensor_constant", "tensor_linspace",
        "fake_tensor", "elementwise_transform", "flatten",
    }


def test_axes_ledger_kinds_all_have_defs() -> None:
    from comfy_research.nodes.registry import load_definitions, model_def_types

    load_definitions()
    assert EXPECTED_MODEL_AXES_KINDS <= model_def_types()
