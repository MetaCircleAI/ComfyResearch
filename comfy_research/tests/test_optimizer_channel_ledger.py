"""optimizer 通道台账 guards(model  同款纪律)。

9 型:7 optimizer(SchemaNode 泛型 UI,OPTIMIZER_BUILDERS 成员)+
lr_schedule/mup_lr_schedule(custom 组件,零 provider 卫星)。
axes 台账钉的是 **9/9 全员有轴**(dataset 教义轴=fields,与 model 44 无轴相反)。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _optimizer_types_from_manifest() -> frozenset[str]:
    m = json.loads((ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text())
    return frozenset(e["type"] for e in m if e.get("category") == "optimizer")


EXPECTED_OPTIMIZER_BUILDER_KINDS: frozenset[str] = frozenset({
    "adam_optimizer", "adamw_optimizer", "sgd_optimizer", "signsgd_optimizer",
    "muon_optimizer", "shampoo_optimizer", "soap_optimizer",
})
# intentionally 有轴(hand else-if 链 + lr/mup 卫星双路径全 sweep; 增
# cyclic_lr_schedule——挂 optimizer lr_schedule handle,generic 轴收集覆盖)。
EXPECTED_OPTIMIZER_AXES_KINDS: frozenset[str] = frozenset({
    *EXPECTED_OPTIMIZER_BUILDER_KINDS, "lr_schedule", "mup_lr_schedule", "cyclic_lr_schedule",
})
# 通道全集 = 有轴 10 + cyclic_batch_schedule(trainer batch 卫星,
# 无 sweep 轴收集路径)。
EXPECTED_OPTIMIZER_NODE_KINDS: frozenset[str] = frozenset({
    *EXPECTED_OPTIMIZER_AXES_KINDS, "cyclic_batch_schedule",
})


def test_optimizer_ledger_matches_manifest() -> None:
    assert _optimizer_types_from_manifest() == EXPECTED_OPTIMIZER_NODE_KINDS
    assert len(EXPECTED_OPTIMIZER_NODE_KINDS) == 11


def test_builder_union_invariant() -> None:
    """设计 review (同款三方对账):hand ∪ providers ==
    registered_builder_node_types_for('optimizer');两侧不相交。"""
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
    from comfy_research.engine.optimizers.optimizer_builders import OPTIMIZER_BUILDERS, optimizer_builder_node_types
    from comfy_research.nodes.registry import optimizer_defs_builders

    hand = frozenset(OPTIMIZER_BUILDERS)
    provs = frozenset(optimizer_defs_builders())
    assert not (hand & provs), sorted(hand & provs)
    assert optimizer_builder_node_types() == hand | provs
    assert hand | provs == frozenset(registered_builder_node_types_for("optimizer"))
    assert hand | provs == EXPECTED_OPTIMIZER_BUILDER_KINDS


def test_lr_schedule_pair_never_registers_builders() -> None:
    """卫星二型(lr_schedule/mup_lr_schedule)不是 trainer optimizer socket 的
    builder——config 经 build_optimizer_stage 塑形阶梯(stay-hand)。"""
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.optimizers.optimizer_builders import OPTIMIZER_BUILDERS
    from comfy_research.nodes.registry import optimizer_defs_builders

    for t in ("lr_schedule", "mup_lr_schedule", "cyclic_lr_schedule", "cyclic_batch_schedule"):
        assert t not in OPTIMIZER_BUILDERS, t
        assert t not in optimizer_defs_builders(), t


def test_optimizer_defs_subset() -> None:
    from comfy_research.nodes.registry import optimizer_def_types

    assert optimizer_def_types() <= _optimizer_types_from_manifest()
