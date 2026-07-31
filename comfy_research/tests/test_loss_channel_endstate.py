"""Loss definition, provider, and recorder contracts."""
from __future__ import annotations

from pathlib import Path

from comfy_research.tests.test_loss_channel_ledger import (
    EXPECTED_LOSS_NODE_KINDS,
    EXPECTED_LOSS_PRIMARY_KINDS,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_7_losses_in_defs() -> None:
    """kan_reg 走 ObservableDef(NODE_DEFS),其余 6 型 LOSS_DEFS——并集 == 台账。"""
    from comfy_research.nodes.registry import NODE_DEFS, load_definitions, loss_def_types

    load_definitions()
    assert loss_def_types() | ({"kan_reg"} & set(NODE_DEFS)) == EXPECTED_LOSS_NODE_KINDS
    assert NODE_DEFS["kan_reg"].category == "loss"


def test_hand_tables_empty_providers_exact() -> None:
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.losses.loss_builders import LOSS_CRITERION_BUILDERS
    from comfy_research.engine.trainer.recorder import _HAND_WRITTEN_HANDLERS, OBSERVABLE_RECORD_HANDLERS
    from comfy_research.nodes.registry import KNOWN_NODEDEF_MIGRATION_DEFERRED, loss_defs_criteria

    assert LOSS_CRITERION_BUILDERS == {}
    assert frozenset(loss_defs_criteria()) == EXPECTED_LOSS_PRIMARY_KINDS
    # legacy 遗留 recorder hand 债务清零(kan_reg 收编)
    assert _HAND_WRITTEN_HANDLERS == {}
    assert KNOWN_NODEDEF_MIGRATION_DEFERRED == {}
    assert "kan_reg" in OBSERVABLE_RECORD_HANDLERS  # provider 侧存活


def test_intentionally_hand_surfaces_enumerated() -> None:
    """loss 收官后的手写残留 = 枚举的例外。"""
    # (a) loss_terms 求和面(_kan_reg_loss_term/权重正则/l2_projection runtime)
    lt = (ROOT / "comfy_research" / "engine" / "trainer" / "loss_terms.py").read_text()
    assert "_kan_reg_loss_term" in lt
    # (b) LOSS_SWEEP_ALLOWLIST 六型恒等 + kan_reg 排除(sweep 门源契约)
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    import re

    m = re.search(r"const LOSS_SWEEP_ALLOWLIST = new Set\(\[(.*?)\]\);", sweep, re.S)
    assert m
    ts_set = frozenset(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    assert ts_set == EXPECTED_LOSS_NODE_KINDS - {"kan_reg"}
    # (c) recorder 分派结构（OBSERVABLE_RECORD_HANDLERS 并集 + record() 查表）存续
    rec = (ROOT / "comfy_research" / "engine" / "trainer" / "recorder.py").read_text()
    assert "**defs_recorders()" in rec
