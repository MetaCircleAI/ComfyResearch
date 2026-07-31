"""Initialization definition and provider contracts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_INITIALIZATION_KINDS = frozenset({
    "idnns_initialization",
    "mup_initialization",
    "rank_aligned_initialization",
    "saxe_initialization",
    "symmetrized_mlp_init",
})


def test_all_supported_initializations_have_definitions() -> None:
    from comfy_research.nodes.registry import initialization_def_types, load_definitions

    load_definitions()
    assert initialization_def_types() == EXPECTED_INITIALIZATION_KINDS


def test_defs_anchor_committed_manifest() -> None:
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_initialization

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(
        (ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text(encoding="utf-8")
    )}
    for t, d in sorted(registry.INITIALIZATION_DEFS.items()):
        got = def_to_entry_initialization(d)
        assert got == committed[t], t
        assert list(got) == list(committed[t]), t


def test_intentionally_hand_surfaces_enumerated() -> None:
    """init socket 三面零接触(设计声明;canvas/trainer 波前不动)。"""
    # 起连接规则住 graph/connectionRules.ts——按联合文本断言
    canvas = (ROOT / "frontend" / "src" / "components" / "ResearchCanvas.tsx").read_text(encoding="utf-8")
    rules = (ROOT / "frontend" / "src" / "graph" / "connectionRules.ts").read_text(encoding="utf-8")
    surface = canvas + rules
    assert "modelInitializationTargetTypes" in surface
    for t in EXPECTED_INITIALIZATION_KINDS:
        assert f'"{t}"' in surface, t  # 连接规则/supported 集按型存续
    # apply 分派在 trainer runtime(server 侧)存续
    engine_files = (ROOT / "comfy_research" / "engine").rglob("*.py")
    assert any(
        "mup_initialization" in path.read_text(encoding="utf-8")
        for path in engine_files
    ), "apply dispatch must survive in engine"


def test_zero_axes_pin() -> None:
    """三型无 sweep 分派点——generated 入册后 collectAxes 无路径可达(源契约)。"""
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text(encoding="utf-8")
    for t in EXPECTED_INITIALIZATION_KINDS:
        assert f'GENERATED_NODE_SPECS[String(n.type)] && "{t}"' not in sweep
        assert f"axesFor{t}" not in sweep
    # 前缀表(assignments 命名)保持 hand
    assert 'mup_initialization: "mup_init"' in sweep
