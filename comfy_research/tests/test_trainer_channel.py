"""Trainer definition and provider contracts."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_TRAINER_KINDS = frozenset({"trainer", "crl_trainer"})


def test_all_trainers_in_defs() -> None:
    from comfy_research.nodes.registry import load_definitions, trainer_def_types

    load_definitions()
    assert trainer_def_types() == EXPECTED_TRAINER_KINDS


def test_defs_anchor_committed_manifest() -> None:
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_trainer

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(
        (ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text()
    )}
    for t, d in sorted(registry.TRAINER_DEFS.items()):
        got = def_to_entry_trainer(d)
        assert got == committed[t], t
        assert list(got) == list(committed[t]), t



def test_no_sweep_knob_regressions() -> None:
    """computeDevice sweepable=False ×2;boolean 字段天然无轴(def 侧不需旋钮)。"""
    from comfy_research.nodes.registry import TRAINER_DEFS, load_definitions

    load_definitions()
    for t in EXPECTED_TRAINER_KINDS:
        cd = next(f for f in TRAINER_DEFS[t].fields if f.key == "computeDevice")
        assert cd.sweepable is False, t
        assert cd.manifest_options is True, t  # spec-level options 入 manifest 实况


def test_intentionally_hand_surfaces_enumerated() -> None:
    """trainer runtime = prepare pipeline 本身(表驱动不适用,零 provider 是终态);
    run 按钮/进度 chrome 在 custom 组件。"""
    assert (ROOT / "comfy_research" / "engine" / "runs" / "trainer_run.py").exists()
    assert (ROOT / "comfy_research" / "engine" / "trainer" / "prepare_finalize.py").exists()
    sweep = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    # 两个 generated-first 分派点存活(crl 分支 + 尾部 trainer)
    assert sweep.count("axesFromGeneratedSpec(trainerId") >= 1
    assert 'axesFromGeneratedSpec(trainerNode.id, "trainer"' in sweep
