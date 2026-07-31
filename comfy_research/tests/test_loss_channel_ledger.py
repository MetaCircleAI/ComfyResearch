"""loss 通道台账 + criterion provider 等价测试。

七型三分群:3 primary criterion(providers)/3 loss-socket aux(零 provider,
loss_terms stay-hand)/kan_reg(L2 走 ObservableDef 收编)。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_LOSS_PRIMARY_KINDS = frozenset({
    "binary_cross_entropy_with_logits_loss",
    "cross_entropy_loss",
    "diffusion_mse_loss",
    "mse_loss",
})
EXPECTED_LOSS_NODE_KINDS = frozenset({
    *EXPECTED_LOSS_PRIMARY_KINDS, "l1_reg", "l2_reg", "l2_projection", "kan_reg",
})


def _loss_types_from_manifest() -> frozenset[str]:
    m = json.loads(
        (ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text(encoding="utf-8")
    )
    return frozenset(e["type"] for e in m if e.get("category") == "loss")


def test_loss_ledger_matches_manifest() -> None:
    assert _loss_types_from_manifest() == EXPECTED_LOSS_NODE_KINDS


def test_criterion_union_invariant() -> None:
    """设计 (三方对账):hand ∪ providers ==
    registered_builder_node_types_for('loss');两侧不相交。"""
    pytest.importorskip("torch")
    from comfy_research.engine.losses.loss_builders import LOSS_CRITERION_BUILDERS, primary_loss_builder_node_types
    from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
    from comfy_research.nodes.registry import loss_defs_criteria

    hand = frozenset(LOSS_CRITERION_BUILDERS)
    provs = frozenset(loss_defs_criteria())
    assert not (hand & provs), sorted(hand & provs)
    assert primary_loss_builder_node_types() == hand | provs
    assert hand | provs == frozenset(registered_builder_node_types_for("loss"))
    assert hand | provs == EXPECTED_LOSS_PRIMARY_KINDS


def test_aux_kinds_never_register_criteria() -> None:
    pytest.importorskip("torch")
    from comfy_research.engine.losses.loss_builders import LOSS_CRITERION_BUILDERS
    from comfy_research.nodes.registry import loss_defs_criteria

    for t in ("l1_reg", "l2_reg", "l2_projection", "kan_reg"):
        assert t not in LOSS_CRITERION_BUILDERS, t
        assert t not in loss_defs_criteria(), t


def test_all_loss_defs_anchor_committed_manifest() -> None:
    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_loss

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(
        (ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text(encoding="utf-8")
    )}
    assert registry.LOSS_DEFS
    for t, d in sorted(registry.LOSS_DEFS.items()):
        got = def_to_entry_loss(d)
        assert got == committed[t], t
        assert list(got) == list(committed[t]), t


@pytest.mark.parametrize("t,data,ctx_kw", [
    ("mse_loss", {"lossScale": 2.0, "lossMaskMode": "all"}, dict(trainer_task="regression", target_flat_dim=4)),
    ("mse_loss", {}, dict(trainer_task="diffusion_noise")),
    ("cross_entropy_loss", {"labelSmoothing": 0.1}, dict(trainer_task="token_classification")),
    ("cross_entropy_loss", {"lossScale": 1.5}, dict(trainer_task="regression", num_logits=7)),
    ("diffusion_mse_loss", {}, dict(trainer_task="diffusion_noise")),
])
def test_criterion_provider_full_path_equivalence(t, data, ctx_kw) -> None:
    """经 build_loss_criterion_for_node 全路径 ≡ engine 函数直调:同输入 forward
    数值相等(torch.manual_seed 双侧)。"""
    pytest.importorskip("torch")
    import torch

    from comfy_research.engine.losses import loss_builders as lb
    from comfy_research.engine.losses.loss_builders import LossCriterionContext, build_loss_criterion_for_node
    from comfy_research.schemas.graph import Node

    ctx = LossCriterionContext(**ctx_kw)
    got = build_loss_criterion_for_node(Node(id="l", type=t, data=dict(data)), ctx)
    want = getattr(lb, f"_build_{t}_criterion")(dict(data), ctx)
    assert type(got) is type(want)
    torch.manual_seed(0)
    if ctx_kw.get("trainer_task") == "token_classification":
        pred, target = torch.randn(3, 7), torch.randint(0, 7, (3,))
    elif ctx_kw.get("num_logits"):
        pred, target = torch.randn(3, 7), torch.randn(3, 7)
    else:
        dim = ctx_kw.get("target_flat_dim") or 4
        pred, target = torch.randn(3, dim), torch.randn(3, dim)
    assert torch.equal(got(pred, target), want(pred, target)), t
