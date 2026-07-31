"""optimizer builder provider 等价测试(param_groups 超参口径)。"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

OPTIMIZER_TYPES = ("adam_optimizer", "adamw_optimizer", "sgd_optimizer", "signsgd_optimizer",
                   "muon_optimizer", "shampoo_optimizer", "soap_optimizer")


def test_all_optimizer_defs_anchor_committed_manifest() -> None:
    """通用锚点(dataset/model 同款):已迁 optimizer def 逐键+键序复刻 manifest。"""
    import json
    from pathlib import Path

    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_optimizer

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(
        (Path(__file__).resolve().parents[2] / "comfy_research" / "generated" / "node_manifest.json").read_text()
    )}
    assert registry.OPTIMIZER_DEFS, "no optimizer defs registered"
    for t, d in sorted(registry.OPTIMIZER_DEFS.items()):
        got = def_to_entry_optimizer(d)
        want = committed[t]
        assert got == want, t
        assert list(got) == list(want), t


def _hyper(opt) -> list[dict]:
    return [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]


@pytest.mark.parametrize("t", OPTIMIZER_TYPES)
def test_builder_provider_full_path_equivalence(t: str) -> None:
    """经 build_optimizer_for_node 全路径(generated-first hook)≡ engine 函数直调:
    同 model/config → 类型相等 + param_groups 超参逐组相等。"""
    import torch

    from comfy_research.engine.optimizers import optimizer_builders as ob
    from comfy_research.engine.optimizers.optimizer_builders import OptimizerBuildConfig, build_optimizer_for_node
    from comfy_research.schemas.graph import Node

    cfg = OptimizerBuildConfig(
        lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, momentum=0.5, weight_decay=0.01,
        muon_ns_steps=5, precondition_frequency=10, max_preconditioner_dim=1024,
        use_mup_lr_schedule=False, mup_embed_lr_mult=1.0, mup_hidden_lr_mult=1.0, mup_output_lr_mult=1.0,
    )
    torch.manual_seed(0)
    model_a = torch.nn.Linear(4, 2)
    got = build_optimizer_for_node(Node(id="o", type=t, data={}), model_a, cfg)
    fn = getattr(ob, f"_build_{t}")
    want = fn(model_a, cfg)
    assert type(got) is type(want), t
    assert _hyper(got) == _hyper(want), t


def test_hand_table_empty_union_holds() -> None:
    from comfy_research.engine.optimizers.optimizer_builders import OPTIMIZER_BUILDERS, optimizer_builder_node_types

    assert OPTIMIZER_BUILDERS == {}
    assert optimizer_builder_node_types() == frozenset(OPTIMIZER_TYPES)
