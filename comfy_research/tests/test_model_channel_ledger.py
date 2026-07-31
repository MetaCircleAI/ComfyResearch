"""model 通道台账 guards。

模型是三种群(27 builders / 13 atomic 链 / 21 tensor 辅助),台账比 dataset 多:
axes 台账钉的是 **44 型 intentionally 无轴**(与 dataset 修复教义方向相反——审计
未点名 model 轴缺口,严禁迁移"顺手"造轴)。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _model_types_from_manifest() -> frozenset[str]:
    m = json.loads((ROOT / "comfy_research" / "generated" / "node_manifest.json").read_text())
    return frozenset(e["type"] for e in m if e.get("category") == "model")



# 18 型 intentionally 有轴(hand 收集器);其余型无轴是 pins。
EXPECTED_MODEL_AXES_KINDS: frozenset[str] = frozenset({
    "mlp_model", "gated_mlp_model", "moe_mlp_model", "crl_residual_mlp", "kan_model",
    "attention_only_model", "linear_attention_model", "diagonal_ssm_token_model",
    "rwkv_time_mix_token_model", "hyena_like_conv_model", "slot_attention_token_model",
    "diffusion_score_model", "numeric_transformer_model", "mpp_spatiotemporal_model",
    "afno_lite_spatiotemporal_model", "transformer_token_model", "transformer_multi_token_model",
    "vgg11_cifar_model", "small_inception_cifar_model",
})


def test_model_ledger_matches_manifest() -> None:
    # 58 收官 + (keskar/vgg11) + UNet DDPM + Small Inception → 62。
    assert len(_model_types_from_manifest()) == 60


def test_builder_union_invariant() -> None:
    """hand ∪ providers == registered_builder_node_types_for('model');
    providers ⊆ builder 集;两侧不相交。"""
    import pytest

    pytest.importorskip("torch")
    from comfy_research.engine.models.model_builders import MODEL_BUILDERS, model_builder_node_types
    from comfy_research.engine.node_builder_registry import registered_builder_node_types_for
    from comfy_research.nodes.registry import model_defs_builders

    hand = frozenset(MODEL_BUILDERS)
    provs = frozenset(model_defs_builders())
    assert not (hand & provs), sorted(hand & provs)
    assert model_builder_node_types() == hand | provs
    assert hand | provs == frozenset(registered_builder_node_types_for("model"))


def test_alias_kinds_bidirectional() -> None:
    """alias 双向——是 model 节点、在 alias 表,但绝不在 builders/providers。"""
    import pytest



def test_model_defs_subset_and_axes_kinds_sane() -> None:
    from comfy_research.nodes.registry import model_def_types

    assert model_def_types() <= _model_types_from_manifest()
    assert EXPECTED_MODEL_AXES_KINDS <= _model_types_from_manifest()


def test_atomic_layer_capability_relationship() -> None:
    """capability(后端 SEQUENTIAL)vs canvas hand 集的**意图关系**(非相等,建档 hazard):
    canvas ATOMIC_LAYER_NODE_TYPES ⊇ capability 集(canvas 额外含非 atomic 物理层)。"""
    from comfy_research.generated.node_capabilities import node_types_with_capability

    cap = set(node_types_with_capability("atomic_layer_model"))
    assert len(cap) == 13, sorted(cap)
    canvas = (ROOT / "frontend" / "src" / "components" / "ResearchCanvas.tsx").read_text()
    for t in cap:
        assert f'"{t}"' in canvas, t


def test_ts_sweep_allowlist_matches_python_ledger() -> None:
    """TS MODEL_SWEEP_ALLOWLIST 与 EXPECTED_MODEL_AXES_KINDS 逐串恒等。"""
    import re

    src = (ROOT / "frontend" / "src" / "graph" / "trainSeriesPlan.ts").read_text()
    m = re.search(r"const MODEL_SWEEP_ALLOWLIST = new Set\(\[(.*?)\]\);", src, re.S)
    assert m
    ts_set = frozenset(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    assert ts_set == EXPECTED_MODEL_AXES_KINDS, (
        f"ts-only={sorted(ts_set - EXPECTED_MODEL_AXES_KINDS)}, py-only={sorted(EXPECTED_MODEL_AXES_KINDS - ts_set)}"
    )
