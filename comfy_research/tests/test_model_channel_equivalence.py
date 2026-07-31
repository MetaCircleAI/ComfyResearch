"""Model builder provider equivalence tests using state_dict SHA-256 hashes."""
from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("torch")

import numpy as np


def _sd_sha(module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def test_all_model_defs_match_committed_manifest() -> None:
    """Every model definition matches the committed manifest, including key order."""
    import json
    from pathlib import Path

    from comfy_research.nodes import registry
    from comfy_research.nodes.generate import def_to_entry_model

    registry.load_definitions()
    committed = {e["type"]: e for e in json.loads(
        (Path(__file__).resolve().parents[2] / "comfy_research" / "generated" / "node_manifest.json").read_text()
    )}
    assert registry.MODEL_DEFS, "no model defs registered"
    for t, d in sorted(registry.MODEL_DEFS.items()):
        got = def_to_entry_model(d)
        want = committed[t]
        assert got == want, t
        assert list(got) == list(want), t


SEQUENCE_MODEL_TYPES = ("attention_only_model", "linear_attention_model", "diagonal_ssm_token_model",
                        "rwkv_time_mix_token_model", "hyena_like_conv_model", "slot_attention_token_model")
TRANSFORMER_MODEL_TYPES = ("transformer_token_model", "transformer_multi_token_model")
# vector/numeric 五型(*_from_canvas_md 系读 md 全键;kan 走 context dims)。
VECTOR_MODEL_DATA = {
    "numeric_transformer_model": {"contextLength": 2, "inputDim": 2, "outputDim": 1, "modelDim": 8,
                                  "numHeads": 1, "numLayers": 1, "ffDim": 16, "seed": 42},
    "numeric_hyena_model": {"contextLength": 4, "inputDim": 2, "outputDim": 2, "modelDim": 16,
                            "depth": 1, "convKernel": 3, "ffMult": 2, "seed": 42},
    "mpp_spatiotemporal_model": {"contextFrames": 2, "channels": 1, "gridSize": 8, "inputDim": 128,
                                 "outputDim": 128, "patchSize": 4, "embedDim": 16, "depth": 1,
                                 "numHeads": 2, "ffRatio": 2, "seed": 42},
    "afno_lite_spatiotemporal_model": {"contextFrames": 2, "channels": 1, "gridSize": 8, "inputDim": 128,
                                       "outputDim": 128, "patchSize": 4, "embedDim": 16, "depth": 1,
                                       "numHeads": 2, "ffRatio": 2, "numSpectralBlocks": 1,
                                       "maxFrequencyModes": 2, "seed": 42},
    "kan_model": {"depth": 2, "width": 4, "grid": 3, "k": 3, "baseFun": "silu", "seed": 42},
}
# diffusion/residual_ln/crl 的 provider 覆盖。
SPECIAL_MODEL_DATA = {
    "diffusion_score_model": {"inputDim": 4, "hiddenDim": 16, "depth": 2, "timeEmbedDim": 8,
                              "diffusionTimesteps": 10, "seed": 42},
    "residual_ln_model": {"dim": 8, "depth": 3, "alpha": 1.0, "lnMode": "pre_ln", "activation": "relu", "seed": 42},
    "crl_residual_mlp": {"stateDim": 4, "actionDim": 2, "goalDim": 2, "actorWidth": 8, "criticWidth": 8,
                         "actorDepth": 4, "criticDepth": 4, "embedDim": 8, "activation": "silu", "seed": 42},
}
# vision 二型消费 context(input_channels/num_classes;vit 另需 image_size)。
VISION_MODEL_DATA = {
    "resnet_model": {"variant": "resnet18", "baseChannels": 8, "blocksStage1": 1, "blocksStage2": 1,
                     "blocksStage3": 1, "blocksStage4": 1, "kernelSize": 3, "seed": 42},
    "vit_model": {"variant": "tiny", "patchSize": 4, "hiddenDim": 16, "depth": 1, "numHeads": 2, "seed": 42},
}


@pytest.mark.parametrize("t", ["mlp_model", "gated_mlp_model", "moe_mlp_model", "mlp_token_model", "gated_mlp_token_model", "moe_mlp_token_model", *SEQUENCE_MODEL_TYPES, *TRANSFORMER_MODEL_TYPES, *VECTOR_MODEL_DATA, *VISION_MODEL_DATA, *SPECIAL_MODEL_DATA])
def test_builder_provider_full_path_equivalence(t: str) -> None:
    """经 build_model_from_type 全路径(generated-first hook)构建 ≡ engine 函数直调
    (同 seed → state_dict 逐张量 sha256 相等)。"""
    import torch

    from comfy_research.engine.models import model_builders as mb
    from comfy_research.engine.models.model_builders import ModelBuildContext, build_model_from_type

    if t in VECTOR_MODEL_DATA:
        data = dict(VECTOR_MODEL_DATA[t])
    elif t in VISION_MODEL_DATA:
        data = dict(VISION_MODEL_DATA[t])
    elif t in SPECIAL_MODEL_DATA:
        data = dict(SPECIAL_MODEL_DATA[t])
    elif t in TRANSFORMER_MODEL_TYPES:
        data = {"vocabSize": 12, "contextLength": 4, "modelDim": 8, "numHeads": 1, "numLayers": 1, "ffDim": 16, "seed": 42}
    elif t in SEQUENCE_MODEL_TYPES:
        data = {"vocabSize": 12, "embedDim": 8, "contextLength": 4, "seed": 42}
    elif "token" in t:
        data = {"vocabSize": 12, "embedDim": 8, "tokensPerInput": 1, "depth": 2, "width": 8, "numExperts": 2, "seed": 42}
    else:
        data = {"inputDim": 6, "outputDim": 2, "depth": 2, "width": 8, "seed": 42}
        if t == "moe_mlp_model":
            data["numExperts"] = 2
    if t in VISION_MODEL_DATA:
        ctx = ModelBuildContext(input_channels=1, num_classes=3, image_size=8)
    else:
        ctx = ModelBuildContext(input_dim=6, output_dim=2)
    torch.manual_seed(0)
    got = build_model_from_type(t, dict(data), ctx)
    fn = getattr(mb, f"_build_{t}")
    torch.manual_seed(0)
    want = fn(dict(data), ctx)
    assert _sd_sha(got) == _sd_sha(want), t


def test_b1_types_absent_from_hand_table_present_in_union() -> None:
    from comfy_research.engine.models.model_builders import MODEL_BUILDERS, model_builder_node_types

    for t in ("mlp_model", "gated_mlp_model", "moe_mlp_model",
              "mlp_token_model", "gated_mlp_token_model", "moe_mlp_token_model"):
        assert t not in MODEL_BUILDERS, t
        assert t in model_builder_node_types(), t
