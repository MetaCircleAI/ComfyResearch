/** Shape 支持子集 guards + rms_norm shape 传播修复回归。 */
import { describe, expect, it } from "vitest";
import type { Edge, Node } from "@xyflow/react";

import { AFNO_SHAPE_UNSUPPORTED, SHAPE_ATOMIC_LAYER_TYPES, SHAPE_FULL_MODEL_TYPES } from "../canvasShapeSupport";
import { runFakeTensorShapeCheck } from "../fakeTensorShapePropagation";
import { nodeRegistryTypesWithFamily } from "../nodeRegistrySpec";

describe("shape support subsets", () => {
  it("atomic shape set == atomic family minus the four afno layers (9, incl the rms_norm fix)", () => {
    const family = new Set(nodeRegistryTypesWithFamily("atomic_layer_model"));
    expect([...SHAPE_ATOMIC_LAYER_TYPES].sort()).toEqual(
      [...family].filter((t) => !t.startsWith("afno_")).sort(),
    );
    expect(SHAPE_ATOMIC_LAYER_TYPES.has("rms_norm_layer")).toBe(true);
    expect(SHAPE_ATOMIC_LAYER_TYPES.size).toBe(9);
  });

  it("full-model shape set is a strict capability subset of the 28 canvas models (intent relation)", () => {
    const full = new Set(
      nodeRegistryTypesWithFamily("canvas_trainer_model_source").filter(
        (t) => !new Set(nodeRegistryTypesWithFamily("atomic_layer_model")).has(t) && t !== "combined_model",
      ),
    );
    for (const t of SHAPE_FULL_MODEL_TYPES) expect(full.has(t), t).toBe(true);
    expect([...SHAPE_FULL_MODEL_TYPES].sort()).toEqual([
      "attention_only_model", "diagonal_ssm_token_model", "diffusion_score_model", "gated_mlp_model",
      "hyena_like_conv_model", "kan_model", "linear_attention_model", "mlp_model", "moe_mlp_model",
      "numeric_hyena_model", "numeric_transformer_model", "residual_ln_model", "rwkv_time_mix_token_model",
      "slot_attention_token_model", "transformer_multi_token_model", "transformer_token_model",
    ]);
    expect([...AFNO_SHAPE_UNSUPPORTED].sort()).toEqual([
      "afno_encoder_block_layer", "afno_patch_decode_layer", "afno_patch_embed_layer", "afno_spectral_mixer_layer",
    ]);
  });

  it("rms_norm_layer participates in shape propagation (declared fix - was absent from the eligibility set)", () => {
    const N = (id: string, type: string, data: Record<string, unknown>) =>
      ({ id, type, position: { x: 0, y: 0 }, data }) as unknown as Node;
    const nodes = [
      N("f", "fake_tensor", { shape: [2, 64], dtype: "float" }),
      N("r", "rms_norm_layer", { normalizedShape: 64, ioMode: "input-output" }),
      N("bad", "rms_norm_layer", { normalizedShape: 32, ioMode: "input-output" }),
    ];
    const edges = [
      { id: "e0", source: "f", target: "r", sourceHandle: "tensor", targetHandle: "tensor_in" },
      { id: "e1", source: "f", target: "bad", sourceHandle: "tensor", targetHandle: "tensor_in" },
    ] as unknown as Edge[];
    const res = runFakeTensorShapeCheck("f", nodes, edges);
    expect(res.errorNodeIds).toContain("bad");   // 尾维 32≠64 → 报错(规则生效)
    expect(res.errorNodeIds).not.toContain("r"); // 64 匹配 → 传播通过
  });
});
